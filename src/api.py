"""HTTP API прогнозу (FastAPI). Рахує лише через src.service (той самий predict, що й скрізь).

  uvicorn src.api:app --port 8000

GET /health, GET /regions, GET /forecast, GET /forecast/{region}, GET /quality.
Регіон — 'Область|Район' (URL-кодований), або назва області для м. Київ / Запорізької / Тернопільської.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data import EXCLUDED_OBLASTS, ROOT
from src.features import WESTERN_OBLASTS
from src.service import STALE_AFTER, ExcludedRegion, Forecaster, UnknownRegion, live_forecaster

DISCLAIMER = ("Це статистичний прогноз, а не офіційне оповіщення і не гарантія безпеки. "
              "Орієнтуйтесь на офіційні сигнали.")
TARGET = "Ймовірність, що в регіоні почнеться НОВА повітряна тривога протягом H годин після as_of."
SOURCE_NOTE = ("Живі дані: alerts.in.ua (червоний і жовтий рівні). Модель навчена на офіційному каналі, "
               "визначення тривоги відрізняються.")
CACHE_S = 30
WORKER_STATUS = ROOT / "data" / "live" / "worker_status.json"


def _iso(t):
    return None if t is None else pd.Timestamp(t).isoformat()


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def _data_block(snap: dict) -> dict:
    stale = snap["stale"]
    return {"requested_for": _iso(snap["requested_for"]), "as_of": _iso(snap["as_of"]),
            "data_last_update": _iso(snap["data_last_update"]), "data_age_s": _num(snap["data_age_s"]),
            "stale": stale, "stale_after_s": STALE_AFTER.total_seconds(),
            "warning": (f"Дані застаріли: прогноз станом на {_iso(snap['as_of'])}, а не на зараз."
                        if stale else None)}


def _item(r) -> dict:
    oblast, raion = r.unit.split("|", 1)
    return {"region": r.unit, "oblast": oblast, "raion": None if raion == "*" else raion,
            "model": r.model, "active_now": bool(r.active),
            "probabilities": {str(h): float(getattr(r, f"p_{h}")) for h in (1, 3, 6)}}


def create_app(forecaster: Forecaster | None = None) -> FastAPI:
    # Публічно — лише прогнози, якість і health: без /docs, /redoc, /openapi.json.
    app = FastAPI(title="Прогноз повітряних тривог", description=DISCLAIMER,
                  docs_url=None, redoc_url=None, openapi_url=None)
    # Сайт (Vercel) живе на іншому домені. Дані публічні й лише для читання: GET без credentials.
    origins = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"],
                       allow_credentials=False, allow_headers=[])
    state = {"fc": forecaster, "cache": None, "lock": threading.Lock()}

    def fc() -> Forecaster:
        if state["fc"] is None:
            state["fc"] = live_forecaster()
        return state["fc"]

    def snapshot(now=None) -> dict:
        """Один розрахунок усіх одиниць на хвилину/оновлення даних (кеш CACHE_S)."""
        now = now or pd.Timestamp.now(tz="UTC")
        with state["lock"]:
            c = state["cache"]
            last = fc().last_update_fn()
            if c and c["last"] == last and time.monotonic() - c["at"] < CACHE_S:
                snap = dict(c["snap"])
            else:
                snap = fc().snapshot(now)
                state["cache"] = {"snap": snap, "last": last, "at": time.monotonic()}
            # вік рахуємо щодо реального "зараз", навіть якщо таблиця з кешу
            if snap["data_last_update"] is not None:
                snap["data_age_s"] = (now - snap["data_last_update"]).total_seconds()
                snap["stale"] = snap["data_age_s"] > STALE_AFTER.total_seconds()
            snap["requested_for"] = now
            return snap

    @app.get("/health")
    def health():
        f = fc()
        as_of, last, age, stale = f.status(pd.Timestamp.now(tz="UTC"))
        worker = None
        if WORKER_STATUS.exists():
            w = json.loads(WORKER_STATUS.read_text())
            # Лише безпечні поля: тексти помилок можуть містити адреси й шляхи — віддаємо тільки кількість.
            worker = {"last_poll": w.get("last_poll"), "last_archive": w.get("last_archive"),
                      "archive_running": w.get("archive_running"), "recent_errors": len(w.get("errors") or [])}
        meta = f.meta
        return {"status": "degraded" if stale else "ok",
                "data": {"last_update": _iso(last), "age_s": _num(age), "stale": stale},
                "worker": worker,
                "model": {"trained_at": meta.get("trained_at"), "train_range": meta.get("train_range"),
                          "code_version": meta.get("code_version"), "units": len(f.units)}}

    @app.get("/regions")
    def regions():
        f = fc()
        items = []
        for u in f.list_regions():
            o, r = u.split("|", 1)
            items.append({"region": u, "oblast": o, "raion": None if r == "*" else r,
                          "model": "climate" if o in WESTERN_OBLASTS else "gbm"})
        excluded = [{"oblast": o, "reason": "прогнозу немає: постійна тривога / окупація / райони не розрізняються"}
                    for o in sorted(EXCLUDED_OBLASTS)]
        return {"regions": items, "excluded": excluded}

    @app.get("/forecast")
    def forecast_all():
        snap = snapshot()
        return {"disclaimer": DISCLAIMER, "target": TARGET, "source_note": SOURCE_NOTE,
                **_data_block(snap), "items": [_item(r) for r in snap["table"].itertuples()]}

    @app.get("/forecast/{region:path}")
    def forecast_one(region: str):
        try:
            unit = fc().resolve(region)
        except ExcludedRegion as e:
            raise HTTPException(404, detail={"error": "excluded", "message": str(e)})
        except UnknownRegion as e:
            raise HTTPException(404, detail={"error": "unknown", "message": str(e)})
        snap = snapshot()
        row = snap["table"].set_index("unit").loc[[unit]].reset_index().itertuples()
        return {"disclaimer": DISCLAIMER, "target": TARGET, "source_note": SOURCE_NOTE,
                **_data_block(snap), **_item(next(row))}

    @app.get("/quality")
    def quality():
        """Якість на test (01–28.08.2026, офіційний канал; оцінено двічі) і окремо зовнішня перевірка."""
        M = pd.read_csv(ROOT / "reports" / "metrics.csv")
        out = {}
        for part in ("test", "external"):
            rows = []
            for group, model in (("western", "climate"), ("non_western", "gbm")):
                for h in (1, 3, 6):
                    x = M[(M.split == part) & (M.group == group) & (M.model == model) & (M.horizon_h == h)]
                    if len(x):
                        r = x.iloc[0]
                        rows.append({"group": group, "model": model, "horizon_h": h, "brier": r.brier,
                                     "brier_climate": float(M[(M.split == part) & (M.group == group) &
                                                              (M.model == "climate") & (M.horizon_h == h)].brier.iloc[0]),
                                     "roc_auc": _num(r.roc_auc), "base_rate": r.base_rate, "n": int(r.n)})
            out[part] = rows
        return {"note": ("test — 01–28.08.2026 (офіційний канал, оцінено двічі); external — 22.09–15.10.2026 "
                         "(alerts.in.ua, інше визначення тривоги, не test). Захід — клімат, решта — GBM."),
                **out}

    return app


app = create_app()
