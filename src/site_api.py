"""JSON для сайту: прогноз усіх одиниць одним об'єктом. Рахує лише через src.service (через ForecastCache бота).

Ймовірності віддаються як є (4 знаки); округлення для показу робить сайт тим самим правилом, що й бот
(src.bot.fmt_p). Поля p — у порядку HORIZONS_BOT, counts — P(0) / P(1) / P(2+) нових тривог.
"""
from __future__ import annotations

import pandas as pd

from src.bot import ForecastCache
from src.service import HORIZONS_BOT, Forecast


def _iso(t) -> str | None:
    return None if t is None else pd.Timestamp(t).tz_convert("UTC").isoformat()


def unit_json(f: Forecast) -> dict:
    out = {
        "id": f.region,
        "state": f.state,                       # "A" — тривоги немає, "B" — тривога йде
        "elapsed_h": None if f.elapsed_h is None else round(f.elapsed_h, 3),
        "climate": f.mode == "climate",         # захід: модель не перевершила клімат
        "p": [round(f.probabilities[h], 4) for h in HORIZONS_BOT],
    }
    if f.counts:
        out["counts"] = {k: [round(f.counts[k][h], 4) for h in HORIZONS_BOT] for k in ("p0", "p1", "p2plus")}
    return out


def build_payload(cache: ForecastCache) -> dict:
    by_unit, status = cache.get()
    return {
        "horizons": list(HORIZONS_BOT),
        "as_of": _iso(status["as_of"]),
        "data_last_update": _iso(status["data_last_update"]),
        "data_age_s": status["data_age_s"],
        "stale": bool(status["stale"]),
        "generated_at": _iso(pd.Timestamp.now(tz="UTC")),
        "units": [unit_json(f) for f in sorted(by_unit.values(), key=lambda f: f.region)],
    }
