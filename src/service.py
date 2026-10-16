"""Прогноз для сервісу: predict(region, now) -> {H: P(почнеться нова тривога в (now, now + H])}.

Стан у now (A — тривоги немає, B — тривога йде, з пройденим часом) повертається окремо; якщо
тривога йде, нова можлива лише після відбою. З моделлю для бота (models/state_models.joblib) —
H = 1 / 3 / 6 / 24 і лічильник P(0) / P(1) / P(2+) нових тривог; без неї — лише 1 / 3 / 6.
Використовує ті самі функції, що й навчання: src.data.clean_events / hromada_events,
src.features.compute_features / add_derived, src.state_features.add_state_features,
src.models.service_predict, src.state_models.bot_predict. На вхід — сирі події (EVENT_COLS) від
будь-якого провайдера; усе, що не раніше now, відкидається ще до ознак.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from src.data import (EXCLUDED_OBLASTS, KYIV_CITY, OBLAST_AS_UNIT, ROOT, clean_events,
                      hromada_events, raions_from_units)
from src.features import CAP_H, COUNT_WINDOWS_H, WESTERN_OBLASTS, add_derived, compute_features
from src.models import service_predict
from src.state_features import DUR_WINDOW, add_state_features
from src.state_models import bot_predict

SERVICE_MODEL = ROOT / "models" / "service_model.joblib"
STATE_MODEL = ROOT / "models" / "state_models.joblib"
HORIZONS_BOT = (1, 3, 6, 24)
# Історія має покривати найдовшу "пам'ять" ознак: обрізання h_since_* (CAP_H = 30 діб), а не лише
# вікна лічильників (168 год) — інакше в тихих районах сервіс бачить 720 замість реальних годин
# (знайдено тестом на skew).
LOOKBACK = pd.Timedelta(hours=max(CAP_H, *COUNT_WINDOWS_H) + 24)
assert LOOKBACK > DUR_WINDOW
STALE_AFTER = pd.Timedelta(minutes=10)


class UnknownRegion(ValueError):
    """Регіону немає серед одиниць моделі."""
    pass


class ExcludedRegion(ValueError):
    """Регіон свідомо виключено з моделі (EXCLUDED_OBLASTS)."""
    pass


class StaleDataError(RuntimeError):
    """Живі дані застаріли (полер зупинений / джерело відстало)."""


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def load_service_model(path: Path = SERVICE_MODEL) -> dict:
    # joblib = pickle: безпечно лише для власних артефактів (scripts/export_service_model.py).
    return joblib.load(path)


@dataclass
class Forecast:
    """Прогноз для однієї одиниці разом зі станом даних."""
    region: str
    requested_for: pd.Timestamp     # на коли просили
    as_of: pd.Timestamp             # станом на коли реально пораховано (= останнє оновлення, якщо stale)
    probabilities: dict[int, float]  # P(нова тривога почнеться в (as_of, as_of + H])
    active_now: bool                # активна станом на as_of
    model: str                      # стара модель старту 1/3/6: "gbm" або "climate"
    data_last_update: pd.Timestamp | None
    data_age_s: float | None        # requested_for - data_last_update
    stale: bool
    state: str = "A"                # "A" — тривоги немає, "B" — тривога йде
    elapsed_h: float | None = None  # скільки триває поточна тривога (лише B)
    mode: str = ""                  # режим моделі: "climate" (захід) або сімейства для 1–6 / 24 год
    counts: dict[str, dict[int, float]] | None = None   # {"p0" | "p1" | "p2plus": {H: p}}


class Forecaster:
    """raw_events_fn() -> сирі події (EVENT_COLS, можуть містити й майбутнє — воно відкидається)."""

    def __init__(self, raw_events_fn: Callable[[], pd.DataFrame], bundle: dict | None = None,
                 last_update_fn: Callable[[], pd.Timestamp | None] | None = None,
                 state_bundle: dict | None = None):
        self.raw_events_fn = raw_events_fn
        self.bundle = bundle or load_service_model()
        # Модель для бота (scripts/train_state.py). Без неї — лише старий прогноз 1/3/6.
        self.state = state_bundle or (load_service_model(STATE_MODEL) if STATE_MODEL.exists() else None)
        self.models, self.meta = self.bundle["models"], self.bundle["meta"]
        self.units = self.meta["units"]
        # Без last_update_fn — історичний режим (офлайн, тести): дані вважаються повними на now.
        self.live = last_update_fn is not None
        self.last_update_fn = last_update_fn or (lambda: None)

    def status(self, now) -> tuple[pd.Timestamp, pd.Timestamp | None, float | None, bool]:
        """(as_of, last_update, age_s, stale). Якщо дані застарілі, прогноз рахується станом на
        останнє оновлення: ми не знаємо, що сталося після, і не вдаємо, що знаємо."""
        now = _utc(now)
        if not self.live:
            return now, None, None, False
        last = self.last_update_fn()
        if last is None:
            return now, None, None, True
        last = _utc(last)
        age = (now - last).total_seconds()
        stale = age > STALE_AFTER.total_seconds()
        return (min(now, last) if stale else now), last, age, stale

    def resolve(self, region: str) -> str:
        """Назва району 'Область|Район', або область, де одна одиниця на всю область."""
        if region in self.units:
            return region
        if region in EXCLUDED_OBLASTS:
            raise ExcludedRegion(f"{region}: виключено з моделі (постійна тривога / окупація / немає різниці між районами)")
        if region in OBLAST_AS_UNIT | {KYIV_CITY}:
            return f"{region}|*"
        raise UnknownRegion(f"{region}: невідомий регіон; див. list_regions()")

    def list_regions(self) -> list[str]:
        """Одиниці моделі ('Область|Район' або 'Область|*')."""
        return list(self.units)

    def feature_frame(self, now: pd.Timestamp) -> pd.DataFrame:
        """Ознаки всіх одиниць у момент now — лише з подій, що почалися до now."""
        now = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
        raw = self.raw_events_fn()
        raw = raw[raw.started_at < now]
        ev, _ = clean_events(raw, now - LOOKBACK, now, raions_from_units(self.units))
        hrom = hromada_events(raw, now - LOOKBACK, now)
        F = compute_features(ev, hrom, self.units, pd.DatetimeIndex([now]))
        return add_state_features(add_derived(F, self.units), ev)

    def predict_all(self, now) -> pd.DataFrame:
        """Для всіх одиниць у момент now: p_1, p_3, p_6 і model (стара модель, без змін), state,
        elapsed_h; з моделлю для бота ще p_any_{1,3,6,24}, p0_/p1_/p2plus_{1,3,6,24}, mode."""
        F = self.feature_frame(now)
        preds, west = service_predict(F, self.models)
        out = F[["unit", "oblast", "t", "active"]].copy()
        for h, p in preds.items():
            out[f"p_{h}"] = p
        out["model"] = np.where(west, "climate", "gbm")
        b = F.active.to_numpy().astype(bool)
        out["state"] = np.where(b, "B", "A")
        out["elapsed_h"] = np.where(b, F.active_elapsed_h, np.nan)
        out["mode"] = out.model
        if self.state is not None:
            bp = bot_predict(F, self.state, self.models)
            for h in HORIZONS_BOT:
                out[f"p_any_{h}"] = bp["any"][h]
                for k in ("p0", "p1", "p2plus"):
                    out[f"{k}_{h}"] = bp[k][h]
            f24 = self.state["family_any24"]
            out["mode"] = np.where(west, "climate", "gbm" if f24 == "gbm" else f"gbm (1–6 год), {f24} (24 год)")
        return out

    def forecast(self, region: str, now) -> Forecast:
        """Прогноз для регіону; якщо дані застарілі — станом на останнє оновлення (stale=True)."""
        unit = self.resolve(region)
        as_of, last, age, stale = self.status(now)
        row = self.predict_all(as_of).set_index("unit").loc[unit]
        return self._forecast_from_row(unit, row, now, as_of, last, age, stale)

    def _forecast_from_row(self, unit, row, now, as_of, last, age, stale) -> Forecast:
        old = {h: float(row[f"p_{h}"]) for h in sorted(self.models)}
        f = Forecast(region=unit, requested_for=_utc(now), as_of=as_of, probabilities=old,
                     active_now=bool(row.active), model=row.model, data_last_update=last,
                     data_age_s=age, stale=stale, state=row.state, mode=row["mode"],
                     elapsed_h=None if row.state == "A" else float(row.elapsed_h))
        if self.state is not None:
            f.probabilities = {h: float(row[f"p_any_{h}"]) for h in HORIZONS_BOT}
            f.counts = {k: {h: float(row[f"{k}_{h}"]) for h in HORIZONS_BOT} for k in ("p0", "p1", "p2plus")}
        return f

    def forecast_many(self, now) -> tuple[list[Forecast], dict]:
        """Прогнози всіх одиниць одним розрахунком ознак + стан даних (для /top бота)."""
        as_of, last, age, stale = self.status(now)
        T = self.predict_all(as_of).set_index("unit")
        fs = [self._forecast_from_row(u, T.loc[u], now, as_of, last, age, stale) for u in T.index]
        return fs, {"as_of": as_of, "data_last_update": last, "data_age_s": age, "stale": stale}

    def predict(self, region: str, now, allow_stale: bool = False) -> dict[int, float]:
        """{H: p}; StaleDataError, якщо дані застарілі й allow_stale=False."""
        f = self.forecast(region, now)
        if f.stale and not allow_stale:
            age = "невідомий (опитувань не було)" if f.data_age_s is None else f"{f.data_age_s / 60:.0f} хв"
            raise StaleDataError(f"живі дані застаріли: вік {age}; прогноз був би станом на {f.as_of}. "
                                 "Запустіть воркер (scripts/live_worker.py) або передайте allow_stale=True")
        return f.probabilities


def live_forecaster(db_path=None) -> Forecaster:
    """Forecaster на живому SQLite-сховищі воркера (вік даних — за останнім опитуванням)."""
    from src.live import LIVE_DB, LiveStore
    store = LiveStore(db_path or LIVE_DB)
    return Forecaster(store.raw_events, last_update_fn=store.last_ok_poll)


def predict(region: str, now=None, allow_stale: bool = False) -> dict[int, float]:
    """Публічна функція: прогноз для регіону на момент now (за замовчуванням — зараз) з живих даних.

    Якщо живі дані старші за STALE_AFTER — StaleDataError (або прогноз станом на останнє
    оновлення з allow_stale=True; деталі — live_forecaster().forecast(...)).
    """
    now = now or pd.Timestamp.now(tz="UTC")
    return live_forecaster().predict(region, now, allow_stale=allow_stale)


__all__ = ["predict", "Forecaster", "Forecast", "StaleDataError", "live_forecaster", "WESTERN_OBLASTS"]
