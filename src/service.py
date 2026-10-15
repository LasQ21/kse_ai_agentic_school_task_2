"""Прогноз для сервісу: predict(region, now) -> {H: ймовірність}.

Використовує ті самі функції, що й навчання: src.data.clean_events / hromada_events,
src.features.compute_features / add_derived, src.models.service_predict. На вхід — сирі події
(EVENT_COLS) від будь-якого провайдера; усе, що не раніше now, відкидається ще до ознак.
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

SERVICE_MODEL = ROOT / "models" / "service_model.joblib"
# Історія має покривати найдовшу "пам'ять" ознак: обрізання h_since_* (CAP_H = 30 діб), а не лише
# вікна лічильників (168 год) — інакше в тихих районах сервіс бачить 720 замість реальних годин
# (знайдено тестом на skew).
LOOKBACK = pd.Timedelta(hours=max(CAP_H, *COUNT_WINDOWS_H) + 24)
STALE_AFTER = pd.Timedelta(minutes=10)


class UnknownRegion(ValueError):
    pass


class ExcludedRegion(ValueError):
    pass


def load_service_model(path: Path = SERVICE_MODEL) -> dict:
    # joblib = pickle: безпечно лише для власних артефактів (scripts/export_service_model.py).
    return joblib.load(path)


@dataclass
class Forecast:
    region: str
    computed_for: pd.Timestamp
    probabilities: dict[int, float]
    active_now: bool
    model: str               # "gbm" або "climate"
    data_last_update: pd.Timestamp | None
    data_age_s: float | None
    stale: bool


class Forecaster:
    """raw_events_fn() -> сирі події (EVENT_COLS, можуть містити й майбутнє — воно відкидається)."""

    def __init__(self, raw_events_fn: Callable[[], pd.DataFrame], bundle: dict | None = None,
                 last_update_fn: Callable[[], pd.Timestamp | None] | None = None):
        self.raw_events_fn = raw_events_fn
        self.bundle = bundle or load_service_model()
        self.models, self.meta = self.bundle["models"], self.bundle["meta"]
        self.units = self.meta["units"]
        self.last_update_fn = last_update_fn or (lambda: None)

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
        return list(self.units)

    def feature_frame(self, now: pd.Timestamp) -> pd.DataFrame:
        """Ознаки всіх одиниць у момент now — лише з подій, що почалися до now."""
        now = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
        raw = self.raw_events_fn()
        raw = raw[raw.started_at < now]
        ev, _ = clean_events(raw, now - LOOKBACK, now, raions_from_units(self.units))
        hrom = hromada_events(raw, now - LOOKBACK, now)
        F = compute_features(ev, hrom, self.units, pd.DatetimeIndex([now]))
        return add_derived(F, self.units)

    def predict_all(self, now) -> pd.DataFrame:
        F = self.feature_frame(now)
        preds, west = service_predict(F, self.models)
        out = F[["unit", "oblast", "t", "active"]].copy()
        for h, p in preds.items():
            out[f"p_{h}"] = p
        out["model"] = np.where(west, "climate", "gbm")
        return out

    def forecast(self, region: str, now) -> Forecast:
        unit = self.resolve(region)
        row = self.predict_all(now).set_index("unit").loc[unit]
        last = self.last_update_fn()
        now_ts = row.t
        age = (now_ts - last).total_seconds() if last is not None else None
        return Forecast(region=unit, computed_for=now_ts,
                        probabilities={h: float(row[f"p_{h}"]) for h in sorted(self.models)},
                        active_now=bool(row.active), model=row.model, data_last_update=last,
                        data_age_s=age, stale=age is None or age > STALE_AFTER.total_seconds())

    def predict(self, region: str, now) -> dict[int, float]:
        return self.forecast(region, now).probabilities


def live_forecaster(db_path=None) -> Forecaster:
    from src.live import LIVE_DB, LiveStore
    store = LiveStore(db_path or LIVE_DB)
    return Forecaster(store.raw_events, last_update_fn=store.last_ok_poll)


def predict(region: str, now=None) -> dict[int, float]:
    """Публічна функція: прогноз для регіону на момент now (за замовчуванням — зараз) з живих даних."""
    now = now or pd.Timestamp.now(tz="UTC")
    return live_forecaster().predict(region, now)


__all__ = ["predict", "Forecaster", "Forecast", "live_forecaster", "WESTERN_OBLASTS"]
