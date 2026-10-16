"""Додаткові ознаки для цілей за станом (src/state_targets.py). Старі ознаки не змінюються.

Усе — лише з минулого: тривалості беруться тільки з тривог, що закінчилися до t
(finished_at < t), і лише за останні DUR_WINDOW, щоб сервіс з вікном історії LOOKBACK бачив те
саме, що й офлайн. «Завислі» тривоги (suspect_end) у тривалості не входять.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import _ns_series

DUR_WINDOW = pd.Timedelta(days=30)
LAST_K = 10
STATE_FEATURES = ["oblasts_active_now", "prev_dur_last_h", "prev_dur_med10_h", "prev_n_ended_30d",
                  "elapsed_over_med"]


def _durations_per_unit(starts: np.ndarray, ends: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """Для кожного t: тривалість останньої завершеної тривоги, медіана останніх ≤ LAST_K
    завершених і їхня кількість за DUR_WINDOW. starts/ends — ns."""
    H = 3600 * 10**9
    n = len(t)
    if len(ends) == 0:
        return {"prev_dur_last_h": np.full(n, np.nan), "prev_dur_med10_h": np.full(n, np.nan),
                "prev_n_ended_30d": np.zeros(n, int)}
    o = np.argsort(ends)
    fn, dur = ends[o], (ends[o] - starts[o]) / H
    m = np.searchsorted(fn, t, side="left")                     # кінці < t
    lo = np.searchsorted(fn, t - DUR_WINDOW.value, side="left")  # кінці ≥ t − DUR_WINDOW
    a = np.maximum(lo, m - LAST_K)
    cache: dict[tuple[int, int], float] = {}
    med = np.full(n, np.nan)
    for i, (x, y) in enumerate(zip(a, m)):
        if y > x:
            key = (int(x), int(y))
            if key not in cache:
                cache[key] = float(np.median(dur[x:y]))
            med[i] = cache[key]
    has = m > lo
    last = np.where(has, dur[np.clip(m - 1, 0, None)], np.nan)
    return {"prev_dur_last_h": last, "prev_dur_med10_h": med, "prev_n_ended_30d": (m - lo).astype(int)}


def add_state_features(F: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """F — вихід src.features.compute_features (+ add_derived); events — очищені події
    (src.data.clean_events). Повертає F з колонками STATE_FEATURES."""
    F = F.copy()
    ok = events[~events.suspect_end.astype(bool)] if "suspect_end" in events else events
    parts = []
    for u, g in F.groupby("unit", sort=False):
        e = ok[ok.unit == u]
        d = _durations_per_unit(_ns_series(e.started_at), _ns_series(e.finished_at), _ns_series(g.t))
        parts.append(pd.DataFrame(d, index=g.index))
    F = F.join(pd.concat(parts))
    act = F.active.astype(bool)
    F["oblasts_active_now"] = F.t.map(F[act].groupby("t").oblast.nunique()).fillna(0).astype(int)
    F["elapsed_over_med"] = np.where(act & (F.prev_dur_med10_h > 0),
                                     F.active_elapsed_h / F.prev_dur_med10_h, np.nan)
    return F


def build_state_dataset():
    """dataset.parquet (старі ознаки, як є) ∩ state_labels.parquet + STATE_FEATURES
    -> data/processed/dataset_state.parquet. Ознаки рахуються окремо по кожному сегменту."""
    from src.data import PROCESSED_DIR
    D = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    L = pd.read_parquet(PROCESSED_DIR / "state_labels.parquet")
    E = pd.read_parquet(PROCESSED_DIR / "events.parquet")
    keep_cols = [c for c in D.columns if not c.startswith("split_")] + ["split_V"]
    X = L.merge(D[keep_cols].drop(columns=["segment"]), on=["unit", "t"], how="left", validate="1:1",
                suffixes=("", "_old"))
    assert X.active.notna().all(), "нова сітка має бути підмножиною старої"
    parts = [add_state_features(g, E[E.source == seg]) for seg, g in X.groupby("segment")]
    X = pd.concat(parts).sort_index()
    X.to_parquet(PROCESSED_DIR / "dataset_state.parquet", index=False)
    return X
