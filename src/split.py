"""Сітка міток (одиниця × година) і розбиття за часом.

Мітка y_H(u, t) = 1, якщо в одиниці u починається тривога в (t, t+H].
Рядок валідний, лише якщо (t, t + max(H)] повністю всередині сегмента покриття, t не раніше
start сегмента + WARMUP (щоб 168-годинні ознаки не були занижені після діри) і вікно
[t, t + max(H)] не перетинає suspect_end-інтервал цієї одиниці.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import PROCESSED_DIR

HORIZONS = (1, 3, 6)
WARMUP = pd.Timedelta(days=7)
EMBARGO = pd.Timedelta(hours=24)   # >= max(HORIZONS)


def _ts(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return (t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")).as_unit("ns")


def _ns(col: pd.Series) -> np.ndarray:
    """int64 наносекунди. Явно, бо pandas 3 за замовчуванням тримає час у мікросекундах,
    а Timedelta.value — завжди в наносекундах (змішування дає вікна в 1000 разів довші)."""
    return col.dt.as_unit("ns").astype("int64").to_numpy()


def build_label_grid(events: pd.DataFrame, coverage: dict,
                     horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    hmax = pd.Timedelta(hours=max(horizons))
    suspect = pd.DataFrame(coverage["suspect_end_intervals"])
    frames = []
    for seg in coverage["segments"]:
        s0, s1 = _ts(seg["start"]), _ts(seg["end"])
        grid = pd.date_range((s0 + WARMUP).ceil("h"), (s1 - hmax).floor("h"), freq="h").as_unit("ns")
        ev = events[(events.source == seg["source"])]
        for unit in coverage["units"]:
            e = ev[ev.unit == unit]
            # Старти до початку сегмента не можуть потрапити в (t, t+H], бо t >= s0 + WARMUP.
            starts = np.sort(_ns(e.started_at))
            t = grid.asi8
            row = {"unit": unit, "t": grid, "segment": seg["source"]}
            for h in horizons:
                hi = np.searchsorted(starts, t + pd.Timedelta(hours=h).value, side="right")
                lo = np.searchsorted(starts, t, side="right")
                row[f"y_{h}"] = (hi - lo) > 0
            # Активна в момент t: почалася до t і не закінчилася до t (відомо в момент t).
            st, fn = _ns(e.started_at), _ns(e.finished_at)
            order = np.argsort(st)
            st, fn = st[order], np.maximum.accumulate(fn[order]) if len(fn) else fn
            k = np.searchsorted(st, t, side="left") - 1
            row["active_at_t"] = (k >= 0) & (fn[np.clip(k, 0, None)] > t) if len(st) else np.zeros(len(t), bool)
            valid = np.ones(len(t), bool)
            if len(suspect):
                for r in suspect[suspect.unit == unit].itertuples():
                    a, b = _ts(r.start).value, _ts(r.end).value
                    valid &= ~((t + hmax.value >= a) & (t <= b))
            row["valid"] = valid
            frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    train: tuple[str, str]
    val: tuple[str, str]
    test: tuple[str, str]
    test_segment: str
    note: str


# Межі — моменти t (UTC), ліва включно, права виключно. Між частинами >= EMBARGO.
SPLITS = {
    # Рекомендований: одне визначення тривоги (офіційний канал) у train/val/test.
    "V": SplitSpec("V", ("2025-11-17", "2026-06-14"), ("2026-06-15", "2026-07-31"),
                   ("2026-08-01", "2026-08-28T19:00"), "vadimkin",
                   "test = останні 4 тижні узгоджених офіційних даних; alerts.in.ua — зовнішня перевірка"),
    # Варіант А: test з alerts.in.ua (інше визначення тривоги, див. звіт п. 7).
    "A": SplitSpec("A", ("2025-11-17", "2026-06-30"), ("2026-07-01", "2026-08-28T19:00"),
                   ("2026-09-22T20:00", "2026-10-15T14:00"), "alerts_in_ua",
                   "test з іншого джерела; метрики змішують якість моделі і зсув визначення"),
}
BACKTEST_ORIGINS = ["2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01"]
BACKTEST_WINDOW = pd.Timedelta(days=28)


def assign_split(labels: pd.DataFrame, spec: SplitSpec) -> pd.Series:
    """'train' / 'val' / 'test' / 'external' / None. Невалідні рядки -> None."""
    out = pd.Series(None, index=labels.index, dtype="object")
    t = labels.t
    for part in ("train", "val", "test"):
        a, b = (_ts(x) for x in getattr(spec, part))
        seg = spec.test_segment if part == "test" else "vadimkin"
        out[(t >= a) & (t < b) & (labels.segment == seg)] = part
    other = "alerts_in_ua" if spec.test_segment == "vadimkin" else None
    if other:
        out[labels.segment == other] = "external"
    out[~labels.valid] = None
    return out


def check_embargo(labels: pd.DataFrame, split: pd.Series, horizons=HORIZONS) -> dict:
    """Перевіряє, що вікна міток (t, t+H] попередньої частини не заходять у наступну."""
    hmax = pd.Timedelta(hours=max(horizons))
    res = {}
    for a, b in (("train", "val"), ("val", "test")):
        last_label_end = labels.t[split == a].max() + hmax
        first_next = labels.t[split == b].min()
        res[f"{a}->{b}"] = {"last_label_end": str(last_label_end), "next_first_t": str(first_next),
                            "gap_h": (first_next - last_label_end).total_seconds() / 3600}
    return res


def backtest_folds(labels: pd.DataFrame, spec: SplitSpec):
    """Rolling-origin: train = усе до origin - EMBARGO, eval = [origin, origin + 28 д). Не заходить у test."""
    test_start = _ts(spec.test[0])
    base = labels.valid & (labels.segment == "vadimkin")
    for o in BACKTEST_ORIGINS:
        o = _ts(o)
        end = min(o + BACKTEST_WINDOW, test_start - EMBARGO)
        yield o, base & (labels.t < o - EMBARGO), base & (labels.t >= o) & (labels.t < end)


def build_and_save() -> pd.DataFrame:
    events = pd.read_parquet(PROCESSED_DIR / "events.parquet")
    coverage = json.loads((PROCESSED_DIR / "coverage.json").read_text())
    labels = build_label_grid(events, coverage)
    for name, spec in SPLITS.items():
        labels[f"split_{name}"] = assign_split(labels, spec)
    labels.to_parquet(PROCESSED_DIR / "labels.parquet", index=False)
    return labels
