"""Цілі для бота (нові; старі мітки y_1/y_3/y_6 і SPLITS не змінюються).

Для кожної одиниці й години t, у будь-якому стані:
- y_new_h — у (t, t + h] почалася нова тривога, h ∈ {1, 3, 6, 24} (для 1/3/6 це те саме, що
  старі y_h; якщо тривога йде в t, нова можлива лише після відбою);
- n_new_h — кількість нових тривог (стартів після злиття перерв ≤ 60 с) у (t, t + h].
Стан у t (A — тривоги немає, B — тривога йде) і пройдений час — для показу й ознак; кінець тривоги
після t в ознаки не потрапляє.

Відкидаються (з лічильниками в звіті):
- хвіст сегмента: t + 24 год виходить за кінець надійних даних;
- рядки, чиє вікно (t, t + 24 год] зачіпає «завислу» тривогу (> 96 год, suspect_end);
- стан B, якщо епізод «завислий» (> 96 год) або його кінець невідомий (обрізаний межею даних):
  тоді невідомо, коли міг початися наступний.

Розбиття SPLIT_STATE має embargo 97 год між частинами: вікно мітки (≤ 24 год) не заходить у
наступну частину, і жоден епізод (≤ 96 год після відкидання завислих) не може бути активним і в
останньому t однієї частини, і в першому t наступної.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data import PROCESSED_DIR, STUCK_ALERT_H
from src.split import WARMUP, SplitSpec, _ns, _ts, assign_split

STATE_HORIZONS = (1, 3, 6, 24)
STATE_HMAX = pd.Timedelta(hours=max(STATE_HORIZONS))
MAX_EPISODE = pd.Timedelta(hours=STUCK_ALERT_H)          # довші — «завислі», відкидаються
EMBARGO_STATE = pd.Timedelta(hours=97)                    # > MAX_EPISODE і > STATE_HMAX
assert EMBARGO_STATE > MAX_EPISODE and EMBARGO_STATE > STATE_HMAX

# Межі — моменти t (UTC), ліва включно, права виключно. Test — той самий період, що й для старих
# цілей (його вже бачили двічі для y_1/y_3/y_6), обрізаний так, щоб t + 24 год ≤ 2026-08-29.
SPLIT_STATE = SplitSpec("S", ("2025-11-17", "2026-06-11"), ("2026-06-15", "2026-07-28"),
                        ("2026-08-01", "2026-08-28T01:00"), "vadimkin",
                        "embargo 97 год (найдовший епізод 96 год + 1 год); test — період старого test")


def build_state_labels(events: pd.DataFrame, coverage: dict) -> tuple[pd.DataFrame, dict]:
    """Сітка (одиниця × година) по сегментах coverage з мітками стану A і B.

    Повертає таблицю (unit, t, segment, state, elapsed_h, episode_id, y_start_*, y_cont_*, keep,
    drop_reason) і лічильники відкинутого.
    """
    H = pd.Timedelta(hours=1).value
    suspect = pd.DataFrame(coverage["suspect_end_intervals"])
    frames, report = [], {"tail_rows_dropped": 0}
    for seg in coverage["segments"]:
        s0, s1 = _ts(seg["start"]), _ts(seg["end"])
        grid = pd.date_range((s0 + WARMUP).ceil("h"), (s1 - STATE_HMAX).floor("h"), freq="h").as_unit("ns")
        old_grid_end = (s1 - pd.Timedelta(hours=6)).floor("h")           # межа старої сітки (H_max = 6)
        report["tail_rows_dropped"] += int(((old_grid_end - grid[-1]) / pd.Timedelta(hours=1))) * len(coverage["units"])
        t = grid.asi8
        ev = events[events.source == seg["source"]]
        for unit in coverage["units"]:
            e = ev[ev.unit == unit].sort_values("started_at")
            st, fn = _ns(e.started_at), _ns(e.finished_at)
            cens, stuck = e.censored.to_numpy(bool), e.suspect_end.to_numpy(bool)
            # Стан — як ознака `active` у src.features: тривога, що почалася до t і не закінчилася
            # до t (кінець ≥ t ще невідомий у момент t).
            k = np.searchsorted(st, t, side="left") - 1                  # останній старт < t
            kk = np.clip(k, 0, None)
            end_k = np.maximum.accumulate(fn)[kk] if len(st) else np.zeros(len(t), "int64")  # кінець епізоду
            active = (k >= 0) & (end_k >= t)
            row = {"unit": unit, "t": grid, "segment": seg["source"],
                   "state": np.where(active, "B", "A"),
                   "elapsed_h": np.where(active, (t - st[kk]) / H, np.nan) if len(st) else np.full(len(t), np.nan),
                   "episode_id": np.where(active, kk, -1)}
            for h in STATE_HORIZONS:
                hv = pd.Timedelta(hours=h).value
                n = np.searchsorted(st, t + hv, side="right") - np.searchsorted(st, t, side="right")
                row[f"n_new_{h}"] = n
                row[f"y_new_{h}"] = (n > 0).astype(int)
            reason = np.full(len(t), "", dtype=object)
            if len(suspect):
                for r in suspect[suspect.unit == unit].itertuples():
                    a, b = _ts(r.start).value, _ts(r.end).value
                    reason[(t + STATE_HMAX.value >= a) & (t <= b)] = "suspect_window"
            if len(st):
                reason[(reason == "") & active & stuck[kk]] = "stuck_episode_gt_96h"
                reason[(reason == "") & active & cens[kk]] = "episode_end_unknown"
            row["drop_reason"] = reason
            frames.append(pd.DataFrame(row))
    L = pd.concat(frames, ignore_index=True)
    L["keep"] = L.drop_reason == ""
    L["valid"] = L.keep                                   # для assign_split
    L["split_S"] = assign_split(L, SPLIT_STATE)
    report["rows_in_grid"] = len(L)
    report["dropped_by_reason"] = L.drop_reason[~L.keep].value_counts().to_dict()
    report["dropped_by_reason_and_state"] = {f"{r}|{s}": int(n) for (r, s), n in
                                             L[~L.keep].groupby(["drop_reason", "state"]).size().items()}
    return L.drop(columns="valid"), report


def check_state_embargo(L: pd.DataFrame) -> dict:
    """Розрахунок embargo на побудованих мітках: кінець вікна мітки і кінець найдовшого епізоду
    останнього t кожної частини проти першого t наступної."""
    s = L.split_S
    out = {}
    for a, b in (("train", "val"), ("val", "test")):
        last_t, first_next = L.t[s == a].max(), L.t[s == b].min()
        B = L[(s == a) & (L.state == "B")]
        longest = pd.Timedelta(hours=float((B.elapsed_h.max() or 0))) if len(B) else pd.Timedelta(0)
        out[f"{a}->{b}"] = {
            "last_t": str(last_t), "next_first_t": str(first_next),
            "gap_h": (first_next - last_t) / pd.Timedelta(hours=1),
            "label_window_end": str(last_t + STATE_HMAX),
            "label_window_inside_next": bool(last_t + STATE_HMAX >= first_next),
            "max_kept_episode_h": MAX_EPISODE / pd.Timedelta(hours=1),
            "episode_could_span": bool(last_t + MAX_EPISODE >= first_next),
            "max_elapsed_h_in_part_B_rows": longest / pd.Timedelta(hours=1),
        }
    return out


def build_and_save() -> tuple[pd.DataFrame, dict]:
    """events.parquet + coverage.json -> data/processed/state_labels.parquet + reports/state_labels.json."""
    from src.data import ROOT
    events = pd.read_parquet(PROCESSED_DIR / "events.parquet")
    coverage = json.loads((PROCESSED_DIR / "coverage.json").read_text())
    L, rep = build_state_labels(events, coverage)
    rep["embargo"] = check_state_embargo(L)
    k = L[L.keep]
    rep["rows_by_split_state"] = {f"{p}|{st}": int(n) for (p, st), n in
                                  k.groupby(["split_S", "state"]).size().items()}
    rep["episodes_by_split"] = {p: int(g[g.state == "B"][["segment", "unit", "episode_id"]].drop_duplicates().shape[0])
                                for p, g in k.groupby("split_S")}
    L.to_parquet(PROCESSED_DIR / "state_labels.parquet", index=False)
    (ROOT / "reports" / "state_labels.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    return L, rep
