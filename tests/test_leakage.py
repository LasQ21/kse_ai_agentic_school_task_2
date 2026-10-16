"""Тести на витік майбутнього в ознаки та на межі розбиття.

Ознаки перевіряються на синтетичних подіях (тести не залежать від data/, яка в .gitignore).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import COUNT_WINDOWS_H, CAP_H, OBLAST_ADJACENCY, compute_features  # noqa: E402
from src.split import EMBARGO, HORIZONS, SPLITS, _ts  # noqa: E402

UNITS = ["Київська область|Бучанський район", "Київська область|Вишгородський район",
         "Житомирська область|Коростенський район", "Львівська область|Львівський район"]
T0 = pd.Timestamp("2026-03-01", tz="UTC")


def synthetic_events(seed=0, n=400):
    rng = np.random.default_rng(seed)
    rows = []
    for u in UNITS:
        t = T0 - pd.Timedelta(days=10)
        for _ in range(n // len(UNITS)):
            t += pd.Timedelta(minutes=int(rng.integers(20, 900)))
            dur = pd.Timedelta(minutes=int(rng.integers(1, 300)))
            rows.append((u, t, t + dur))
            t += dur
    ev = pd.DataFrame(rows, columns=["unit", "started_at", "finished_at"])
    ev["oblast"] = ev.unit.str.split("|").str[0]
    return ev


EMPTY_HROMADA = pd.DataFrame({"unit": pd.Series(dtype=str),
                              "started_at": pd.Series(dtype="datetime64[ns, UTC]"),
                              "finished_at": pd.Series(dtype="datetime64[ns, UTC]")})
FEATURE_COLS = None
TIMES = pd.date_range(T0 + pd.Timedelta(days=2), periods=6, freq="17h").floor("h")


def features_at(ev, t):
    F = compute_features(ev, EMPTY_HROMADA, UNITS, pd.DatetimeIndex([t]))
    return F.drop(columns=["t"]).set_index("unit").sort_index()


@pytest.mark.parametrize("t", list(TIMES))
def test_future_does_not_change_features(t):
    """(а) Видалення / зміна всіх подій після t не змінює ознак у момент t."""
    ev = synthetic_events()
    base = features_at(ev, t)

    past = ev[ev.started_at < t].copy()
    # Кінець активних у t тривог невідомий -> міняємо його на довільний майбутній.
    still_open = past.finished_at >= t
    past.loc[still_open, "finished_at"] = t + pd.Timedelta(days=3)
    fake_future = pd.DataFrame({"unit": UNITS, "oblast": [u.split("|")[0] for u in UNITS],
                                "started_at": t + pd.Timedelta(minutes=1),
                                "finished_at": t + pd.Timedelta(hours=2)})
    for variant in (past, pd.concat([past, fake_future], ignore_index=True)):
        pd.testing.assert_frame_equal(base, features_at(variant, t), check_dtype=False)


def reference_features(ev, unit, t):
    """(б) Еталон, що бачить лише дані < t; явно перевіряє max використаного timestamp."""
    e = ev[(ev.unit == unit) & (ev.started_at < t)]
    known_end = e.finished_at.where(e.finished_at < t)
    used = pd.concat([e.started_at, known_end.dropna()])
    assert used.empty or used.max() < t
    H = pd.Timedelta(hours=1)
    out = {"active": bool(known_end.isna().any())}
    for w in COUNT_WINDOWS_H:
        out[f"starts_{w}h"] = int((e.started_at >= t - w * H).sum())
    out["h_since_last_start"] = min((t - e.started_at.max()) / H, CAP_H) if len(e) else CAP_H
    ke = known_end.dropna()
    out["h_since_last_end"] = min((t - ke.max()) / H, CAP_H) if len(ke) else CAP_H
    return out


@pytest.mark.parametrize("t", list(TIMES))
def test_vectorized_matches_past_only_reference(t):
    ev = synthetic_events(seed=1)
    F = features_at(ev, t)
    for u in UNITS:
        ref = reference_features(ev, u, t)
        for k, v in ref.items():
            assert F.loc[u, k] == pytest.approx(v), (u, k)


def test_aggregates_exclude_self():
    ev = synthetic_events(seed=2)
    t = TIMES[3]
    F = features_at(ev, t)
    assert (F.country_active_other == F.active.sum() - F.active).all()


@pytest.mark.parametrize("name", list(SPLITS))
def test_split_boundaries_with_embargo(name):
    """(в) Межі train/val/test не перетинаються з урахуванням embargo і горизонту міток."""
    s = SPLITS[name]
    hmax = pd.Timedelta(hours=max(HORIZONS))
    tr, va, te = ([_ts(a), _ts(b)] for a, b in (s.train, s.val, s.test))
    assert tr[0] < tr[1] <= va[0] < va[1] <= te[0] < te[1]
    # Останнє t частини (права межа виключна, крок 1 год) + H_max <= перше t наступної.
    for (a0, a1), (b0, _) in ((tr, va), (va, te)):
        last_t = a1 - pd.Timedelta(hours=1)
        assert last_t + hmax <= b0
        assert b0 - a1 + pd.Timedelta(hours=1) >= EMBARGO


def test_split_on_built_labels(artifacts):
    """Мітки з пайплайна (зразок або справжні): embargo між частинами, сегменти, валідність."""
    L = pd.read_parquet(artifacts.dataset, columns=["t", "segment", "valid", "split_V", "y_1", "y_3", "y_6"])
    s = L.split_V
    hmax = pd.Timedelta(hours=max(HORIZONS))
    parts = [x for x in ("train", "val", "test") if (s == x).any()]
    assert parts[:2] == ["train", "val"]
    for a, b in zip(parts, parts[1:]):
        assert L.t[s == a].max() + hmax < L.t[s == b].min()
    if (s == "external").any():
        assert set(L.segment[s == "external"]) == {"alerts_in_ua"}
    assert set(L.segment[s.isin(["train", "val", "test"])]) == {"vadimkin"}
    assert not L.loc[s.notna(), "valid"].eq(False).any()
    assert ((L.y_1 <= L.y_3) & (L.y_3 <= L.y_6)).all()


def test_adjacency_symmetric_and_connected():
    for a, bs in OBLAST_ADJACENCY.items():
        assert a not in bs
        for b in bs:
            assert a in OBLAST_ADJACENCY[b], (a, b)
    seen, stack = set(), [next(iter(OBLAST_ADJACENCY))]
    while stack:
        x = stack.pop()
        if x not in seen:
            seen.add(x)
            stack += OBLAST_ADJACENCY[x]
    assert seen == set(OBLAST_ADJACENCY)


def test_unit_without_events_gets_defaults():
    """Одиниця без жодної події (типово у короткому вікні сервісу) не ламає ознаки."""
    ev = synthetic_events(seed=3)
    ev = ev[ev.unit != UNITS[0]]
    F = features_at(ev, TIMES[2])
    row = F.loc[UNITS[0]]
    assert not row.active and row.starts_168h == 0 and row.h_since_last_start == CAP_H


def test_oblast_alert_expands_to_static_raion_list():
    """Обласна тривога розгортається на всі райони з переліку моделі, навіть без їхніх подій у вікні."""
    from src.data import clean_events, raions_from_units
    t0 = pd.Timestamp("2026-10-01 10:00", tz="UTC")
    df = pd.DataFrame({"oblast": ["Миколаївська область"], "raion": [None], "hromada": [None],
                       "level": ["oblast"], "started_at": [t0], "finished_at": [t0 + pd.Timedelta(hours=1)],
                       "source": ["alerts_in_ua"], "censored": [False]})
    units = ["Миколаївська область|Баштанський район", "Миколаївська область|Миколаївський район",
             "Запорізька область|*"]
    ev, rep = clean_events(df, t0 - pd.Timedelta(days=1), t0 + pd.Timedelta(days=1), raions_from_units(units))
    assert sorted(ev.unit) == units[:2]
    assert rep["rows_added_by_oblast_expansion"] == 1
