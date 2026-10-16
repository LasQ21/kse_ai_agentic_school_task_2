"""Витік для цілей бота (src/state_targets.py, src/state_features.py).

- truncation: ознаки в t однакові на повних і обрізаних у t даних (600 випадкових t);
- ознаки активної тривоги не бачать її кінця після t;
- мітки (стан, y_new_h, n_new_h) суворо по (t, t + h]: перевірка «в лоб» циклом;
- мутації: протічка в ознаках чи неправильне вікно мітки провалюють ці перевірки;
- embargo SPLIT_STATE покриває H = 24 і найдовший епізод.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import add_derived, compute_features
from src.split import _ts
from src.state_features import STATE_FEATURES, _durations_per_unit, add_state_features
from src.state_models import counter_from
from src.state_targets import (EMBARGO_STATE, MAX_EPISODE, SPLIT_STATE, STATE_HMAX, STATE_HORIZONS,
                               build_state_labels)
import src.state_features as sf

UNITS = ["Київська область|Бучанський район", "Київська область|Вишгородський район",
         "Житомирська область|Коростенський район", "Львівська область|Львівський район"]
T0 = pd.Timestamp("2026-03-01", tz="UTC")
EMPTY_HROMADA = pd.DataFrame({"unit": pd.Series(dtype=str),
                              "started_at": pd.Series(dtype="datetime64[us, UTC]"),
                              "finished_at": pd.Series(dtype="datetime64[us, UTC]")})


def synthetic_events(seed=0, days=40):
    """Неперетинні тривоги по одиницях: паузи 20 хв – 20 год, тривалості 5 хв – 30 год."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in UNITS:
        t = T0 - pd.Timedelta(days=5)
        while t < T0 + pd.Timedelta(days=days):
            t += pd.Timedelta(minutes=int(rng.integers(20, 1200)))
            dur = pd.Timedelta(minutes=int(rng.choice([rng.integers(5, 240), rng.integers(240, 1800)], p=[.9, .1])))
            rows.append((u, t, t + dur))
            t += dur
    ev = pd.DataFrame(rows, columns=["unit", "started_at", "finished_at"])
    ev["oblast"] = ev.unit.str.split("|").str[0]
    ev["suspect_end"] = False
    ev["censored"] = False
    ev["source"] = "vadimkin"
    for c in ("started_at", "finished_at"):
        ev[c] = ev[c].dt.as_unit("us")
    return ev


def state_features_at(ev, t):
    F = add_derived(compute_features(ev, EMPTY_HROMADA, UNITS, pd.DatetimeIndex([t])), UNITS)
    F = add_state_features(F, ev)
    return F.set_index("unit").sort_index()[["active", "active_elapsed_h", *STATE_FEATURES]]


def truncate_at(ev, t, rng):
    """Що відомо в момент t: лише старти < t; кінці ≥ t замінено довільними майбутніми;
    додано вигадані майбутні тривоги."""
    past = ev[ev.started_at < t].copy()
    open_ = past.finished_at >= t
    past.loc[open_, "finished_at"] = t + pd.Timedelta(minutes=int(rng.integers(1, 5000)))
    fake = pd.DataFrame({"unit": UNITS, "oblast": [u.split("|")[0] for u in UNITS],
                         "started_at": t + pd.Timedelta(minutes=2), "finished_at": t + pd.Timedelta(hours=40),
                         "suspect_end": False, "censored": False, "source": "vadimkin"})
    fake[["started_at", "finished_at"]] = fake[["started_at", "finished_at"]].apply(lambda c: c.dt.as_unit("us"))
    return pd.concat([past, fake], ignore_index=True)


def truncation_mismatches(n_times=600, seed=0):
    ev = synthetic_events(seed)
    rng = np.random.default_rng(seed)
    span = int((pd.Timedelta(days=35)) / pd.Timedelta(minutes=1))
    times = [T0 + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, span, n_times)]
    bad, n_active = 0, 0
    for t in times:
        a, b = state_features_at(ev, t), state_features_at(truncate_at(ev, t, rng), t)
        n_active += int(a.active.sum())
        bad += int((~np.isclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)).any(axis=1).sum())
    return bad, n_active


def test_truncation_600_random_times():
    bad, n_active = truncation_mismatches()
    assert n_active > 200          # серед перевірених чимало моментів зі станом B
    assert bad == 0


def test_active_alert_end_after_t_is_invisible():
    ev = synthetic_events(seed=3)
    for t in pd.date_range(T0 + pd.Timedelta(days=3), periods=40, freq="19h"):
        base = state_features_at(ev, t)
        open_ = (ev.started_at < t) & (ev.finished_at >= t)
        for new_end in (t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(days=4)):
            alt = ev.copy()
            alt.loc[open_, "finished_at"] = pd.Timestamp(new_end).as_unit("us")
            pd.testing.assert_frame_equal(base, state_features_at(alt, t))


def coverage_for(ev, days=30):
    return {"segments": [{"source": "vadimkin", "start": T0.isoformat(),
                          "end": (T0 + pd.Timedelta(days=days)).isoformat()}],
            "suspect_end_intervals": [], "units": UNITS}


def brute_force_mismatches(L, ev, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    bad = 0
    for i in rng.choice(len(L), size=min(n, len(L)), replace=False):
        r = L.iloc[i]
        e = ev[ev.unit == r.unit]
        known = e[e.started_at < r.t]
        on = known[known.finished_at >= r.t]
        state = "B" if len(on) else "A"
        bad += state != r.state
        for h in STATE_HORIZONS:
            end = r.t + pd.Timedelta(hours=h)
            n = int(((e.started_at > r.t) & (e.started_at <= end)).sum())
            bad += (n != r[f"n_new_{h}"]) + (int(n > 0) != r[f"y_new_{h}"])
    return bad


def test_labels_brute_force():
    ev = synthetic_events(seed=4)
    L, _ = build_state_labels(ev, coverage_for(ev))
    assert (L.state == "B").sum() > 500 and (L.n_new_6 >= 2).sum() > 20 and (L.n_new_24 >= 2).sum() > 500
    assert brute_force_mismatches(L, ev) == 0


def test_mutation_leaky_feature_is_caught(monkeypatch):
    """Протічка: тривалості з тривог, що закінчаться протягом години після t."""
    real = _durations_per_unit

    def leaky(starts, ends, t):
        return real(starts, ends, t + 3600 * 10**9)
    monkeypatch.setattr(sf, "_durations_per_unit", leaky)
    bad, _ = truncation_mismatches(n_times=150, seed=1)
    assert bad > 0


def test_mutation_wrong_label_window_is_caught():
    """Мутація: мітка з неправильним вікном (для H = 3 / 6 / 24 підставлено вікно H = 1 / 3 / 6)."""
    ev = synthetic_events(seed=5)
    L, _ = build_state_labels(ev, coverage_for(ev))
    M = L.copy()
    for h, wrong in ((3, 1), (6, 3), (24, 6)):
        for c in ("y_new", "n_new"):
            M[f"{c}_{h}"] = L[f"{c}_{wrong}"]
    assert brute_force_mismatches(M, ev) > 0


def test_state_embargo_covers_h24_and_longest_episode():
    s = SPLIT_STATE
    for (a0, a1), (b0, _) in ((s.train, s.val), (s.val, s.test)):
        last_t = _ts(a1) - pd.Timedelta(hours=1)
        assert _ts(b0) - last_t >= EMBARGO_STATE
        assert last_t + STATE_HMAX < _ts(b0)
        assert last_t + MAX_EPISODE < _ts(b0)


def test_stuck_and_unknown_end_are_dropped():
    ev = synthetic_events(seed=6)
    cov = coverage_for(ev, days=20)
    end = T0 + pd.Timedelta(days=20)
    # епізод із невідомим кінцем (обрізаний межею даних) і «завислий» епізод
    long_ = (ev.finished_at - ev.started_at) > pd.Timedelta(hours=3)
    i = ev.index[(ev.unit == UNITS[0]) & long_ & (ev.started_at > T0 + pd.Timedelta(days=8))][0]
    ev.loc[i, "censored"] = True
    j = ev.index[(ev.unit == UNITS[1]) & long_ & (ev.started_at > T0 + pd.Timedelta(days=8))][0]
    ev.loc[j, "suspect_end"] = True
    L, rep = build_state_labels(ev, cov)
    assert L.t.max() + STATE_HMAX <= end
    for idx, reason in ((i, "episode_end_unknown"), (j, "stuck_episode_gt_96h")):
        r = ev.loc[idx]
        rows = L[(L.unit == r.unit) & (L.t > r.started_at) & (L.t <= r.finished_at)]
        assert len(rows) and (rows.drop_reason == reason).all() and not rows.keep.any()
    assert rep["tail_rows_dropped"] > 0


def test_counter_is_consistent_and_monotone():
    """P(0) + P(1) + P(2+) = 1, P(0) = 1 − P(≥1), усі ≥ 0, P(2+) не спадає з h."""
    rng = np.random.default_rng(0)
    p = np.sort(rng.random((1000, 4)), axis=1)
    any_ = {h: p[:, i] for i, h in enumerate(STATE_HORIZONS)}
    q = {h: rng.random(1000) for h in STATE_HORIZONS}
    c = counter_from(any_, q)
    for i, h in enumerate(STATE_HORIZONS):
        tot = c["p0"][h] + c["p1"][h] + c["p2plus"][h]
        assert np.allclose(tot, 1) and np.allclose(c["p0"][h], 1 - any_[h])
        assert (c["p1"][h] >= -1e-12).all() and (c["p2plus"][h] >= 0).all()
        if i:
            assert (c["p2plus"][h] >= c["p2plus"][STATE_HORIZONS[i - 1]] - 1e-12).all()
