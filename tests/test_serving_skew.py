"""Training/serving skew: predict() на історичний момент t дає ті самі ознаки й прогнози,
що й офлайн-таблиця, з якої рахувалися метрики. Моменти беруться лише з validation.

Проганяється на синтетичному зразку (tests/sample_data.py) і, якщо вони є, на справжніх data/ і models/.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES_GBM
from src.models import service_predict

N_TIMES = 12


@pytest.fixture(scope="module")
def setup(artifacts):
    from src.service import Forecaster
    D = pd.read_parquet(artifacts.dataset)
    val = D[D.split_V == "val"]
    times = pd.Series(val.t.unique()).sample(N_TIMES, random_state=0).sort_values().tolist()
    raw = artifacts.raw()
    bundle = artifacts.bundle()
    return D, times, raw, bundle, Forecaster(lambda: raw, bundle, state_bundle=artifacts.state_bundle())


def test_features_and_predictions_match_offline(setup):
    D, times, raw, bundle, fc = setup
    for t in times:
        live = fc.feature_frame(t).set_index("unit").sort_index()
        off = D[D.t == t].set_index("unit").sort_index()
        assert list(live.index) == list(off.index)
        for f in FEATURES_GBM + ["active"]:
            a, b = live[f].to_numpy(float), off[f].to_numpy(float)
            assert np.allclose(a, b, equal_nan=True), (t, f, np.flatnonzero(~np.isclose(a, b, equal_nan=True))[:5])
        p_live, _ = service_predict(live.reset_index(), bundle["models"])
        p_off, _ = service_predict(off.reset_index(), bundle["models"])
        for h in p_live:
            assert np.allclose(p_live[h], p_off[h], atol=1e-12), (t, h)


def test_predict_api_matches_predict_all(setup):
    D, times, raw, bundle, fc = setup
    t = times[0]
    allp = fc.predict_all(t).set_index("unit")
    for unit in ["Харківська область|Харківський район", "Львівська область|Львівський район", "м. Київ|*"]:
        p = fc.predict(unit, t)
        assert p == {h: pytest.approx(allp.loc[unit, f"p_any_{h}"]) for h in (1, 3, 6, 24)}
        assert p[1] <= p[3] <= p[6] <= p[24]
        old = allp.loc[unit, ["p_1", "p_3", "p_6"]].to_numpy(float)
        assert np.array_equal(old, allp.loc[unit, ["p_any_1", "p_any_3", "p_any_6"]].to_numpy(float))
    assert fc.forecast("Львівська область|Львівський район", t).model == "climate"
    assert fc.forecast("Харківська область|Харківський район", t).model == "gbm"


def test_future_events_do_not_change_prediction(setup):
    D, times, raw, bundle, fc = setup
    from src.service import Forecaster
    t = times[1]
    past = raw[raw.started_at < t].copy()
    # Кінці тривог, активних у t, невідомі: замінюємо їх на довільні майбутні.
    open_at_t = past.finished_at >= t
    altered = past.copy()
    altered.loc[open_at_t, "finished_at"] = t + pd.Timedelta(days=2)
    # Вигадані майбутні тривоги в кожному районі через хвилину після t.
    fake = raw[raw.level == "raion"].drop_duplicates(["oblast", "raion"]).copy()
    fake["started_at"], fake["finished_at"] = t + pd.Timedelta(minutes=1), t + pd.Timedelta(hours=1)
    base = fc.predict_all(t)
    for variant in (past, pd.concat([altered, fake], ignore_index=True)):
        pd.testing.assert_frame_equal(base, Forecaster(lambda v=variant: v, bundle, state_bundle=fc.state).predict_all(t))


def test_excluded_and_unknown_regions(setup):
    from src.service import ExcludedRegion, UnknownRegion
    fc = setup[4]
    with pytest.raises(ExcludedRegion):
        fc.resolve("Донецька область")
    with pytest.raises(UnknownRegion):
        fc.resolve("Неіснуюча область")
    assert fc.resolve("Запорізька область") == "Запорізька область|*"


@pytest.fixture(scope="module")
def state_setup(artifacts):
    from src.service import Forecaster
    X = pd.read_parquet(artifacts.dataset_state)
    times = pd.Series(X[X.split_S == "val"].t.unique()).sample(N_TIMES, random_state=1).sort_values().tolist()
    raw = artifacts.raw()
    S = artifacts.state_bundle()
    return X, times, S, artifacts.bundle()["models"], Forecaster(lambda: raw, artifacts.bundle(), state_bundle=S)


def test_state_features_and_predictions_match_offline(state_setup):
    """Ознаки бота й прогнози (P(≥1) на 1–24 год, лічильник) сервісу = офлайн-таблиця dataset_state."""
    from src.state_models import FEATURES_Q_GBM, bot_predict
    X, times, S, old, fc = state_setup
    n_b = 0
    for t in times:
        live = fc.feature_frame(t).set_index("unit").sort_index()
        off = X[X.t == t].set_index("unit").sort_index()
        assert list(live.index) == list(off.index)
        assert ((off.state == "B") == (live.active == 1)).all()
        n_b += int((off.state == "B").sum())
        for f in FEATURES_Q_GBM:
            a, b = live[f].to_numpy(float), off[f].to_numpy(float)
            assert np.allclose(a, b, equal_nan=True), (t, f)
        p_off = bot_predict(off.reset_index(), S, old)
        allp = fc.predict_all(t).set_index("unit").sort_index()
        for h in (1, 3, 6, 24):
            for k, col in (("any", "p_any"), ("p0", "p0"), ("p1", "p1"), ("p2plus", "p2plus")):
                assert np.allclose(allp[f"{col}_{h}"], p_off[k][h], atol=1e-12), (t, h, k)
    assert n_b > 0
