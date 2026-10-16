"""Живе сховище: відкриті тривоги активні до моменту прогнозу; закриття й повторне відкриття."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import compute_features  # noqa: E402
from src.live import LiveStore  # noqa: E402
from src.data import clean_events, hromada_events  # noqa: E402

UNIT = "Київська область|Бучанський район"


def alert(i, start, level="red"):
    return {"id": i, "location_title": "Бучанський район", "location_type": "raion",
            "location_oblast": "Київська область", "location_raion": None, "alert_type": "air_raid",
            "alert_level": level, "started_at": start, "finished_at": None}


def active_at(store, t):
    t = pd.Timestamp(t)
    raw = store.raw_events()
    ev, _ = clean_events(raw[raw.started_at < t], t - pd.Timedelta(days=31), t)
    hrom = hromada_events(raw[raw.started_at < t], t - pd.Timedelta(days=31), t)
    return bool(compute_features(ev, hrom, [UNIT], pd.DatetimeIndex([t])).active.iloc[0])


def test_open_alert_is_active_after_last_poll(tmp_path):
    s = LiveStore(tmp_path / "a.sqlite")
    s.record_active("alerts_in_ua", [alert(1, "2026-10-15T20:00:00Z")], "2026-10-15T20:05:00+00:00")
    assert active_at(s, "2026-10-15T20:30:00Z")          # опитування було о 20:05, кінець невідомий


def test_disappeared_alert_closed_and_flicker_reopens(tmp_path):
    s = LiveStore(tmp_path / "a.sqlite")
    s.record_active("alerts_in_ua", [alert(1, "2026-10-15T20:00:00Z")], "2026-10-15T20:05:00+00:00")
    s.record_active("alerts_in_ua", [], "2026-10-15T20:07:00+00:00")
    assert not active_at(s, "2026-10-15T20:10:00Z")
    s.record_active("alerts_in_ua", [alert(1, "2026-10-15T20:00:00Z")], "2026-10-15T20:09:00+00:00")
    assert active_at(s, "2026-10-15T20:10:00Z")


def test_history_gives_exact_end(tmp_path):
    s = LiveStore(tmp_path / "a.sqlite")
    s.record_active("alerts_in_ua", [alert(1, "2026-10-15T20:00:00Z")], "2026-10-15T20:05:00+00:00")
    s.record_active("alerts_in_ua", [], "2026-10-15T20:30:00+00:00")
    exact = dict(alert(1, "2026-10-15T20:00:00Z"), finished_at="2026-10-15T20:12:00Z")
    s.record_history("alerts_in_ua", [exact], "2026-10-15T20:40:00+00:00")
    assert s.raw_events().finished_at.iloc[0] == pd.Timestamp("2026-10-15T20:12:00Z")


def test_stale_data_is_reported_not_hidden(tmp_path):
    """Полер зупинився: прогноз станом на останнє оновлення, stale=True, predict() відмовляє."""
    import pytest
    from src.service import Forecaster, StaleDataError, SERVICE_MODEL
    if not SERVICE_MODEL.exists():
        pytest.skip("немає models/service_model.joblib")
    s = LiveStore(tmp_path / "a.sqlite")
    s.record_active("alerts_in_ua", [alert(1, "2026-10-15T20:00:00Z")], "2026-10-15T20:05:00+00:00")
    fc = Forecaster(s.raw_events, last_update_fn=s.last_ok_poll)
    now = pd.Timestamp("2026-10-15T23:05:00Z")
    f = fc.forecast(UNIT, now)
    assert f.stale and f.as_of == pd.Timestamp("2026-10-15T20:05:00+00:00") and f.data_age_s == 3 * 3600
    with pytest.raises(StaleDataError):
        fc.predict(UNIT, now)
    assert fc.predict(UNIT, now, allow_stale=True) == f.probabilities
    fresh = fc.forecast(UNIT, pd.Timestamp("2026-10-15T20:07:00Z"))
    assert not fresh.stale and fresh.as_of == pd.Timestamp("2026-10-15T20:07:00Z") and fresh.active_now
