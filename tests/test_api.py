"""API (TestClient) на тимчасовому SQLite-сховищі: схема відповіді, застереження, застарілі дані,
виключені й невідомі регіони. Потребує models/service_model.joblib (інакше пропуск)."""
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.service import SERVICE_MODEL  # noqa: E402

pytestmark = pytest.mark.skipif(not SERVICE_MODEL.exists(), reason="немає models/service_model.joblib")
BUCHA = "Київська область|Бучанський район"


def make_client(tmp_path, poll_age: pd.Timedelta):
    from fastapi.testclient import TestClient
    from src.api import create_app
    from src.live import LiveStore
    from src.service import Forecaster
    now = pd.Timestamp.now(tz="UTC")
    s = LiveStore(tmp_path / "a.sqlite")
    alert = {"id": 1, "location_title": "Бучанський район", "location_type": "raion",
             "location_oblast": "Київська область", "location_raion": None, "alert_type": "air_raid",
             "alert_level": "red", "started_at": (now - poll_age - pd.Timedelta(minutes=10)).isoformat(),
             "finished_at": None}
    s.record_active("alerts_in_ua", [alert], (now - poll_age).isoformat())
    return TestClient(create_app(Forecaster(s.raw_events, last_update_fn=s.last_ok_poll))), now - poll_age


def test_fresh_data(tmp_path):
    from src.api import DISCLAIMER
    c, _ = make_client(tmp_path, pd.Timedelta(seconds=5))
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["data"]["stale"] is False and h["model"]["units"] == 105

    r = c.get("/regions").json()
    assert len(r["regions"]) == 105
    assert {"Донецька область", "Луганська область", "Автономна Республіка Крим", "м. Севастополь"} <= \
        {e["oblast"] for e in r["excluded"]}
    models = {x["region"]: x["model"] for x in r["regions"]}
    assert models["Львівська область|Львівський район"] == "climate" and models[BUCHA] == "gbm"

    a = c.get("/forecast").json()
    assert a["disclaimer"] == DISCLAIMER and a["stale"] is False and a["warning"] is None
    assert len(a["items"]) == 105
    for it in a["items"]:
        p = it["probabilities"]
        assert 0 <= p["1"] <= p["3"] <= p["6"] <= 1
    one = c.get("/forecast/" + quote(BUCHA)).json()
    assert one["region"] == BUCHA and one["active_now"] is True and one["model"] == "gbm"
    assert one["disclaimer"] == DISCLAIMER and "data_age_s" in one and "as_of" in one
    assert c.get("/forecast/" + quote("м. Київ")).json()["region"] == "м. Київ|*"


def test_stale_data_is_flagged(tmp_path):
    c, last = make_client(tmp_path, pd.Timedelta(hours=2))
    assert c.get("/health").json()["status"] == "degraded"
    one = c.get("/forecast/" + quote(BUCHA)).json()
    assert one["stale"] is True and one["warning"] and one["data_age_s"] >= 2 * 3600 - 5
    assert pd.Timestamp(one["as_of"]) == last          # станом на останнє оновлення, не на зараз


def test_excluded_and_unknown(tmp_path):
    c, _ = make_client(tmp_path, pd.Timedelta(seconds=5))
    r = c.get("/forecast/" + quote("Донецька область"))
    assert r.status_code == 404 and r.json()["detail"]["error"] == "excluded"
    r = c.get("/forecast/" + quote("Неіснуюча область|Район"))
    assert r.status_code == 404 and r.json()["detail"]["error"] == "unknown"


def test_quality_has_test_and_external_separately(tmp_path):
    c, _ = make_client(tmp_path, pd.Timedelta(seconds=5))
    q = c.get("/quality").json()
    assert {"test", "external"} <= set(q)
    assert {(x["group"], x["model"]) for x in q["test"]} == {("western", "climate"), ("non_western", "gbm")}
