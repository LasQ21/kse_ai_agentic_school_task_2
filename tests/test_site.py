"""Сайт: JSON прогнозу (без моделей і мережі) і відповідність мапи одиницям моделі."""
import json
import threading
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path

import pandas as pd
import pytest

from src.service import Forecast
from src.site_api import build_payload, unit_json

ROOT = Path(__file__).resolve().parents[1]
T = pd.Timestamp("2026-10-16 12:00", tz="UTC")


def forecast(region="Київська область|Бучанський район", state="A", mode="gbm"):
    probs = {1: 0.0123, 3: 0.2, 6: 0.41, 24: 0.9}
    counts = {"p0": {h: 1 - p for h, p in probs.items()}, "p1": {h: p * 0.8 for h, p in probs.items()},
              "p2plus": {h: p * 0.2 for h, p in probs.items()}}
    return Forecast(region=region, requested_for=T, as_of=T, probabilities=probs, active_now=state == "B",
                    model=mode, data_last_update=T, data_age_s=30.0, stale=False, state=state,
                    elapsed_h=1.5 if state == "B" else None, mode=mode, counts=counts)


class FakeCache:
    def get(self):
        fs = {f.region: f for f in (forecast("Б|*", "B", "climate"), forecast("А|Район"))}
        return fs, {"as_of": T, "data_last_update": T, "data_age_s": 30.0, "stale": False}


def test_unit_json_fields_and_order():
    u = unit_json(forecast(state="B"))
    assert u["state"] == "B" and u["elapsed_h"] == 1.5 and not u["climate"]
    assert u["p"] == [0.0123, 0.2, 0.41, 0.9]          # порядок 1 / 3 / 6 / 24 год
    assert set(u["counts"]) == {"p0", "p1", "p2plus"} and len(u["counts"]["p0"]) == 4
    assert unit_json(forecast(mode="climate"))["climate"] is True


def test_payload_is_sorted_and_json_serialisable():
    d = build_payload(FakeCache())
    assert d["horizons"] == [1, 3, 6, 24] and not d["stale"]
    assert [u["id"] for u in d["units"]] == sorted(u["id"] for u in d["units"])
    assert json.loads(json.dumps(d)) == d
    assert d["as_of"].endswith("+00:00")


def test_endpoint_returns_503_without_details_when_source_fails(monkeypatch):
    import api.forecast as f

    class Broken:
        def get(self):
            raise RuntimeError("секретний рядок підключення")

    monkeypatch.setattr(f, "CACHE", Broken())
    srv = HTTPServer(("127.0.0.1", 0), f.handler)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    c = HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    c.request("GET", "/api/forecast")
    r = c.getresponse()
    body = r.read().decode()
    c.close()
    srv.server_close()
    assert r.status == 503 and "секретний" not in body and r.getheader("Cache-Control") == "no-store"


def test_map_covers_every_model_unit():
    meta, mp = ROOT / "models" / "service_model_meta.json", ROOT / "public" / "geo" / "map.json"
    if not meta.exists():
        pytest.skip("models/ немає (чистий клон): перевірка мапи потребує списку одиниць моделі")
    units = set(json.loads(meta.read_text())["units"])
    drawn = {u["id"] for u in json.loads(mp.read_text())["units"]}
    assert drawn == units


def _get(handler, path):
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    c = HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    out = r.status, r.getheader("Content-Type"), r.read().decode()
    c.close()
    srv.server_close()
    return out


def test_index_router_serves_forecast_and_keeps_webhook_health(monkeypatch):
    import api.forecast as f
    import api.index as idx

    monkeypatch.setattr(f, "CACHE", FakeCache())
    status, ctype, body = _get(idx.handler, "/api/forecast?x=1")
    assert status == 200 and ctype.startswith("application/json") and len(json.loads(body)["units"]) == 2
    assert _get(idx.handler, "/api/webhook")[::2] == (200, "ok")


def test_index_router_forecast_failure_is_503_not_crash(monkeypatch):
    import api.forecast as f
    import api.index as idx

    class Broken:
        def get(self):
            raise RuntimeError("x")

    monkeypatch.setattr(f, "CACHE", Broken())
    assert _get(idx.handler, "/api/forecast")[0] == 503
