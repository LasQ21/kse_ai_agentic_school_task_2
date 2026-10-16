"""Знімок для сайту: перевірка на секрети справді відкидає погане, а наявний snapshot.json чистий."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_snapshot import OUT, check  # noqa: E402


@pytest.mark.parametrize("bad", ['{"p": "/Users/x/data/live/alerts.sqlite"}', '{"t": "Bearer abc"}',
                                 '{"k": "1234567890:AAFj6NbXjbbAa3LyWVVlPaNm0Sstz0SYtqZ"}',
                                 '{"j": "eyJhbGciOiJIUzI1NiJ9.e30.x"}', '{"h": "' + "a" * 46 + '"}'])
def test_check_rejects_secrets_and_paths(bad):
    with pytest.raises(SystemExit):
        check(bad)


def test_published_snapshot_is_clean_and_complete():
    if not OUT.exists():
        pytest.skip("site/public/snapshot.json ще не згенеровано")
    text = OUT.read_text()
    check(text)
    s = json.loads(text)
    assert s["kind"] == "snapshot" and s["units"] == len(s["items"]) == 105
    assert {"snapshot_at_utc", "snapshot_at_kyiv", "data_age_s", "active_count", "excluded", "quality"} <= set(s)
    assert {e["oblast"] for e in s["excluded"]} == {"Донецька область", "Луганська область",
                                                    "Автономна Республіка Крим", "м. Севастополь"}
    for it in s["items"]:
        assert set(it) == {"region", "oblast", "raion", "model", "active_now", "probabilities"}
        p = it["probabilities"]
        assert 0 <= p["1"] <= p["3"] <= p["6"] <= 1
