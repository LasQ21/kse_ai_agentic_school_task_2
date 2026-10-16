"""Знімок прогнозу з локального API -> site/public/snapshot.json (статичний файл для сайту).

  python scripts/make_snapshot.py [--api http://127.0.0.1:8000]

Потрібні запущені воркер і API (scripts/local_stack.sh start). Поля копіюються за білим списком:
у файл не потрапляють шляхи, токени, сирі події чи стан воркера. Перед записом файл
перевіряється на заборонені рядки; при знахідці скрипт падає і нічого не пише.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "public" / "snapshot.json"
FORBIDDEN = [r"/Users/", r"/private/", r"/home/", r"\.sqlite", r"\.parquet", r"\.joblib", r"data/(raw|live|processed)",
             r"Bearer", r"token", r"eyJ[A-Za-z0-9_-]{10,}\.", r"[0-9]{8,10}:[A-Za-z0-9_-]{30,}", r"[0-9a-f]{40,}",
             r"Traceback", r"@gmail"]


def get(api, path):
    r = requests.get(api + path, timeout=30)
    r.raise_for_status()
    return r.json()


def build(api: str) -> dict:
    health, fc, regions, quality = (get(api, p) for p in ("/health", "/forecast", "/regions", "/quality"))
    now = datetime.now(timezone.utc)
    items = [{"region": i["region"], "oblast": i["oblast"], "raion": i["raion"], "model": i["model"],
              "active_now": bool(i["active_now"]),
              "probabilities": {h: round(float(i["probabilities"][h]), 4) for h in ("1", "3", "6")}}
             for i in fc["items"]]
    q = {part: [{k: r[k] for k in ("group", "model", "horizon_h", "brier", "brier_climate", "roc_auc", "base_rate", "n")}
                for r in quality.get(part, [])] for part in ("test", "external")}
    return {
        "kind": "snapshot",
        "snapshot_at_utc": now.isoformat(timespec="seconds"),
        "snapshot_at_kyiv": now.astimezone(ZoneInfo("Europe/Kyiv")).isoformat(timespec="seconds"),
        "as_of": fc["as_of"], "data_last_update": fc["data_last_update"],
        "data_age_s": None if fc["data_age_s"] is None else round(float(fc["data_age_s"]), 1),
        "stale": bool(fc["stale"]), "status_at_snapshot": health["status"],
        "active_count": sum(i["active_now"] for i in items), "units": len(items),
        "disclaimer": fc["disclaimer"], "target": fc["target"], "source_note": fc["source_note"],
        "items": items,
        "excluded": [{"oblast": e["oblast"], "reason": e["reason"]} for e in regions["excluded"]],
        "quality": {"note": quality["note"], **q},
    }


def check(text: str):
    hits = [p for p in FORBIDDEN if re.search(p, text)]
    if hits:
        sys.exit(f"знімок НЕ записано: знайдено заборонені шаблони {hits}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    snap = build(a.api.rstrip("/"))
    text = json.dumps(snap, ensure_ascii=False, indent=1)
    check(text)
    OUT.write_text(text + "\n")
    print(f"{OUT.relative_to(ROOT)}: {snap['units']} одиниць, активних {snap['active_count']}, "
          f"знімок {snap['snapshot_at_kyiv']}, вік даних {snap['data_age_s']} с, stale={snap['stale']}")
