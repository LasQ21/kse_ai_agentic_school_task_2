"""Заморожені моделі -> models/service_model.joblib (лише те, що потрібно сервісу) + метадані."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import ALERTS_IN_UA_LEVELS, MERGE_GAP_S, PROCESSED_DIR  # noqa: E402
from src.features import WESTERN_OBLASTS  # noqa: E402


def code_version() -> dict:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or None
    h = hashlib.sha256()
    for f in sorted((ROOT / "src").glob("*.py")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return {"git_commit": commit, "src_sha256": h.hexdigest()[:16]}


if __name__ == "__main__":
    # joblib = pickle: безпечно лише для власних артефактів (models.joblib створює scripts/train.py).
    bundle = joblib.load(ROOT / "models" / "models.joblib")
    models = {h: {k: m[k] for k in ("climate", "gbm", "gbm_features", "gbm_cal")}
              for h, m in bundle["models"].items()}
    units = json.loads((PROCESSED_DIR / "coverage.json").read_text())["units"]
    meta = {**bundle["meta"], "exported_at": datetime.now(timezone.utc).isoformat(),
            "code_version": code_version(), "units": units,
            "routing": {"climate": sorted(WESTERN_OBLASTS), "gbm": "усі інші одиниці"},
            "live_alert_levels": sorted(ALERTS_IN_UA_LEVELS), "merge_gap_s": MERGE_GAP_S,
            "note": "Модель навчена на офіційному каналі (до 2026-06-13); живі дані alerts.in.ua мають інше визначення тривоги."}
    joblib.dump({"models": models, "meta": meta}, ROOT / "models" / "service_model.joblib")
    (ROOT / "models" / "service_model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("trained_at", "exported_at", "code_version", "train_range", "routing")},
                     ensure_ascii=False, indent=1))
