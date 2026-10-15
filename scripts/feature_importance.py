"""Permutation importance заморожених GBM на validation (test не використовується)."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import PROCESSED_DIR  # noqa: E402
from src.split import HORIZONS  # noqa: E402

if __name__ == "__main__":
    D = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    va = D[D.split_V == "val"].sample(40000, random_state=0)
    # joblib = pickle: безпечно лише для власних артефактів (models.joblib створює scripts/train.py).
    models = joblib.load(ROOT / "models" / "models.joblib")["models"]
    rows = []
    for h in HORIZONS:
        m, feats = models[h]["gbm"], models[h]["gbm_features"]
        r = permutation_importance(m, va[feats], va[f"y_{h}"], scoring="neg_log_loss",
                                   n_repeats=3, random_state=0, n_jobs=-1)
        for f, mu, sd in zip(feats, r.importances_mean, r.importances_std):
            rows.append({"horizon_h": h, "feature": f, "logloss_increase": mu, "std": sd})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "reports" / "feature_importance_val.csv", index=False)
    print(out.pivot(index="feature", columns="horizon_h", values="logloss_increase")
          .sort_values(1, ascending=False).round(4).to_string())
