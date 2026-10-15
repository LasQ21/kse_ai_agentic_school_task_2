"""Діагностика західних областей на VALIDATION (test не завантажується)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import PROCESSED_DIR  # noqa: E402
from src.evaluate import metrics  # noqa: E402
from src.features import WESTERN_OBLASTS  # noqa: E402
from src.split import HORIZONS  # noqa: E402


def val_predictions():
    D = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    D = D[D.split_V.isin(["train", "val"])]
    assert not D.split_V.isin(["test", "external"]).any()
    tr, va = D[D.split_V == "train"], D[D.split_V == "val"].reset_index(drop=True)
    # joblib = pickle: безпечно лише для власних артефактів (models.joblib створює scripts/train.py).
    models = joblib.load(ROOT / "models" / "models.joblib")["models"]
    for h in HORIZONS:
        m = models[h]
        va[f"clim_{h}"] = m["climate"].predict(va)
        va[f"gbm_{h}"] = m["gbm_cal"].predict(m["gbm"].predict_proba(va[m["gbm_features"]])[:, 1])
    return tr, va


def brier_decomp(y, p, bins=10):
    """Murphy: Brier = reliability - resolution + uncertainty (квантильні біни)."""
    d = pd.DataFrame({"y": y, "p": p})
    d["b"] = pd.qcut(d.p.rank(method="first"), bins, labels=False)
    g = d.groupby("b").agg(n=("y", "size"), o=("y", "mean"), f=("p", "mean"))
    ybar = d.y.mean()
    rel = (g.n * (g.f - g.o) ** 2).sum() / len(d)
    res = (g.n * (g.o - ybar) ** 2).sum() / len(d)
    return rel, res, ybar * (1 - ybar)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    tr, va = val_predictions()
    w = va.oblast.isin(WESTERN_OBLASTS)
    wtr = tr.oblast.isin(WESTERN_OBLASTS)
    print("західні одиниці:", va[w].unit.nunique(), "| рядків val:", int(w.sum()))
    for h in HORIZONS:
        y = va.loc[w, f"y_{h}"].to_numpy(int)
        print(f"\nH={h}: позитивних {y.sum()} | частка train {tr.loc[wtr, f'y_{h}'].mean():.4f} -> val {y.mean():.4f}")
        for name in ("clim", "gbm"):
            p = va.loc[w, f"{name}_{h}"].to_numpy()
            mt = metrics(y, p); rel, res, unc = brier_decomp(y, p)
            print(f"  {name:5s} brier {mt['brier']:.4f} mean_pred {mt['mean_pred']:.4f} auc {mt['roc_auc']:.3f} | "
                  f"reliability {rel:.5f} resolution {res:.5f} uncertainty {unc:.5f}")
        # де GBM помиляється: за кількістю активних тривог по країні (хвиля атаки)
        d = va[w].assign(err_gbm=(va[f"gbm_{h}"] - va[f"y_{h}"]) ** 2, err_clim=(va[f"clim_{h}"] - va[f"y_{h}"]) ** 2,
                         wave=pd.cut(va.country_active_other, [-1, 0, 10, 30, 60, 200]))
        print(d.groupby("wave", observed=True).agg(rows=("t", "size"), y=(f"y_{h}", "mean"), gbm=(f"gbm_{h}", "mean"),
              clim=(f"clim_{h}", "mean"), brier_gbm=("err_gbm", "mean"), brier_clim=("err_clim", "mean")).round(4).to_string())
