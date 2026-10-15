"""Rolling-origin backtest із замороженими гіперпараметрами (лише дані до test)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import PROCESSED_DIR  # noqa: E402
from src.evaluate import metrics  # noqa: E402
from src.features import FEATURES_LOGREG, WESTERN_OBLASTS  # noqa: E402
from src.models import Climate, gbm_features, make_gbm, make_logreg  # noqa: E402
from src.split import HORIZONS, SPLITS, backtest_folds  # noqa: E402

if __name__ == "__main__":
    D = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    D = D[D.split_V.isin(["train", "val"])].reset_index(drop=True)   # test не завантажуємо
    sel = json.loads((ROOT / "reports" / "model_selection.json").read_text())["selection"]
    rows = []
    for origin, tr_m, ev_m in backtest_folds(D, SPLITS["V"]):
        tr, ev = D[tr_m], D[ev_m]
        w = ev.oblast.isin(WESTERN_OBLASTS).to_numpy()
        for h in HORIZONS:
            y_tr, y_ev = tr[f"y_{h}"], ev[f"y_{h}"].to_numpy()
            params = sel[str(h)]["gbm"]["chosen"]
            feats = gbm_features(params.get("use_unit", True))
            preds = {
                "climate": Climate().fit(tr, y_tr).predict(ev),
                "logreg": make_logreg(sel[str(h)]["logreg"]["chosen_C"]).fit(tr[FEATURES_LOGREG], y_tr)
                .predict_proba(ev[FEATURES_LOGREG])[:, 1],
                "gbm_raw": make_gbm(**params).fit(tr[feats], y_tr).predict_proba(ev[feats])[:, 1],
            }
            for name, p in preds.items():
                for grp, mask in (("overall", slice(None)), ("western", w), ("non_western", ~w)):
                    rows.append({"origin": origin.date(), "horizon_h": h, "group": grp, "model": name,
                                 **metrics(y_ev[mask], p[mask])})
    out = pd.DataFrame(rows)
    clim = out[out.model == "climate"].set_index(["origin", "horizon_h", "group"]).brier
    out["brier_skill_vs_climate"] = 1 - out.brier / out.set_index(["origin", "horizon_h", "group"]).index.map(clim)
    out.to_csv(ROOT / "reports" / "backtest.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[out.model != "climate"].pivot_table(index=["horizon_h", "group", "model"], columns="origin",
                                                  values="brier_skill_vs_climate").round(3).to_string())
