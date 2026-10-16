"""Етап розробки: навчання на train, вибір гіперпараметрів і калібрування на validation.

Test і зовнішня перевірка тут НЕ завантажуються. Результат — заморожені моделі в models/
та reports/model_selection.json. Фінальна оцінка — окремо: scripts/evaluate_final.py.
"""
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import PROCESSED_DIR  # noqa: E402
from src.evaluate import grouped_metrics, metrics  # noqa: E402
from src.features import FEATURES_GBM, FEATURES_LOGREG  # noqa: E402
from src.features import WESTERN_OBLASTS  # noqa: E402
from src.models import (GBM_GRID, LOGREG_C, Calibrator, Climate, ClimateWave, Persistence,  # noqa: E402
                        choose_calibration, enforce_monotone, gbm_features, make_gbm, make_logreg)
from src.split import HORIZONS  # noqa: E402

MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
log = logging.getLogger("train")


def load_dev():
    D = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    D = D[D.split_V.isin(["train", "val"])].reset_index(drop=True)
    assert not D.split_V.isin(["test", "external"]).any()
    return D[D.split_V == "train"], D[D.split_V == "val"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tr, va = load_dev()
    log.info("train %d rows (%s..%s), val %d rows (%s..%s)", len(tr), tr.t.min(), tr.t.max(),
             len(va), va.t.min(), va.t.max())
    selection, frozen, val_preds = {}, {}, {}
    for h in HORIZONS:
        y_tr, y_va = tr[f"y_{h}"].to_numpy(int), va[f"y_{h}"].to_numpy(int)
        sel = {}
        clim = Climate().fit(tr, y_tr)
        pers = Persistence(h).fit(tr, y_tr)
        p_clim = clim.predict(va)
        clim_wave = ClimateWave(clim).fit(tr, y_tr)
        # Клімат + калібрування на val: чесніший бейзлайн, бо частота тривог росте з часом.
        m_cc, sc_cc = choose_calibration(p_clim, y_va, va.t.to_numpy())
        cal_clim = Calibrator(m_cc).fit(p_clim, y_va)
        sel["climate_calibration"] = {"chosen": m_cc, "val_logloss_crossfit": sc_cc}

        lr_scores = {}
        for C in LOGREG_C:
            m = make_logreg(C).fit(tr[FEATURES_LOGREG], y_tr)
            lr_scores[C] = metrics(y_va, m.predict_proba(va[FEATURES_LOGREG])[:, 1])["log_loss"]
        best_C = min(lr_scores, key=lr_scores.get)
        lr = make_logreg(best_C).fit(tr[FEATURES_LOGREG], y_tr)
        sel["logreg"] = {"val_logloss_by_C": lr_scores, "chosen_C": best_C}

        gbm_scores, gbm_models = [], []
        for i, params in enumerate(GBM_GRID):
            t0 = time.time()
            feats = gbm_features(params.get("use_unit", True))
            m = make_gbm(**params).fit(tr[feats], y_tr)
            p = m.predict_proba(va[feats])[:, 1]
            ll = metrics(y_va, p)["log_loss"]
            w = va.oblast.isin(WESTERN_OBLASTS).to_numpy()
            mw = metrics(y_va[w], p[w])
            gbm_scores.append({**params, "val_logloss": ll, "val_brier": metrics(y_va, p)["brier"],
                               "val_brier_western": mw["brier"], "val_mean_pred_western": mw["mean_pred"],
                               "fit_s": round(time.time() - t0, 1)})
            gbm_models.append(m)
            log.info("H=%d gbm %s -> val logloss %.5f (%.0fs)", h, params, ll, time.time() - t0)
        best = int(np.argmin([s["val_logloss"] for s in gbm_scores]))
        gbm = gbm_models[best]
        gbm_feats = gbm_features(GBM_GRID[best].get("use_unit", True))
        p_gbm_raw = gbm.predict_proba(va[gbm_feats])[:, 1]
        m_cal, sc_cal = choose_calibration(p_gbm_raw, y_va, va.t.to_numpy())
        cal = Calibrator(m_cal).fit(p_gbm_raw, y_va)
        sel["gbm"] = {"grid": gbm_scores, "chosen": GBM_GRID[best], "features": gbm_feats,
                      "calibration": {"chosen": m_cal, "val_logloss_crossfit": sc_cal}}
        selection[h] = sel
        frozen[h] = {"climate": clim, "climate_cal": cal_clim, "climate_wave": clim_wave,
                     "persistence": pers, "logreg": lr,
                     "gbm": gbm, "gbm_features": gbm_feats, "gbm_cal": cal}
        val_preds[h] = {"climate": p_clim, "climate_cal_in_sample": cal_clim.predict(p_clim),
                        "climate_wave": clim_wave.predict(va),
                        "persistence": pers.predict(va),
                        "logreg": lr.predict_proba(va[FEATURES_LOGREG])[:, 1],
                        "gbm_raw": p_gbm_raw, "gbm_cal_in_sample": cal.predict(p_gbm_raw)}

    mono, viol = enforce_monotone({h: val_preds[h]["gbm_cal_in_sample"] for h in HORIZONS})
    selection["monotonicity_violations_val"] = viol

    rows = []
    for h in HORIZONS:
        df = va[["t", "oblast", "active", f"y_{h}"]].copy()
        cols = {}
        for k, p in val_preds[h].items():
            df[k] = mono[h] if k == "gbm_cal_in_sample" else p
            cols[k] = k
        g = grouped_metrics(df, cols, f"y_{h}")
        g.insert(0, "horizon_h", h)
        rows.append(g)
    val_metrics = pd.concat(rows, ignore_index=True)
    val_metrics.insert(0, "split", "validation")
    REPORTS.mkdir(exist_ok=True)
    val_metrics.to_csv(REPORTS / "metrics_val.csv", index=False)

    MODELS_DIR.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or "uncommitted"
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(), "code_version": commit,
            "train_range": [str(tr.t.min()), str(tr.t.max())],
            "val_range": [str(va.t.min()), str(va.t.max())],
            "features_gbm": FEATURES_GBM, "features_logreg": FEATURES_LOGREG,
            "horizons": list(HORIZONS)}
    joblib.dump({"models": frozen, "meta": meta}, MODELS_DIR / "models.joblib")
    (REPORTS / "model_selection.json").write_text(
        json.dumps({"meta": meta, "selection": selection}, ensure_ascii=False, indent=1, default=str))
    log.info("done; frozen models -> %s", MODELS_DIR / "models.joblib")


if __name__ == "__main__":
    main()
