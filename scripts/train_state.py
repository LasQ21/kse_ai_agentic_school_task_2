"""Цілі за станом: навчання на train, вибір і калібрування на validation (SPLIT_STATE).

Test і зовнішня перевірка тут НЕ завантажуються. Старі моделі (models/models.joblib,
models/service_model.joblib) не змінюються. Результат: models/state_models.joblib,
reports/state_selection.json, reports/state_metrics_val.csv.
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
from src.evaluate import metrics  # noqa: E402
from src.features import FEATURES_LOGREG, WESTERN_OBLASTS  # noqa: E402
from src.models import (GBM_GRID, LOGREG_C, Calibrator, Climate, ClimateWave,  # noqa: E402
                        choose_calibration, gbm_features, make_gbm, make_logreg)
from src.split import _ts  # noqa: E402
from src.state_models import (FEATURES_B_LOGREG, MedianRule, SurvivalTable, add_state_derived,  # noqa: E402
                              ece, gbm_b_features, make_gbm_b, make_logreg_b, monotone_cont)
from src.state_targets import MAX_EPISODE, SPLIT_STATE, STATE_HORIZONS  # noqa: E402

log = logging.getLogger("train_state")


def load_dev():
    """Рядки train і val (keep) з dataset_state.parquet; test і external відкидаються до будь-чого."""
    D = pd.read_parquet(PROCESSED_DIR / "dataset_state.parquet")
    D = D[D.keep & D.split_S.isin(["train", "val"])].reset_index(drop=True)
    assert not D.split_S.isin(["test", "external"]).any()
    return add_state_derived(D)


def train_durations():
    """Тривалості train-епізодів (закінчились до межі train − 96 год) для MedianRule."""
    E = pd.read_parquet(PROCESSED_DIR / "events.parquet")
    end = _ts(SPLIT_STATE.train[1]) - MAX_EPISODE
    e = E[(E.source == "vadimkin") & (E.finished_at < end) & ~E.suspect_end.astype(bool)]
    return pd.DataFrame({"oblast": e.oblast, "duration_h": (e.finished_at - e.started_at).dt.total_seconds() / 3600})


def fit_families(tr, va, y_tr, y_va, gbm_feats_fn, make_gbm_fn, lr_feats, make_lr_fn, tag):
    """GBM по GBM_GRID і лог. регресія по LOGREG_C, вибір за log loss на validation."""
    lr_scores = {}
    for C in LOGREG_C:
        m = make_lr_fn(C).fit(tr[lr_feats], y_tr)
        lr_scores[C] = metrics(y_va, m.predict_proba(va[lr_feats])[:, 1])["log_loss"]
    best_C = min(lr_scores, key=lr_scores.get)
    lr = make_lr_fn(best_C).fit(tr[lr_feats], y_tr)
    grid, models = [], []
    for params in GBM_GRID:
        t0 = time.time()
        feats = gbm_feats_fn(params.get("use_unit", True))
        m = make_gbm_fn(**params).fit(tr[feats], y_tr)
        ll = metrics(y_va, m.predict_proba(va[feats])[:, 1])["log_loss"]
        grid.append({**params, "val_logloss": ll, "fit_s": round(time.time() - t0, 1)})
        models.append(m)
        log.info("%s gbm %s -> val logloss %.5f", tag, params, ll)
    best = int(np.argmin([g["val_logloss"] for g in grid]))
    return {"gbm": models[best], "gbm_features": gbm_feats_fn(GBM_GRID[best].get("use_unit", True)),
            "logreg": lr, "logreg_features": lr_feats}, {
        "logreg": {"val_logloss_by_C": lr_scores, "chosen_C": best_C},
        "gbm": {"grid": grid, "chosen": GBM_GRID[best]}}


def nonwest_logloss(m, va, y_va, kind):
    w = ~va.oblast.isin(WESTERN_OBLASTS).to_numpy()
    p = (m["gbm"].predict_proba(va[m["gbm_features"]])[:, 1] if kind == "gbm"
         else m["logreg"].predict_proba(va[m["logreg_features"]])[:, 1])
    return metrics(y_va[w], p[w])["log_loss"], p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    D = load_dev()
    selection, rows = {}, []

    # ---- стан A, H = 24
    A = D[D.state == "A"]
    trA, vaA = A[A.split_S == "train"], A[A.split_S == "val"]
    y_tr, y_va = trA.y_start_24.to_numpy(int), vaA.y_start_24.to_numpy(int)
    clim = Climate().fit(trA, y_tr)
    mA, selA = fit_families(trA, vaA, y_tr, y_va, gbm_features, make_gbm, FEATURES_LOGREG, make_logreg, "A24")
    ll = {k: nonwest_logloss(mA, vaA, y_va, k) for k in ("gbm", "logreg")}
    family_A = min(ll, key=lambda k: ll[k][0])
    p_raw = ll[family_A][1]
    m_cal, sc = choose_calibration(p_raw, y_va, vaA.t.to_numpy())
    mA.update(climate=clim, climate_wave=ClimateWave(clim).fit(trA, y_tr), cal=Calibrator(m_cal).fit(p_raw, y_va))
    selection["A24"] = {**selA, "family_val_logloss_non_western": {k: v[0] for k, v in ll.items()},
                        "family": family_A, "calibration": {"chosen": m_cal, "val_logloss_crossfit": sc}}
    preds = {"climate": clim.predict(vaA), "climate_wave": mA["climate_wave"].predict(vaA),
             "logreg": ll["logreg"][1], "gbm_raw": ll["gbm"][1], f"{family_A}_cal_in_sample": mA["cal"].predict(p_raw)}
    rows += metric_rows("A", 24, vaA, y_va, preds)

    # ---- стан B, h = 1, 3, 6, 24
    B = D[D.state == "B"]
    trB, vaB = B[B.split_S == "train"], B[B.split_S == "val"]
    durs = train_durations()
    mB, selB, llB = {}, {}, {"gbm": 0.0, "logreg": 0.0}
    for h in STATE_HORIZONS:
        y_tr, y_va = trB[f"y_cont_{h}"].to_numpy(int), vaB[f"y_cont_{h}"].to_numpy(int)
        m, s = fit_families(trB, vaB, y_tr, y_va, gbm_b_features, make_gbm_b, FEATURES_B_LOGREG, make_logreg_b, f"B{h}")
        m.update(climate=Climate().fit(trB, y_tr), survival=SurvivalTable().fit(trB, y_tr),
                 median_rule=MedianRule(h).fit(durs))
        for k in llB:
            llB[k] += nonwest_logloss(m, vaB, y_va, k)[0]
        mB[h], selB[h] = m, s
    family_B = min(llB, key=llB.get)
    raw_B, cal_B = {}, {}
    for h in STATE_HORIZONS:
        y_va = vaB[f"y_cont_{h}"].to_numpy(int)
        p_raw = nonwest_logloss(mB[h], vaB, y_va, family_B)[1]
        m_cal, sc = choose_calibration(p_raw, y_va, vaB.t.to_numpy())
        mB[h]["cal"] = Calibrator(m_cal).fit(p_raw, y_va)
        selB[h]["calibration"] = {"chosen": m_cal, "val_logloss_crossfit": sc}
        raw_B[h], cal_B[h] = p_raw, mB[h]["cal"].predict(p_raw)
    cal_B_mono, viol = monotone_cont(cal_B)
    selection["B"] = {"by_h": selB, "family_val_logloss_non_western_sum_over_h": llB, "family": family_B,
                      "monotonicity_violations_val": viol}
    for h in STATE_HORIZONS:
        y_va = vaB[f"y_cont_{h}"].to_numpy(int)
        m = mB[h]
        other = "logreg" if family_B == "gbm" else "gbm"
        preds = {"climate": m["climate"].predict(vaB), "survival": m["survival"].predict(vaB),
                 "median_rule": m["median_rule"].predict(vaB), f"{family_B}_raw": raw_B[h],
                 f"{other}_raw": nonwest_logloss(m, vaB, y_va, other)[1],
                 f"{family_B}_cal_in_sample_monotone": cal_B_mono[h]}
        rows += metric_rows("B", h, vaB, y_va, preds)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip() or "uncommitted"
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(), "code_version": commit,
            "split": SPLIT_STATE.name, "train": SPLIT_STATE.train, "val": SPLIT_STATE.val,
            "rows": {"A_train": len(trA), "A_val": len(vaA), "B_train": len(trB), "B_val": len(vaB)}}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"A24": mA, "B": mB, "family_A": family_A, "family_B": family_B, "meta": meta},
                ROOT / "models" / "state_models.joblib")
    (ROOT / "reports" / "state_selection.json").write_text(
        json.dumps({"meta": meta, "selection": selection}, ensure_ascii=False, indent=1, default=str))
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "state_metrics_val.csv", index=False)
    log.info("family A24=%s, B=%s", family_A, family_B)


def metric_rows(state, h, df, y, preds):
    """Метрики по групах overall / western / non_western; для B — кількість епізодів."""
    out = []
    w = df.oblast.isin(WESTERN_OBLASTS).to_numpy()
    for g, mask in (("overall", np.ones(len(df), bool)), ("western", w), ("non_western", ~w)):
        n_ep = (int(df[mask][["segment", "unit", "episode_id"]].drop_duplicates().shape[0])
                if state == "B" else None)
        for k, p in preds.items():
            out.append({"split": "val", "state": state, "horizon_h": h, "group": g, "model": k,
                        **metrics(y[mask], p[mask]), "ece": ece(y[mask], p[mask]), "n_episodes": n_ep})
    return out


if __name__ == "__main__":
    main()
