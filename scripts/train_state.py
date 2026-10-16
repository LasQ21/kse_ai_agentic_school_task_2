"""Моделі для бота: навчання на train, вибір і калібрування на validation (SPLIT_STATE).

- any24: P(≥1 нової тривоги за 24 год) — той самий протокол і ознаки, що й старі моделі 1/3/6;
- q_h: P(≥2 нових | ≥1 нової) за h ∈ {1, 3, 6, 24} — для лічильника P(0) / P(1) / P(2+).
Test і зовнішня перевірка тут НЕ завантажуються. Старі моделі не змінюються.
Результат: models/state_models.joblib, reports/state_selection.json, reports/state_metrics_val.csv.
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
from src.state_models import ece, gbm_q_features, make_gbm_q  # noqa: E402
from src.state_targets import SPLIT_STATE, STATE_HORIZONS  # noqa: E402

log = logging.getLogger("train_state")


def load_dev():
    """Рядки train і val (keep) з dataset_state.parquet; test і external відкидаються до будь-чого."""
    D = pd.read_parquet(PROCESSED_DIR / "dataset_state.parquet")
    D = D[D.keep & D.split_S.isin(["train", "val"])].reset_index(drop=True)
    assert not D.split_S.isin(["test", "external"]).any()
    return D


def fit_families(tr, va, y_tr, y_va, gbm_feats_fn, make_gbm_fn, tag):
    """GBM по GBM_GRID і лог. регресія (FEATURES_LOGREG) по LOGREG_C; вибір за log loss на validation."""
    lr_scores = {}
    for C in LOGREG_C:
        m = make_logreg(C).fit(tr[FEATURES_LOGREG], y_tr)
        lr_scores[C] = metrics(y_va, m.predict_proba(va[FEATURES_LOGREG])[:, 1])["log_loss"]
    best_C = min(lr_scores, key=lr_scores.get)
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
    return ({"gbm": models[best], "gbm_features": gbm_feats_fn(GBM_GRID[best].get("use_unit", True)),
             "logreg": make_logreg(best_C).fit(tr[FEATURES_LOGREG], y_tr), "logreg_features": FEATURES_LOGREG},
            {"logreg": {"val_logloss_by_C": lr_scores, "chosen_C": best_C},
             "gbm": {"grid": grid, "chosen": GBM_GRID[best]}})


def raw(m, df, kind):
    return m[kind].predict_proba(df[m[f"{kind}_features"]])[:, 1]


def nonwest_ll(m, va, y_va, kind):
    w = ~va.oblast.isin(WESTERN_OBLASTS).to_numpy()
    return metrics(y_va[w], raw(m, va, kind)[w])["log_loss"]


def metric_rows(target, h, df, y, preds):
    """Метрики по групах overall / western / non_western і стану в t (A / B)."""
    out = []
    w = df.oblast.isin(WESTERN_OBLASTS).to_numpy()
    groups = {"overall": np.ones(len(df), bool), "western": w, "non_western": ~w,
              "state_A": (df.state == "A").to_numpy(), "state_B": (df.state == "B").to_numpy()}
    for g, mask in groups.items():
        for k, p in preds.items():
            out.append({"split": "val", "target": target, "horizon_h": h, "group": g, "model": k,
                        **metrics(y[mask], p[mask]), "ece": ece(y[mask], p[mask])})
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    D = load_dev()
    tr, va = D[D.split_S == "train"], D[D.split_S == "val"]
    selection, rows = {}, []

    # ---- any24: P(≥1 нової за 24 год), усі рядки
    y_tr, y_va = tr.y_new_24.to_numpy(int), va.y_new_24.to_numpy(int)
    clim = Climate().fit(tr, y_tr)
    m24, s24 = fit_families(tr, va, y_tr, y_va, gbm_features, make_gbm, "any24")
    ll = {k: nonwest_ll(m24, va, y_va, k) for k in ("gbm", "logreg")}
    fam24 = min(ll, key=ll.get)
    p = raw(m24, va, fam24)
    mc, sc = choose_calibration(p, y_va, va.t.to_numpy())
    m24.update(climate=clim, climate_wave=ClimateWave(clim).fit(tr, y_tr), cal=Calibrator(mc).fit(p, y_va))
    selection["any24"] = {**s24, "family_val_logloss_non_western": ll, "family": fam24,
                          "calibration": {"chosen": mc, "val_logloss_crossfit": sc}}
    rows += metric_rows("any", 24, va, y_va, {
        "climate": clim.predict(va), "climate_wave": m24["climate_wave"].predict(va),
        "logreg": raw(m24, va, "logreg"), "gbm": raw(m24, va, "gbm"),
        f"{fam24}_cal_in_sample": m24["cal"].predict(p)})

    # ---- q_h: P(≥2 | ≥1) на рядках з хоча б одним стартом за h
    mq, sq, llq = {}, {}, {"gbm": 0.0, "logreg": 0.0}
    for h in STATE_HORIZONS:
        trq, vaq = tr[tr[f"n_new_{h}"] >= 1], va[va[f"n_new_{h}"] >= 1]
        yq_tr, yq_va = (trq[f"n_new_{h}"] >= 2).to_numpy(int), (vaq[f"n_new_{h}"] >= 2).to_numpy(int)
        m, s = fit_families(trq, vaq, yq_tr, yq_va, gbm_q_features, make_gbm_q, f"q{h}")
        m["climate"] = Climate().fit(trq, yq_tr)
        s["rows"] = {"train": len(trq), "val": len(vaq), "val_share_2plus": float(yq_va.mean())}
        for k in llq:
            llq[k] += nonwest_ll(m, vaq, yq_va, k)
        mq[h], sq[h] = m, s
    famq = min(llq, key=llq.get)
    for h in STATE_HORIZONS:
        vaq = va[va[f"n_new_{h}"] >= 1]
        yq_va = (vaq[f"n_new_{h}"] >= 2).to_numpy(int)
        p = raw(mq[h], vaq, famq)
        mc, sc = choose_calibration(p, yq_va, vaq.t.to_numpy())
        mq[h]["cal"] = Calibrator(mc).fit(p, yq_va)
        sq[h]["calibration"] = {"chosen": mc, "val_logloss_crossfit": sc}
        rows += metric_rows("q", h, vaq, yq_va, {
            "climate": mq[h]["climate"].predict(vaq), "logreg": raw(mq[h], vaq, "logreg"),
            "gbm": raw(mq[h], vaq, "gbm"), f"{famq}_cal_in_sample": mq[h]["cal"].predict(p)})
    selection["q"] = {"by_h": sq, "family_val_logloss_non_western_sum_over_h": llq, "family": famq}

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip() or "uncommitted"
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(), "code_version": commit,
            "uncommitted_changes": dirty, "split": SPLIT_STATE.name, "train": SPLIT_STATE.train,
            "val": SPLIT_STATE.val, "rows": {"train": len(tr), "val": len(va)}}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"any24": m24, "q": mq, "family_any24": fam24, "family_q": famq, "meta": meta},
                ROOT / "models" / "state_models.joblib")
    (ROOT / "reports" / "state_selection.json").write_text(
        json.dumps({"meta": meta, "selection": selection}, ensure_ascii=False, indent=1, default=str))
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "state_metrics_val.csv", index=False)
    log.info("family any24=%s, q=%s", fam24, famq)


if __name__ == "__main__":
    main()
