"""Оцінка цілей за станом на замороженому models/state_models.joblib.

  python scripts/evaluate_state.py --part val        # validation (калібратор навчався на ній)
  python scripts/evaluate_state.py --part external   # alerts.in.ua: інше визначення тривоги, НЕ test
  python scripts/evaluate_state.py --part test --allow-test   # лише з явної команди замовника

Test для нових цілей — той самий період, що test старих цілей (його бачили двічі для y_1/3/6).
Вихід: reports/state_metrics.csv (рядки частини замінюються), reports/state_bootstrap_<part>.json,
reports/state_reliability_<part>.csv.
"""
import argparse
import json
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
from src.state_models import add_state_derived, day_bootstrap, ece, reliability, state_service_predict  # noqa: E402
from src.state_targets import STATE_HORIZONS  # noqa: E402

PART_NOTE = {"val": "validation (калібратор і вибір — на ній; оптимістично)",
             "external": "зовнішня перевірка alerts.in.ua: інше визначення тривоги (red + yellow), НЕ test",
             "test": "test (01–28.08.2026); цей період уже бачили двічі для старих цілей"}


def predictions(df, S, old):
    """Прогнози баз, сирих сімейств і сервісу для станів A (H = 24) і B (h = 1, 3, 6, 24)."""
    svc = state_service_predict(df, S, old)
    A, B = {}, {}
    a = S["A24"]
    A[24] = {"climate": a["climate"].predict(df), "climate_wave": a["climate_wave"].predict(df),
             "logreg": a["logreg"].predict_proba(df[a["logreg_features"]])[:, 1],
             "gbm": a["gbm"].predict_proba(df[a["gbm_features"]])[:, 1], "service": svc["start"][24]}
    for h, m in S["B"].items():
        B[h] = {"climate": m["climate"].predict(df), "survival": m["survival"].predict(df),
                "median_rule": m["median_rule"].predict(df),
                "logreg": m["logreg"].predict_proba(df[m["logreg_features"]])[:, 1],
                "gbm": m["gbm"].predict_proba(df[m["gbm_features"]])[:, 1], "service": svc["cont"][h]}
    return A, B


def evaluate(part):
    D = pd.read_parquet(PROCESSED_DIR / "dataset_state.parquet")
    D = add_state_derived(D[D.keep & (D.split_S == part)].reset_index(drop=True))
    assert set(D.split_S) == {part}
    # joblib = pickle: лише власні артефакти (scripts/train_state.py, scripts/export_service_model.py).
    S = joblib.load(ROOT / "models" / "state_models.joblib")
    old = joblib.load(ROOT / "models" / "service_model.joblib")["models"]
    rows, boot, rel = [], {}, []
    for state, y_prefix, hs in (("A", "y_start", (24,)), ("B", "y_cont", STATE_HORIZONS)):
        df = D[D.state == state].reset_index(drop=True)
        P = predictions(df, S, old)[0 if state == "A" else 1]
        w = df.oblast.isin(WESTERN_OBLASTS).to_numpy()
        for h in hs:
            y = df[f"{y_prefix}_{h}"].to_numpy(float)
            for g, mask in (("overall", np.ones(len(df), bool)), ("western", w), ("non_western", ~w)):
                sub = df[mask]
                n_ep = int(sub[["segment", "unit", "episode_id"]].drop_duplicates().shape[0]) if state == "B" else None
                for k, p in P[h].items():
                    rows.append({"split": part, "state": state, "horizon_h": h, "group": g, "model": k,
                                 **metrics(y[mask], p[mask]), "ece": ece(y[mask], p[mask]),
                                 "n_episodes": n_ep, "n_days": int(sub.t.dt.floor("D").nunique())})
                base = ["climate", "climate_wave", "logreg", "gbm"] if state == "A" else \
                       ["climate", "survival", "logreg", "gbm"]
                boot[f"{state}|{h}|{g}"] = day_bootstrap(sub.t, y[mask], {k: P[h][k][mask] for k in P[h]},
                                                        [("service", b) for b in base])
                r = reliability(y[mask], P[h]["service"][mask])
                rel.append(r.assign(split=part, state=state, horizon_h=h, group=g))
    M = pd.DataFrame(rows)
    path = ROOT / "reports" / "state_metrics.csv"
    old_m = pd.read_csv(path) if path.exists() else M.iloc[0:0]
    pd.concat([old_m[old_m.split != part], M], ignore_index=True).to_csv(path, index=False)
    (ROOT / "reports" / f"state_bootstrap_{part}.json").write_text(
        json.dumps({"note": PART_NOTE[part], "n_boot": 1000, **boot}, ensure_ascii=False, indent=1))
    pd.concat(rel).to_csv(ROOT / "reports" / f"state_reliability_{part}.csv", index=False)
    print(PART_NOTE[part])
    print(M[M.group != "western"].pivot_table(index=["state", "horizon_h", "group"], columns="model",
                                              values="brier").round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=list(PART_NOTE), required=True)
    ap.add_argument("--allow-test", action="store_true", help="test для нових цілей — лише з явної команди")
    a = ap.parse_args()
    if a.part == "test" and not a.allow_test:
        sys.exit("test для нових цілей оцінюється один раз і лише з явної команди: --allow-test")
    evaluate(a.part)
