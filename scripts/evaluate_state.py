"""Оцінка моделей бота на замороженому models/state_models.joblib.

  python scripts/evaluate_state.py --part val        # validation (вибір і калібрування — на ній)
  python scripts/evaluate_state.py --part external   # alerts.in.ua: інше визначення тривоги, НЕ test
  python scripts/evaluate_state.py --part test --allow-test   # лише з явної команди замовника

Цілі: P(≥1 нової тривоги) за 1 / 3 / 6 / 24 год (1–6 — стара модель без змін) і лічильник
P(0) / P(1) / P(2+). Test для нових цілей — той самий період, що test старих цілей (його бачили
двічі для y_1/3/6). Вихід: reports/state_metrics.csv, reports/state_counter_metrics.csv (рядки
частини замінюються), reports/state_bootstrap_<part>.json, reports/state_reliability_<part>.csv.
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
from src.state_models import (bot_predict, counter_from, counter_loss, day_bootstrap, ece,  # noqa: E402
                              predict_family, reliability)
from src.state_targets import STATE_HORIZONS  # noqa: E402

PART_NOTE = {"val": "validation (вибір і калібрування — на ній; для 1–6 год калібратор старої моделі теж "
                    "навчався на цьому періоді; оптимістично)",
             "external": "зовнішня перевірка alerts.in.ua: інше визначення тривоги (red + yellow), НЕ test",
             "test": "test (01–28.08.2026); цей період уже бачили двічі для старих цілей"}


def update_csv(path, part, new):
    old = pd.read_csv(path) if path.exists() else new.iloc[0:0]
    pd.concat([old[old.split != part], new], ignore_index=True).to_csv(path, index=False)


def evaluate(part):
    D = pd.read_parquet(PROCESSED_DIR / "dataset_state.parquet")
    df = D[D.keep & (D.split_S == part)].reset_index(drop=True)
    assert set(df.split_S) == {part}
    # joblib = pickle: лише власні артефакти (scripts/train_state.py, scripts/export_service_model.py).
    S = joblib.load(ROOT / "models" / "state_models.joblib")
    old = joblib.load(ROOT / "models" / "service_model.joblib")["models"]
    svc = bot_predict(df, S, old)
    w = svc["west"]
    a = S["any24"]
    # Бази для P(≥1): для 1/3/6 — клімат старої моделі; для 24 — нові бази.
    any_preds = {h: {"climate": old[h]["climate"].predict(df), "service": svc["any"][h]} for h in (1, 3, 6)}
    any_preds[24] = {"climate": a["climate"].predict(df), "climate_wave": a["climate_wave"].predict(df),
                     "logreg": predict_family(a, df, "logreg"), "gbm": predict_family(a, df, "gbm"),
                     "service": svc["any"][24]}
    clim_any = {h: any_preds[h]["climate"] for h in STATE_HORIZONS}
    clim_cnt = counter_from(clim_any, {h: S["q"][h]["climate"].predict(df) for h in STATE_HORIZONS})
    groups = {"overall": np.ones(len(df), bool), "western": w, "non_western": ~w,
              "state_A": (df.state == "A").to_numpy(), "state_B": (df.state == "B").to_numpy()}
    rows, crow, boot, rel = [], [], {}, []
    for h in STATE_HORIZONS:
        y = df[f"y_new_{h}"].to_numpy(float)
        n = df[f"n_new_{h}"].to_numpy(int)
        Pc = {"service": np.column_stack([svc[k][h] for k in ("p0", "p1", "p2plus")]),
              "climate": np.column_stack([clim_cnt[k][h] for k in ("p0", "p1", "p2plus")])}
        for g, m in groups.items():
            sub = df[m]
            days = int(sub.t.dt.floor("D").nunique())
            n_ep = int(sub[sub.state == "B"][["segment", "unit", "episode_id"]].drop_duplicates().shape[0])
            for k, p in any_preds[h].items():
                rows.append({"split": part, "target": "any", "horizon_h": h, "group": g, "model": k,
                             **metrics(y[m], p[m]), "ece": ece(y[m], p[m]), "n_days": days,
                             "n_episodes_state_B": n_ep})
            losses = {}
            for k, P in Pc.items():
                b, ll = counter_loss(n[m], P[m])
                crow.append({"split": part, "horizon_h": h, "group": g, "model": k, "n": int(m.sum()),
                             "share_0": float(np.mean(n[m] == 0)), "share_1": float(np.mean(n[m] == 1)),
                             "share_2plus": float(np.mean(n[m] >= 2)), "brier_multi": float(b.mean()),
                             "log_loss_multi": float(ll.mean()), "n_days": days})
                losses[f"counter_{k}"] = b
            for k, p in any_preds[h].items():
                losses[k] = (p[m] - y[m]) ** 2
            pairs = [("service", b) for b in any_preds[h] if b != "service"] + [("counter_service", "counter_climate")]
            boot[f"{h}|{g}"] = day_bootstrap(sub.t, losses, pairs)
            rel.append(reliability(y[m], any_preds[h]["service"][m]).assign(split=part, horizon_h=h, group=g))
    update_csv(ROOT / "reports" / "state_metrics.csv", part, pd.DataFrame(rows))
    update_csv(ROOT / "reports" / "state_counter_metrics.csv", part, pd.DataFrame(crow))
    (ROOT / "reports" / f"state_bootstrap_{part}.json").write_text(json.dumps(
        {"note": PART_NOTE[part], "n_boot": 1000, "loss": "Brier (для counter_* — багатокласний Brier)",
         **boot}, ensure_ascii=False, indent=1))
    pd.concat(rel).to_csv(ROOT / "reports" / f"state_reliability_{part}.csv", index=False)
    M = pd.DataFrame(rows)
    print(PART_NOTE[part])
    print(M[M.group.isin(["overall", "non_western"])].pivot_table(index=["horizon_h", "group"], columns="model",
                                                                  values="brier").round(4).to_string())
    C = pd.DataFrame(crow)
    print(C[C.group.isin(["overall", "non_western"])].pivot_table(index=["horizon_h", "group"], columns="model",
                                                                  values="brier_multi").round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=list(PART_NOTE), required=True)
    ap.add_argument("--allow-test", action="store_true", help="test для нових цілей — лише з явної команди")
    a = ap.parse_args()
    if a.part == "test" and not a.allow_test:
        sys.exit("test для нових цілей оцінюється один раз і лише з явної команди: --allow-test")
    evaluate(a.part)
