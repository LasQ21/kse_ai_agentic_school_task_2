"""Rolling-origin backtest для моделей бота: 5 вікон по 28 діб, заморожені гіперпараметри
(reports/state_selection.json), без калібрування. Train = до origin − EMBARGO_STATE; вікна не
заходять у test. P(≥1 за 24 год) і q_h = P(≥2 | ≥1). Вихід: reports/state_backtest.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import PROCESSED_DIR  # noqa: E402
from src.evaluate import metrics  # noqa: E402
from src.features import FEATURES_LOGREG, WESTERN_OBLASTS  # noqa: E402
from src.models import Climate, ClimateWave, gbm_features, make_gbm, make_logreg  # noqa: E402
from src.split import BACKTEST_ORIGINS, BACKTEST_WINDOW, _ts  # noqa: E402
from src.state_models import ece, gbm_q_features, make_gbm_q  # noqa: E402
from src.state_targets import EMBARGO_STATE, SPLIT_STATE, STATE_HORIZONS  # noqa: E402


def folds(D):
    """(origin, маска train, маска вікна оцінки) на рядках vadimkin до test − EMBARGO_STATE."""
    test_start = _ts(SPLIT_STATE.test[0])
    base = D.keep & (D.segment == "vadimkin")
    for o in BACKTEST_ORIGINS:
        o = _ts(o)
        end = min(o + BACKTEST_WINDOW, test_start - EMBARGO_STATE)
        yield o, base & (D.t < o - EMBARGO_STATE), base & (D.t >= o) & (D.t < end)


def fit_predict(tr, ev, y_tr, gp, C, feats_fn, make_fn, wave):
    clim = Climate().fit(tr, y_tr)
    f = feats_fn(gp.get("use_unit", True))
    out = {"climate": clim.predict(ev),
           "logreg": make_logreg(C).fit(tr[FEATURES_LOGREG], y_tr).predict_proba(ev[FEATURES_LOGREG])[:, 1],
           "gbm": make_fn(**gp).fit(tr[f], y_tr).predict_proba(ev[f])[:, 1]}
    if wave:
        out["climate_wave"] = ClimateWave(clim).fit(tr, y_tr).predict(ev)
    return out


if __name__ == "__main__":
    D = pd.read_parquet(PROCESSED_DIR / "dataset_state.parquet")
    D = D[D.split_S.isin(["train", "val"]) | (D.split_S.isna() & (D.segment == "vadimkin")
                                              & (D.t < _ts(SPLIT_STATE.test[0]) - EMBARGO_STATE))]
    assert not D.split_S.isin(["test", "external"]).any()
    D = D.reset_index(drop=True)
    sel = json.loads((ROOT / "reports" / "state_selection.json").read_text())["selection"]
    rows = []
    for origin, tr_m, ev_m in folds(D):
        tr, ev = D[tr_m], D[ev_m]
        tasks = [("any", 24, tr, ev, tr.y_new_24, ev.y_new_24, sel["any24"], gbm_features, make_gbm, True)]
        for h in STATE_HORIZONS:
            a, b = tr[tr[f"n_new_{h}"] >= 1], ev[ev[f"n_new_{h}"] >= 1]
            tasks.append(("q", h, a, b, a[f"n_new_{h}"] >= 2, b[f"n_new_{h}"] >= 2, sel["q"]["by_h"][str(h)],
                          gbm_q_features, make_gbm_q, False))
        for target, h, a, b, ya, yb, s, ff, mf, wave in tasks:
            P = fit_predict(a, b, ya.to_numpy(int), s["gbm"]["chosen"], s["logreg"]["chosen_C"], ff, mf, wave)
            w = b.oblast.isin(WESTERN_OBLASTS).to_numpy()
            yb = yb.to_numpy(int)
            for k, p in P.items():
                for g, m in (("overall", np.ones(len(b), bool)), ("western", w), ("non_western", ~w)):
                    rows.append({"origin": origin.date(), "target": target, "horizon_h": h, "group": g,
                                 "model": k, **metrics(yb[m], p[m]), "ece": ece(yb[m], p[m])})
    out = pd.DataFrame(rows)
    key = ["origin", "target", "horizon_h", "group"]
    clim = out[out.model == "climate"].set_index(key).brier
    out["brier_skill_vs_climate"] = 1 - out.brier / out.set_index(key).index.map(clim)
    out.to_csv(ROOT / "reports" / "state_backtest.csv", index=False)
    print(out[out.group == "non_western"].pivot_table(index=["target", "horizon_h", "model"], columns="origin",
                                                      values="brier_skill_vs_climate").round(3).to_string())
