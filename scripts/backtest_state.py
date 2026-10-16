"""Rolling-origin backtest для цілей за станом: 5 вікон по 28 діб, заморожені гіперпараметри
(reports/state_selection.json), без калібрування. Train = до origin − EMBARGO_STATE; вікна не
заходять у test. Вихід: reports/state_backtest.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import PROCESSED_DIR  # noqa: E402
from src.evaluate import metrics  # noqa: E402
from src.features import WESTERN_OBLASTS  # noqa: E402
from src.models import Climate, ClimateWave, make_logreg  # noqa: E402
from src.split import BACKTEST_ORIGINS, BACKTEST_WINDOW, _ts  # noqa: E402
from src.state_models import (FEATURES_A_LOGREG, FEATURES_B_LOGREG, MedianRule, SurvivalTable,  # noqa: E402
                              add_state_derived, ece, gbm_a_features, gbm_b_features, make_gbm_a,
                              make_gbm_b, make_logreg_b)
from src.state_targets import EMBARGO_STATE, MAX_EPISODE, SPLIT_STATE, STATE_HORIZONS  # noqa: E402


def folds(D):
    test_start = _ts(SPLIT_STATE.test[0])
    base = D.keep & (D.segment == "vadimkin")
    for o in BACKTEST_ORIGINS:
        o = _ts(o)
        end = min(o + BACKTEST_WINDOW, test_start - EMBARGO_STATE)
        yield o, base & (D.t < o - EMBARGO_STATE), base & (D.t >= o) & (D.t < end)


if __name__ == "__main__":
    D = pd.read_parquet(PROCESSED_DIR / "dataset_state.parquet")
    D = D[D.split_S.isin(["train", "val"]) | (D.split_S.isna() & (D.segment == "vadimkin")
                                              & (D.t < _ts(SPLIT_STATE.test[0]) - EMBARGO_STATE))]
    assert not D.split_S.isin(["test", "external"]).any()
    D = add_state_derived(D.reset_index(drop=True))
    sel = json.loads((ROOT / "reports" / "state_selection.json").read_text())["selection"]
    E = pd.read_parquet(PROCESSED_DIR / "events.parquet")
    rows = []
    for origin, tr_m, ev_m in folds(D):
        e = E[(E.source == "vadimkin") & (E.finished_at < origin - EMBARGO_STATE - MAX_EPISODE)
              & ~E.suspect_end.astype(bool)]
        durs = pd.DataFrame({"oblast": e.oblast, "duration_h": (e.finished_at - e.started_at).dt.total_seconds() / 3600})
        for state, prefix, hs in (("A", "y_start", (24,)), ("B", "y_cont", STATE_HORIZONS)):
            tr, ev = D[tr_m & (D.state == state)], D[ev_m & (D.state == state)]
            w = ev.oblast.isin(WESTERN_OBLASTS).to_numpy()
            for h in hs:
                y_tr, y_ev = tr[f"{prefix}_{h}"].to_numpy(int), ev[f"{prefix}_{h}"].to_numpy(int)
                s = sel["A24"] if state == "A" else sel["B"]["by_h"][str(h)]
                gp, C = s["gbm"]["chosen"], s["logreg"]["chosen_C"]
                clim = Climate().fit(tr, y_tr)
                if state == "A":
                    f = gbm_a_features(gp.get("use_unit", True))
                    preds = {"climate": clim.predict(ev), "climate_wave": ClimateWave(clim).fit(tr, y_tr).predict(ev),
                             "logreg": make_logreg(C).fit(tr[FEATURES_A_LOGREG], y_tr).predict_proba(ev[FEATURES_A_LOGREG])[:, 1],
                             "gbm": make_gbm_a(**gp).fit(tr[f], y_tr).predict_proba(ev[f])[:, 1]}
                else:
                    f = gbm_b_features(gp.get("use_unit", True))
                    preds = {"climate": clim.predict(ev), "survival": SurvivalTable().fit(tr, y_tr).predict(ev),
                             "median_rule": MedianRule(h).fit(durs).predict(ev),
                             "logreg": make_logreg_b(C).fit(tr[FEATURES_B_LOGREG], y_tr).predict_proba(ev[FEATURES_B_LOGREG])[:, 1],
                             "gbm": make_gbm_b(**gp).fit(tr[f], y_tr).predict_proba(ev[f])[:, 1]}
                for k, p in preds.items():
                    for g, mask in (("overall", np.ones(len(ev), bool)), ("western", w), ("non_western", ~w)):
                        n_ep = int(ev[mask][["unit", "episode_id"]].drop_duplicates().shape[0]) if state == "B" else None
                        rows.append({"origin": origin.date(), "state": state, "horizon_h": h, "group": g, "model": k,
                                     **metrics(y_ev[mask], p[mask]), "ece": ece(y_ev[mask], p[mask]), "n_episodes": n_ep})
    out = pd.DataFrame(rows)
    key = ["origin", "state", "horizon_h", "group"]
    clim = out[out.model == "climate"].set_index(key).brier
    out["brier_skill_vs_climate"] = 1 - out.brier / out.set_index(key).index.map(clim)
    out.to_csv(ROOT / "reports" / "state_backtest.csv", index=False)
    print(out[out.group == "non_western"].pivot_table(index=["state", "horizon_h", "model"], columns="origin",
                                                      values="brier_skill_vs_climate").round(3).to_string())
