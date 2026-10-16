"""GBM проти баз (клімат, клімат + хвиля, логістична регресія) з 95% CI.

Validation і зовнішня перевірка рахуються тут. Test НЕ завантажується й не перераховується:
його числа беруться з reports/final_eval.json і reports/metrics.csv (друга, остання оцінка test;
для бази «клімат + хвиля» test немає). Backtest — з reports/backtest.csv.

На validation GBM береться без калібрування (калібратор навчався на тій самій validation).
На зовнішній перевірці — як у сервісі: калібрований і монотонний за H.
CI: блочний бутстреп по днях для Brier skill = 1 - Brier(a) / Brier(b), 2000 повторів.
Вихід: reports/baseline_comparison.csv і reports/baseline_comparison.md.
"""
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
from src.features import FEATURES_LOGREG, WESTERN_OBLASTS  # noqa: E402
from src.models import enforce_monotone  # noqa: E402
from src.split import HORIZONS  # noqa: E402

N_BOOT, SEED = 2000, 0
PAIRS = [("gbm", "climate"), ("gbm", "logreg"), ("gbm", "climate_wave"), ("logreg", "climate"),
         ("climate_wave", "climate")]
GROUPS = {"overall": None, "non_western": False}


def predictions(df, models, part):
    raw = {h: models[h]["gbm"].predict_proba(df[models[h]["gbm_features"]])[:, 1] for h in HORIZONS}
    if part == "external":
        raw, _ = enforce_monotone({h: models[h]["gbm_cal"].predict(raw[h]) for h in HORIZONS})
    out = {}
    for h in HORIZONS:
        m = models[h]
        out[h] = {"climate": m["climate"].predict(df), "climate_wave": m["climate_wave"].predict(df),
                  "logreg": m["logreg"].predict_proba(df[FEATURES_LOGREG])[:, 1], "gbm": raw[h]}
    return out


def day_bootstrap(t, y, preds, pairs, rng):
    codes, days = pd.factorize(t.dt.floor("D"))
    idx = rng.integers(0, len(days), size=(N_BOOT, len(days)))
    se = {k: np.bincount(codes, weights=(p - y) ** 2, minlength=len(days)) for k, p in preds.items()}
    res = {}
    for a, b in pairs:
        bss = 1 - se[a][idx].sum(1) / se[b][idx].sum(1)
        point = 1 - se[a].sum() / se[b].sum()
        res[(a, b)] = (point, float(np.quantile(bss, 0.025)), float(np.quantile(bss, 0.975)), len(days))
    return res


def main():
    D = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    D = D[D.split_V.isin(["val", "external"])].reset_index(drop=True)
    assert not D.split_V.isin(["test"]).any()
    # joblib = pickle: безпечно лише для власних артефактів (models.joblib створює scripts/train.py).
    models = joblib.load(ROOT / "models" / "models.joblib")["models"]
    rng = np.random.default_rng(SEED)
    rows = []
    for part in ("val", "external"):
        df = D[D.split_V == part].reset_index(drop=True)
        P = predictions(df, models, part)
        for h in HORIZONS:
            y = df[f"y_{h}"].to_numpy(float)
            for g, west in GROUPS.items():
                mask = np.ones(len(df), bool) if west is None else (~df.oblast.isin(WESTERN_OBLASTS)).to_numpy()
                preds = {k: v[mask] for k, v in P[h].items()}
                for k, p in preds.items():
                    rows.append({"part": part, "horizon_h": h, "group": g, "kind": "brier", "a": k, "b": "",
                                 "value": metrics(y[mask], p)["brier"], "lo": np.nan, "hi": np.nan, "n_days": np.nan})
                for (a, b), (pt, lo, hi, nd) in day_bootstrap(df.t[mask], y[mask], preds, PAIRS, rng).items():
                    rows.append({"part": part, "horizon_h": h, "group": g, "kind": "skill", "a": a, "b": b,
                                 "value": pt, "lo": lo, "hi": hi, "n_days": nd})
    # test: лише збережені числа (не перераховуються)
    E = json.loads((ROOT / "reports" / "final_eval.json").read_text())["test"]["bootstrap"]
    M = pd.read_csv(ROOT / "reports" / "metrics.csv")
    for h in HORIZONS:
        for g, key_sfx in (("overall", ""), ("non_western", "_non_western")):
            for b in ("climate", "logreg"):
                k = f"gbm_vs_{b}{key_sfx}"
                if k in E[str(h)]:
                    lo, hi = E[str(h)][k]["brier_skill_ci95"]
                    grp = "overall" if g == "overall" else "non_western"
                    br = M[(M.split == "test") & (M.group == grp) & (M.horizon_h == h)].set_index("model").brier
                    rows.append({"part": "test", "horizon_h": h, "group": g, "kind": "skill", "a": "gbm", "b": b,
                                 "value": 1 - br["gbm"] / br[b], "lo": lo, "hi": hi, "n_days": E[str(h)][k]["n_days"]})
    # backtest: Brier skill по вікнах
    bt = pd.read_csv(ROOT / "reports" / "backtest.csv")
    for h in HORIZONS:
        for g in GROUPS:
            x = bt[(bt.horizon_h == h) & (bt.group == g)].pivot_table(index="origin", columns="model", values="brier")
            for a, b, col_a in (("gbm", "climate", "gbm_raw"), ("gbm", "logreg", "gbm_raw"),
                                ("gbm", "climate_wave", "gbm_raw"),
                                ("logreg", "climate", "logreg"), ("climate_wave", "climate", "climate_wave")):
                if col_a in x and b in x:
                    s = 1 - x[col_a] / x[b]  # b — назва колонки бази в backtest.csv
                    rows.append({"part": "backtest", "horizon_h": h, "group": g, "kind": "skill_folds", "a": a, "b": b,
                                 "value": float(s.median()), "lo": float(s.min()), "hi": float(s.max()), "n_days": len(s)})
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "reports" / "baseline_comparison.csv", index=False)
    write_md(R)
    print(R[R.kind != "brier"].round(3).to_string(index=False))


def write_md(R):
    names = {"gbm": "GBM", "climate": "клімат", "logreg": "лог. регресія", "climate_wave": "клімат + хвиля"}
    L = ["# GBM проти баз", "", "Згенеровано `scripts/compare_baselines.py`. Brier skill = 1 − Brier(A) / Brier(B):",
         "> 0 — A краща. CI 95%: блочний бутстреп по днях (validation, зовнішня перевірка: 2000 повторів;",
         "test: збережені CI другої оцінки, 300 повторів, не перераховано). Backtest: медіана [мін, макс] по 5 вікнах.",
         "На validation GBM без калібрування; рішення щодо гіперпараметрів ухвалювались на validation, тож вона",
         "зміщена на користь GBM.", ""]
    for g, title in (("overall", "Усі 105 одиниць"), ("non_western", "Незахідні області (там сервіс використовує GBM)")):
        L += [f"## {title}", "", "| частина | H | GBM vs клімат | GBM vs лог. регресія | GBM vs клімат + хвиля | лог. регресія vs клімат |",
              "|---|---|---|---|---|---|"]
        for part, pname in (("val", "validation"), ("backtest", "backtest (5 вікон)"), ("test", "test (не перераховано)"),
                            ("external", "зовнішня (alerts.in.ua)")):
            for h in HORIZONS:
                cells = []
                for a, b in (("gbm", "climate"), ("gbm", "logreg"), ("gbm", "climate_wave"), ("logreg", "climate")):
                    x = R[(R.part == part) & (R.group == g) & (R.horizon_h == h) & (R.a == a) & (R.b == b) & (R.kind != "brier")]
                    cells.append("—" if x.empty else f"{x.value.iloc[0]:+.3f} [{x.lo.iloc[0]:+.3f}, {x.hi.iloc[0]:+.3f}]")
                L.append(f"| {pname} | {h} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Що дають самі ознаки хвилі: клімат + хвиля vs клімат", "",
          "| частина | група | H=1 | H=3 | H=6 |", "|---|---|---|---|---|"]
    for part, pname in (("val", "validation"), ("backtest", "backtest (5 вікон)"), ("external", "зовнішня (alerts.in.ua)")):
        for g in ("overall", "non_western"):
            cells = []
            for h in HORIZONS:
                x = R[(R.part == part) & (R.group == g) & (R.horizon_h == h) & (R.a == "climate_wave") & (R.b == "climate")]
                cells.append("—" if x.empty else f"{x.value.iloc[0]:+.3f} [{x.lo.iloc[0]:+.3f}, {x.hi.iloc[0]:+.3f}]")
            L.append(f"| {pname} | {g} | " + " | ".join(cells) + " |")
    L.append("")
    (ROOT / "reports" / "baseline_comparison.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
