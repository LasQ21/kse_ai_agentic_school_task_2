"""Оцінка заморожених моделей і сервісної моделі (клімат — захід, GBM — решта).

Test (01–28.08.2026) уже оцінено ДВІЧІ (див. reports/model_report.md) і більше не оцінюється:
запуск на test вимагає явного --allow-test. За замовчуванням — лише зовнішня перевірка
(22.09–15.10.2026, alerts.in.ua; інше визначення тривоги). Результат замінює рядки цієї частини
в reports/metrics.csv, решта рядків (зокрема записи test) лишаються як є.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import PROCESSED_DIR  # noqa: E402
from src.evaluate import bootstrap_by_day, calibration_table, grouped_metrics  # noqa: E402
from src.features import FEATURES_LOGREG  # noqa: E402
from src.models import enforce_monotone, service_predict  # noqa: E402
from src.split import HORIZONS  # noqa: E402

PARTS = {"test": "Test (01–28.08.2026, офіційний канал)",
         "external": "Зовнішня перевірка (22.09–15.10.2026, alerts.in.ua; інше визначення тривоги)"}
SERIES = {"service": ("#2a78d6", "Сервіс: клімат (захід) + GBM (решта)"), "climate": ("#eb6834", "Клімат")}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def predict_all(df, models):
    out = {}
    gbm = {h: models[h]["gbm_cal"].predict(models[h]["gbm"].predict_proba(df[models[h]["gbm_features"]])[:, 1])
           for h in HORIZONS}
    gbm, viol = enforce_monotone(gbm)
    service, _ = service_predict(df, models)
    for h in HORIZONS:
        p_clim = models[h]["climate"].predict(df)
        out[h] = {"climate": p_clim, "climate_cal": models[h]["climate_cal"].predict(p_clim),
                  "persistence": models[h]["persistence"].predict(df),
                  "logreg": models[h]["logreg"].predict_proba(df[FEATURES_LOGREG])[:, 1],
                  "gbm": gbm[h], "service": service[h]}
    return out, viol


def calibration_figure(df, preds, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor=SURFACE)
    for ax, h in zip(axes, HORIZONS):
        y = df[f"y_{h}"].to_numpy()
        hi = 0
        for key, (color, label) in SERIES.items():
            ct = calibration_table(y, preds[h][key])
            hi = max(hi, ct.mean_pred.max(), ct.observed.max())
            ax.plot(ct.mean_pred, ct.observed, color=color, lw=2, marker="o", ms=4, label=label)
        lim = min(1.0, hi * 1.08)
        ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(f"H = {h} год", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Середній прогноз у біні", fontsize=9, color=MUTED)
        ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("Спостережена частота", fontsize=9, color=MUTED)
    fig.suptitle(f"Калібрування, 10 квантильних бінів — {title}", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=list(PARTS), default="external")
    ap.add_argument("--allow-test", action="store_true",
                    help="test уже оцінено двічі; запуск лише для нових, свідомо погоджених випадків")
    args = ap.parse_args()
    if args.part == "test" and not args.allow_test:
        sys.exit("test уже оцінено двічі й заморожено; потрібен явний --allow-test")
    part = args.part
    D = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    # joblib = pickle: безпечно лише для власних артефактів (models.joblib створює scripts/train.py).
    models = joblib.load(ROOT / "models" / "models.joblib")["models"]
    df = D[D.split_V == part].reset_index(drop=True)
    preds, viol = predict_all(df, models)
    rows, boot = [], {}
    for h in HORIZONS:
        d = df[["t", "oblast", "active", f"y_{h}"]].copy()
        for k, p in preds[h].items():
            d[k] = p
        g = grouped_metrics(d, {k: k for k in preds[h]}, f"y_{h}")
        g.insert(0, "horizon_h", h); g.insert(0, "split", part)
        rows.append(g)
        boot[h] = {"service_vs_climate": bootstrap_by_day(d, f"y_{h}", "service", "climate")}
    new = pd.concat(rows, ignore_index=True)
    mpath = ROOT / "reports" / "metrics.csv"
    old = pd.read_csv(mpath) if mpath.exists() else new.iloc[0:0]
    pd.concat([old[old.split != part], new], ignore_index=True).to_csv(mpath, index=False)
    epath = ROOT / "reports" / "final_eval.json"
    E = json.loads(epath.read_text()) if epath.exists() else {}
    E.setdefault(part, {})["service_bootstrap"] = boot
    E[part]["monotonicity_violations_gbm_before_fix"] = viol
    epath.write_text(json.dumps(E, ensure_ascii=False, indent=1, default=str))
    calibration_figure(df, preds, PARTS[part], ROOT / "reports" / "figures" / f"calibration_{part}.png")
    print(f"{part}: metrics.csv оновлено ({len(new)} рядків), figures/calibration_{part}.png")


if __name__ == "__main__":
    main()
