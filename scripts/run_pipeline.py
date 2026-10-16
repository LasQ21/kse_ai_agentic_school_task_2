"""Одна команда: сирі дані -> оброблені -> мітки -> ознаки -> навчання -> оцінка -> звіт.

  python scripts/run_pipeline.py            # з наявних data/raw (якщо їх немає — завантажить)
  python scripts/run_pipeline.py --refresh  # спершу свіжий Vadimkin і знімок історії alerts.in.ua (~14 хв)

Розбиття (src.split.SPLITS["V"]) і правило «захід — клімат, решта — GBM» (src.models.service_predict)
зафіксовані в коді. TEST НЕ ОЦІНЮЄТЬСЯ: test старих цілей оцінено двічі, нових — один раз (ea6b738), обидва
заморожено. Команди нижче лишено для відтворення, повторно їх не запускати:
  python scripts/evaluate_final.py --part test --allow-test   # старі цілі 1/3/6
  python scripts/evaluate_state.py --part test --allow-test   # цілі бота (24 год, лічильник)
Підсумок прогону: reports/pipeline_run.md.
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import PROCESSED_DIR, RAW_DIR  # noqa: E402

PY = sys.executable
STEPS = []


def step(name):
    def deco(fn):
        STEPS.append((name, fn))
        return fn
    return deco


def run_script(*args):
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def have_raw() -> bool:
    return (any((RAW_DIR / "vadimkin").glob("official_data_uk_*.csv"))
            and (RAW_DIR / "alerts_in_ua").exists() and any((RAW_DIR / "alerts_in_ua").iterdir()))


@step("сирі дані")
def raw_data(args):
    from src.data import archive_alerts_in_ua_history, download_vadimkin
    if args.refresh or not have_raw():
        download_vadimkin()
        archive_alerts_in_ua_history()


@step("очищення -> data/processed/events.parquet, coverage.json")
def processed(args):
    from src.data import build_processed
    build_processed()


@step("мітки й розбиття -> labels.parquet")
def labels(args):
    from src.split import build_and_save
    build_and_save()


@step("ознаки -> dataset.parquet")
def features(args):
    from src.features import build_feature_table
    build_feature_table()


@step("навчання, вибір і калібрування на validation")
def train(args):
    run_script("scripts/train.py")


@step("rolling-origin backtest")
def backtest(args):
    run_script("scripts/backtest.py")


@step("важливість ознак (validation)")
def importance(args):
    run_script("scripts/feature_importance.py")


@step("експорт моделі сервісу")
def export(args):
    run_script("scripts/export_service_model.py")


@step("зовнішня перевірка (alerts.in.ua; НЕ test)")
def external(args):
    run_script("scripts/evaluate_final.py", "--part", "external")


@step("цілі й ознаки бота -> state_labels.parquet, dataset_state.parquet")
def state_data(args):
    from src.state_features import build_state_dataset
    from src.state_targets import build_and_save as build_state_labels
    build_state_labels()
    build_state_dataset()


@step("моделі бота: навчання, вибір і калібрування на validation")
def state_train(args):
    run_script("scripts/train_state.py")


@step("моделі бота: validation, backtest, зовнішня перевірка (НЕ test)")
def state_eval(args):
    run_script("scripts/evaluate_state.py", "--part", "val")
    run_script("scripts/backtest_state.py")
    run_script("scripts/evaluate_state.py", "--part", "external")


@step("pytest")
def tests(args):
    if not args.skip_tests:
        run_script("-m", "pytest", "-q", "-p", "no:cacheprovider")


def write_summary(timings):
    """Підсумок прогону -> reports/pipeline_run.md (без test)."""
    R = ROOT / "reports"
    cov = json.loads((PROCESSED_DIR / "coverage.json").read_text())
    L = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    sel = json.loads((R / "model_selection.json").read_text())
    val = pd.read_csv(R / "metrics_val.csv")
    bt = pd.read_csv(R / "backtest.csv")
    M = pd.read_csv(R / "metrics.csv")

    def pick(df, group, model, h, col="brier"):
        x = df[(df.group == group) & (df.model == model) & (df.horizon_h == h)]
        return float(x[col].iloc[0]) if len(x) else float("nan")

    lines = [f"# Прогін пайплайна {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", "",
             "Згенеровано `scripts/run_pipeline.py`. **Цей прогін test не оцінював** (старі цілі оцінено "
             "двічі, нові — один раз окремо; див. `model_report.md`).", "",
             "## Вхідні дані", f"- Vadimkin: `{cov['inputs']['vadimkin']}`",
             f"- знімки alerts.in.ua: {', '.join(cov['inputs']['alerts_in_ua_archives'])}",
             *[f"- сегмент `{s['source']}`: {s['start']} → {s['end']}" for s in cov["segments"]],
             f"- одиниць: {len(cov['units'])}", "",
             "## Розбиття (варіант В)", "| частина | рядків | t від | t до | y_1 |", "|---|---|---|---|---|"]
    for part in ("train", "val", "test", "external"):
        d = L[L.split_V == part]
        lines.append(f"| {part} | {len(d)} | {d.t.min()} | {d.t.max()} | {d.y_1.mean():.3f} |")
    lines += ["", "## Вибір на validation",
              *[f"- H={h}: GBM {sel['selection'][str(h)]['gbm']['chosen']}, калібрування "
                f"{sel['selection'][str(h)]['gbm']['calibration']['chosen']}" for h in (1, 3, 6)], "",
              "## Validation, Brier: захід (сервіс = клімат) і решта (сервіс = GBM)",
              "GBM на validation — після калібрування на тій самій validation (in-sample для калібратора).", "",
              "| H | захід: клімат | захід: GBM | решта: GBM | решта: клімат |", "|---|---|---|---|---|"]
    for h in (1, 3, 6):
        lines.append(f"| {h} | {pick(val, 'western', 'climate', h):.4f} | {pick(val, 'western', 'gbm_cal_in_sample', h):.4f} | "
                     f"{pick(val, 'non_western', 'gbm_cal_in_sample', h):.4f} | {pick(val, 'non_western', 'climate', h):.4f} |")
    b = bt[(bt.model == "gbm_raw")].groupby(["horizon_h", "group"]).brier_skill_vs_climate.median().unstack()
    bt_md = ["| H | " + " | ".join(b.columns) + " |", "|---" * (len(b.columns) + 1) + "|",
             *[f"| {h} | " + " | ".join(f"{v:.3f}" for v in r) + " |" for h, r in b.iterrows()]]
    lines += ["", "## Backtest (медіана Brier skill GBM vs клімат по 5 вікнах)", "", *bt_md, "",
              "## Зовнішня перевірка (alerts.in.ua, інше визначення тривоги; НЕ test), Brier", "",
              "| H | захід: клімат | решта: GBM | решта: клімат | сервіс загалом |", "|---|---|---|---|---|"]
    ext = M[M.split == "external"]
    for h in (1, 3, 6):
        lines.append(f"| {h} | {pick(ext, 'western', 'climate', h):.4f} | {pick(ext, 'non_western', 'gbm', h):.4f} | "
                     f"{pick(ext, 'non_western', 'climate', h):.4f} | {pick(ext, 'overall', 'service', h):.4f} |")
    lines += ["", "## Кроки", "| крок | с |", "|---|---|", *[f"| {n} | {t:.0f} |" for n, t in timings], ""]
    (R / "pipeline_run.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="завантажити свіжі сирі дані")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    timings = []
    for name, fn in STEPS:
        t0 = time.time()
        print(f"\n=== {name}", flush=True)
        fn(args)
        timings.append((name, time.time() - t0))
    write_summary(timings)
    print("\nготово: reports/pipeline_run.md. Цей прогін test не оцінював (повторно його не запускати).")


if __name__ == "__main__":
    main()
