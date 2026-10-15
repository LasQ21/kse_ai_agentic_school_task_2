"""Метрики: загалом, по областях, макроусереднення, західні окремо, підмножина «тривоги немає»."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.features import WESTERN_OBLASTS

EPS = 1e-6


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    both = 0 < y.sum() < len(y)
    return {
        "n": len(y), "positives": int(y.sum()), "base_rate": y.mean() if len(y) else np.nan,
        "mean_pred": p.mean() if len(y) else np.nan,
        "brier": brier_score_loss(y, p) if len(y) else np.nan,
        "log_loss": log_loss(y, p, labels=[0, 1]) if len(y) else np.nan,
        "roc_auc": roc_auc_score(y, p) if both else np.nan,
        "pr_auc": average_precision_score(y, p) if both else np.nan,
    }


def grouped_metrics(df: pd.DataFrame, pred_cols: dict[str, str], y_col: str) -> pd.DataFrame:
    """df має колонки oblast, active, y_col і колонки прогнозів. Рядки: (група, модель)."""
    groups = {"overall": df, "western": df[df.oblast.isin(WESTERN_OBLASTS)],
              "non_western": df[~df.oblast.isin(WESTERN_OBLASTS)],
              "not_active_at_t": df[df.active == 0]}
    groups.update({f"oblast: {o}": g for o, g in df.groupby("oblast")})
    rows = []
    for gname, g in groups.items():
        for model, col in pred_cols.items():
            rows.append({"group": gname, "model": model, **metrics(g[y_col], g[col])})
    out = pd.DataFrame(rows)
    per_ob = out[out.group.str.startswith("oblast: ")]
    macro = per_ob.groupby("model")[["brier", "log_loss", "roc_auc", "pr_auc"]].mean().reset_index()
    macro["group"] = "macro_over_oblasts"
    out = pd.concat([out, macro], ignore_index=True)
    clim = out[out.model == "climate"].set_index("group").brier
    out["brier_skill_vs_climate"] = 1 - out.brier / out.group.map(clim)
    return out


def bootstrap_by_day(df: pd.DataFrame, y_col: str, a: str, b: str, n_boot=300, seed=0) -> dict:
    """Блочний бутстреп по днях: CI для Brier skill (a відносно b) і різниці ROC-AUC."""
    rng = np.random.default_rng(seed)
    codes, days = pd.factorize(df.t.dt.floor("D"))
    idx_by_day = {d: np.flatnonzero(codes == d) for d in range(len(days))}
    bss, dauc = [], []
    y, pa, pb = df[y_col].to_numpy(int), df[a].to_numpy(), df[b].to_numpy()
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_day[d] for d in rng.integers(0, len(days), len(days))])
        bss.append(1 - brier_score_loss(y[idx], pa[idx]) / brier_score_loss(y[idx], pb[idx]))
        dauc.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"brier_skill_ci95": q(bss), "roc_auc_diff_ci95": q(dauc), "n_days": len(days)}


def calibration_table(y, p, n_bins=10) -> pd.DataFrame:
    """Квантильні біни прогнозу: середній прогноз vs спостережена частота."""
    d = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)})
    d["bin"] = pd.qcut(d.p.rank(method="first"), n_bins, labels=False)
    return d.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
