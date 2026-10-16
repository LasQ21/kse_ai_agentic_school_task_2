"""Моделі й бази для цілей за станом (A: старт за 24 год; B: тривога триває ще h год).

Старі моделі (src/models.py, models/*.joblib) тут не змінюються й не перенавчаються.
Правило вибору для сервісу зафіксоване до оцінки test і зовнішньої перевірки (scripts/train_state.py):
- гіперпараметри GBM / C логістичної регресії — за log loss на validation;
- сімейство (GBM чи лог. регресія) — одне на стан, за сумарним log loss на validation серед
  незахідних областей (для B — сума по h ∈ {1, 3, 6, 24});
- калібрування — choose_calibration на validation;
- захід — клімат (базовий кліматичний рівень).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models import CLIMATE_SHRINK, SEED

FEATURES_B_GBM = [
    "unit_code", "active_elapsed_h", "hour", "dow", "oblasts_active_now", "country_active_other",
    "same_oblast_active_share", "neighbors_active_share", "neighbors_any_active", "prev_dur_last_h",
    "prev_dur_med10_h", "prev_n_ended_30d", "elapsed_over_med", "starts_24h", "starts_168h",
]
FEATURES_B_LOGREG = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "log_elapsed_h", "log_prev_dur_last_h",
    "log_prev_dur_med10_h", "log_prev_n_ended_30d", "log_elapsed_over_med", "oblasts_active_now",
    "country_active_other", "neighbors_active_share",
]
ELAPSED_BINS_H = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 9, 12, 18, 24, 48, 96, np.inf]


def add_state_derived(df: pd.DataFrame) -> pd.DataFrame:
    """log1p-перетворення для лог. регресії стану B (NaN лишаються NaN -> імпутація в пайплайні)."""
    df = df.copy()
    for c in ("elapsed_h", "prev_dur_last_h", "prev_dur_med10_h", "prev_n_ended_30d", "elapsed_over_med"):
        src = "active_elapsed_h" if c == "elapsed_h" else c
        df[f"log_{c}"] = np.log1p(df[src])
    return df


def gbm_b_features(use_unit: bool) -> list[str]:
    """Ознаки GBM для стану B; без unit_code, якщо use_unit=False."""
    return FEATURES_B_GBM if use_unit else [f for f in FEATURES_B_GBM if f != "unit_code"]


def make_gbm_b(learning_rate, max_iter, max_leaf_nodes, min_samples_leaf=200, l2=1.0, use_unit=True):
    """Як src.models.make_gbm, але з ознаками стану B."""
    cat = [f == "unit_code" for f in gbm_b_features(use_unit)]
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2, categorical_features=cat,
        early_stopping=False, random_state=SEED)


def make_logreg_b(C):
    """Лог. регресія з імпутацією медіаною (і індикатором пропуску) та стандартизацією."""
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


class SurvivalTable:
    """Емпірична виживаність: частка «триває ще h» серед train-рядків стану B з тим самим
    (область, кошик пройденого часу, година), стягнута до (область, кошик) і далі до (кошик)."""

    def _keys(self, df):
        return df.oblast.to_numpy(), pd.cut(df.active_elapsed_h, ELAPSED_BINS_H, right=False,
                                            labels=False).to_numpy(), df.hour.to_numpy()

    def fit(self, df, y):
        o, b, h = self._keys(df)
        d = pd.DataFrame({"o": o, "b": b, "h": h, "y": np.asarray(y, float)})
        self.global_ = d.y.mean()
        g_b = d.groupby("b").y.agg(["sum", "count"])
        self.p_b = (g_b["sum"] + CLIMATE_SHRINK * self.global_) / (g_b["count"] + CLIMATE_SHRINK)
        g_ob = d.groupby(["o", "b"]).y.agg(["sum", "count"])
        prior = g_ob.index.get_level_values("b").map(self.p_b).to_numpy()
        self.p_ob = (g_ob["sum"] + CLIMATE_SHRINK * prior) / (g_ob["count"] + CLIMATE_SHRINK)
        g = d.groupby(["o", "b", "h"]).y.agg(["sum", "count"])
        prior = self.p_ob.reindex(pd.MultiIndex.from_arrays([g.index.get_level_values("o"),
                                                             g.index.get_level_values("b")])).to_numpy()
        self.p_obh = (g["sum"] + CLIMATE_SHRINK * prior) / (g["count"] + CLIMATE_SHRINK)
        return self

    def predict(self, df):
        o, b, h = self._keys(df)
        p = self.p_obh.reindex(pd.MultiIndex.from_arrays([o, b, h])).to_numpy()
        p_ob = self.p_ob.reindex(pd.MultiIndex.from_arrays([o, b])).to_numpy()
        p_b = pd.Series(b).map(self.p_b).fillna(self.global_).to_numpy()
        return np.where(np.isnan(p), np.where(np.isnan(p_ob), p_b, p_ob), p)


class MedianRule:
    """«Триватиме до медіани тривалості»: 1, якщо пройдений час + h < медіани тривалості
    train-епізодів області, інакше 0 (детерміноване правило, не ймовірність)."""

    def __init__(self, h):
        self.h = h

    def fit(self, durations: pd.DataFrame):
        """durations: колонки oblast, duration_h (train-епізоди)."""
        self.med = durations.groupby("oblast").duration_h.median()
        self.global_ = durations.duration_h.median()
        return self

    def predict(self, df):
        m = df.oblast.map(self.med).fillna(self.global_).to_numpy()
        return (df.active_elapsed_h.to_numpy() + self.h < m).astype(float)


def monotone_start(p24: np.ndarray, p6: np.ndarray) -> tuple[np.ndarray, float]:
    """P(старт за 24 год) ≥ P(старт за 6 год). Повертає виправлене p24 і частку порушень."""
    return np.maximum(p24, p6), float(np.mean(p24 < p6))


def monotone_cont(preds: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], dict]:
    """P(триває ≥ h) не зростає з h: кумулятивний мінімум за зростанням h."""
    hs = sorted(preds)
    out, viol = {hs[0]: preds[hs[0]]}, {}
    for a, b in zip(hs, hs[1:]):
        viol[f"p{b}>p{a}"] = float(np.mean(preds[b] > out[a]))
        out[b] = np.minimum(preds[b], out[a])
    return out, viol


def ece(y, p, n_bins=10) -> float:
    """Expected calibration error: рівноширокі біни [0, 1], вага — частка рядків у біні."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    tot = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def reliability(y, p, n_bins=10) -> pd.DataFrame:
    """Рівноширокі біни: середній прогноз, спостережена частота, кількість рядків."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["bin"] = np.clip((d.p * n_bins).astype(int), 0, n_bins - 1)
    return d.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def day_bootstrap(t: pd.Series, y: np.ndarray, preds: dict[str, np.ndarray], pairs, n_boot=1000, seed=0) -> dict:
    """Блочний бутстреп по днях (UTC): 95% CI для Brier кожної моделі й Brier skill пар (a, b)."""
    rng = np.random.default_rng(seed)
    codes, days = pd.factorize(pd.Series(t).dt.floor("D"))
    nd = len(days)
    idx = rng.integers(0, nd, size=(n_boot, nd))
    se = {k: np.bincount(codes, weights=(p - y) ** 2, minlength=nd) for k, p in preds.items()}
    cnt = np.bincount(codes, minlength=nd)
    n_b = cnt[idx].sum(1)
    out = {"n_days": nd}
    for k in preds:
        b = se[k][idx].sum(1) / n_b
        out[f"brier_{k}"] = (float(se[k].sum() / cnt.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975)))
    for a, b in pairs:
        s = 1 - se[a][idx].sum(1) / se[b][idx].sum(1)
        out[f"skill_{a}_vs_{b}"] = (float(1 - se[a].sum() / se[b].sum()), float(np.quantile(s, .025)),
                                    float(np.quantile(s, .975)))
    return out


def predict_family(m: dict, df: pd.DataFrame, kind: str) -> np.ndarray:
    """Сирий прогноз сімейства kind ('gbm' / 'logreg') з замороженого словника моделей одного h."""
    if kind == "gbm":
        return m["gbm"].predict_proba(df[m["gbm_features"]])[:, 1]
    return m["logreg"].predict_proba(df[m["logreg_features"]])[:, 1]


def state_service_predict(df: pd.DataFrame, state: dict, old_models: dict) -> dict:
    """Прогноз сервісу для всіх рядків df (ознаки з compute_features + add_derived +
    add_state_features + add_state_derived).

    Повертає {"start": {1, 3, 6, 24: p}, "cont": {1, 3, 6, 24: p}, "west": маска}. Старт 1/3/6 —
    стара модель сервісу (src.models.service_predict) без змін; старт 24 — нова модель (захід —
    клімат) з умовою P24 ≥ P6. Продовження — нова модель (захід — клімат), не зростає з h.
    Який стан у рядку, вирішує викликач (колонка active).
    """
    from src.features import WESTERN_OBLASTS
    from src.models import service_predict
    start, west = service_predict(df, old_models)
    west = df.oblast.isin(WESTERN_OBLASTS).to_numpy()
    a = state["A24"]
    p24 = np.where(west, a["climate"].predict(df), a["cal"].predict(predict_family(a, df, state["family_A"])))
    start = {**start, 24: monotone_start(p24, start[6])[0]}
    cont = {}
    for h, m in state["B"].items():
        cont[h] = np.where(west, m["climate"].predict(df), m["cal"].predict(predict_family(m, df, state["family_B"])))
    cont, _ = monotone_cont(cont)
    return {"start": start, "cont": cont, "west": west}
