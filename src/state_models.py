"""Моделі для бота: P(нова тривога за 24 год) і лічильник P(0) / P(1) / P(2+) нових тривог за h год.

Старі моделі (src/models.py, models/models.joblib, models/service_model.joblib) не змінюються й не
перенавчаються: P(нова тривога за 1 / 3 / 6 год) береться з них як є.

Лічильник будується так, щоб не суперечити основній ймовірності:
  P(0) = 1 − P(≥1),  P(2+) = P(≥1) · q_h,  P(1) = P(≥1) − P(2+),
де q_h = P(≥2 нових | ≥1 нової) — окрема модель, навчена на рядках з хоча б одним стартом.

Правило вибору зафіксоване до оцінки test і зовнішньої перевірки (scripts/train_state.py):
- гіперпараметри GBM / C лог. регресії — за log loss на validation;
- сімейство (GBM чи лог. регресія) — окремо для P(≥1, 24 год) і для q (одне на всі h), за log
  loss на validation серед незахідних областей;
- калібрування — choose_calibration на validation;
- захід — клімат (базовий кліматичний рівень).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import FEATURES_GBM, WESTERN_OBLASTS
from src.models import SEED
from src.state_features import STATE_FEATURES

# q_h: ознаки старої моделі + тривалості попередніх тривог і кількість областей під тривогою.
FEATURES_Q_GBM = FEATURES_GBM + STATE_FEATURES


def gbm_q_features(use_unit: bool) -> list[str]:
    """Ознаки GBM для q_h; без unit_code, якщо use_unit=False."""
    return FEATURES_Q_GBM if use_unit else [f for f in FEATURES_Q_GBM if f != "unit_code"]


def make_gbm_q(learning_rate, max_iter, max_leaf_nodes, min_samples_leaf=200, l2=1.0, use_unit=True):
    """Як src.models.make_gbm, але з ознаками q_h."""
    cat = [f == "unit_code" for f in gbm_q_features(use_unit)]
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2, categorical_features=cat,
        early_stopping=False, random_state=SEED)


def predict_family(m: dict, df: pd.DataFrame, kind: str) -> np.ndarray:
    """Сирий прогноз сімейства kind ('gbm' / 'logreg') із замороженого словника моделей."""
    if kind == "gbm":
        return m["gbm"].predict_proba(df[m["gbm_features"]])[:, 1]
    return m["logreg"].predict_proba(df[m["logreg_features"]])[:, 1]


def counter_from(p_any: dict[int, np.ndarray], q: dict[int, np.ndarray]) -> dict[str, dict[int, np.ndarray]]:
    """P(0), P(1), P(2+) з P(≥1) (не спадає з h) і q_h. P(2+) робиться неспадним за h
    (кумулятивний максимум) і не більшим за P(≥1)_h."""
    hs = sorted(p_any)
    p2, run = {}, None
    for h in hs:
        v = p_any[h] * q[h]
        run = v if run is None else np.maximum(run, v)
        p2[h] = np.minimum(run, p_any[h])
    return {"p0": {h: 1 - p_any[h] for h in hs}, "p1": {h: p_any[h] - p2[h] for h in hs}, "p2plus": p2}


def bot_predict(df: pd.DataFrame, bot: dict, old_models: dict) -> dict:
    """Прогноз для бота для всіх рядків df (ознаки з compute_features + add_derived +
    add_state_features).

    Повертає {"any": {1, 3, 6, 24: P(≥1 нової)}, "p0" / "p1" / "p2plus": {h: ...}, "west": маска}.
    P(≥1) на 1/3/6 — стара модель сервісу (src.models.service_predict) без змін; на 24 — нова
    (захід — клімат) з умовою P24 ≥ P6.
    """
    from src.models import service_predict
    any_, _ = service_predict(df, old_models)
    west = df.oblast.isin(WESTERN_OBLASTS).to_numpy()
    a = bot["any24"]
    p24 = np.where(west, a["climate"].predict(df), a["cal"].predict(predict_family(a, df, bot["family_any24"])))
    any_ = {**any_, 24: np.maximum(p24, any_[6])}
    q = {h: np.where(west, m["climate"].predict(df), m["cal"].predict(predict_family(m, df, bot["family_q"])))
         for h, m in bot["q"].items()}
    return {"any": any_, **counter_from(any_, q), "west": west}


def ece(y, p, n_bins=10) -> float:
    """Expected calibration error: рівноширокі біни [0, 1], вага — частка рядків у біні."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum(np.mean(b == k) * abs(p[b == k].mean() - y[b == k].mean())
                     for k in range(n_bins) if (b == k).any()))


def reliability(y, p, n_bins=10) -> pd.DataFrame:
    """Рівноширокі біни: середній прогноз, спостережена частота, кількість рядків."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["bin"] = np.clip((d.p * n_bins).astype(int), 0, n_bins - 1)
    return d.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def counter_loss(n: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Поточкові втрати лічильника: багатокласний Brier (сума по класах 0 / 1 / 2+) і log loss.
    n — фактична кількість нових тривог, P — (рядки × 3)."""
    k = np.minimum(np.asarray(n, int), 2)
    Y = np.eye(3)[k]
    P = np.clip(P, 1e-6, 1)
    return ((P - Y) ** 2).sum(1), -np.log(P[np.arange(len(k)), k])


def day_bootstrap(t: pd.Series, losses: dict[str, np.ndarray], pairs, n_boot=1000, seed=0) -> dict:
    """Блочний бутстреп по днях (UTC). losses — поточкові втрати (напр. (p − y)² чи багатокласний
    Brier). 95% CI для середньої втрати кожної моделі й skill = 1 − L(a) / L(b) для пар (a, b)."""
    rng = np.random.default_rng(seed)
    codes, days = pd.factorize(pd.Series(t).dt.floor("D"))
    nd = len(days)
    idx = rng.integers(0, nd, size=(n_boot, nd))
    se = {k: np.bincount(codes, weights=v, minlength=nd) for k, v in losses.items()}
    cnt = np.bincount(codes, minlength=nd)
    n_b = cnt[idx].sum(1)

    def q(v):
        return float(np.quantile(v, .025)), float(np.quantile(v, .975))
    out = {"n_days": nd}
    for k in losses:
        out[f"loss_{k}"] = (float(se[k].sum() / cnt.sum()), *q(se[k][idx].sum(1) / n_b))
    for a, b in pairs:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = 1 - se[a][idx].sum(1) / se[b][idx].sum(1)
        out[f"skill_{a}_vs_{b}"] = (float(1 - se[a].sum() / se[b].sum()), *q(s[np.isfinite(s)]))
    return out
