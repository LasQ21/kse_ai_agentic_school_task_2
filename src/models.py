"""Бейзлайни (клімат, persistence, логістична регресія) і gradient boosting з калібруванням."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES_GBM, FEATURES_LOGREG, WESTERN_OBLASTS

SEED = 0
CLIMATE_SHRINK = 10  # псевдоспостережень до середнього по одиниці


class Climate:
    """Частота старту за (одиниця × година тижня, Київ) на train, стягнута до частоти одиниці."""

    def fit(self, df, y):
        d = df.assign(y=np.asarray(y, float), how=df.dow * 24 + df.hour)
        unit_rate = d.groupby("unit").y.mean()
        cell = d.groupby(["unit", "how"]).y.agg(["sum", "count"])
        prior = cell.index.get_level_values("unit").map(unit_rate).to_numpy()
        self.table = (cell["sum"] + CLIMATE_SHRINK * prior) / (cell["count"] + CLIMATE_SHRINK)
        self.unit_rate, self.global_rate = unit_rate, d.y.mean()
        return self

    def predict(self, df):
        key = pd.MultiIndex.from_arrays([df.unit, df.dow * 24 + df.hour])
        p = self.table.reindex(key).to_numpy()
        fallback = df.unit.map(self.unit_rate).fillna(self.global_rate).to_numpy()
        return np.where(np.isnan(p), fallback, p)


class Persistence:
    """Чи була тривога в останні H год (активна в t або старт у [t-H, t)) -> частота на train."""

    def __init__(self, h):
        self.h = h

    def _x(self, df):
        return ((df.active == 1) | (df[f"starts_{self.h}h"] > 0)).to_numpy()

    def fit(self, df, y):
        x, y = self._x(df), np.asarray(y, float)
        self.rate = {True: y[x].mean(), False: y[~x].mean()}
        return self

    def predict(self, df):
        x = self._x(df)
        return np.where(x, self.rate[True], self.rate[False])


# Ознаки «хвилі атаки» для бази «клімат + хвиля»: скільки областей мали тривогу за останню годину
# і скільки одиниць під тривогою по країні в момент t (без власної одиниці).
WAVE_FEATURES = ["oblasts_touched_1h", "country_active_other"]


class ClimateWave:
    """База «клімат + хвиля»: логістична регресія на logit(клімат) і WAVE_FEATURES, без інших ознак.

    C=1 фіксоване (без тюнінгу); клімат береться вже навчений на train.
    """

    def __init__(self, climate: "Climate"):
        self.climate = climate

    def _x(self, df):
        return np.column_stack([_logit(self.climate.predict(df)), df[WAVE_FEATURES].to_numpy(float)])

    def fit(self, df, y):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(self._x(df), y)
        return self

    def predict(self, df):
        return self.m.predict_proba(self._x(df))[:, 1]


def make_logreg(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def gbm_features(use_unit: bool) -> list[str]:
    return FEATURES_GBM if use_unit else [f for f in FEATURES_GBM if f != "unit_code"]


def make_gbm(learning_rate, max_iter, max_leaf_nodes, min_samples_leaf=200, l2=1.0, use_unit=True):
    cat = [f == "unit_code" for f in gbm_features(use_unit)]
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2, categorical_features=cat,
        early_stopping=False, random_state=SEED)


# Перші 4 — початкова сітка; оптимум опинився на її межі (найпростіша модель), тому сітку
# розширено в бік сильнішої регуляризації і варіантів без unit_code (рішення — лише за val).
GBM_GRID = [
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31),
    dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31),
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=63),
    dict(learning_rate=0.1, max_iter=300, max_leaf_nodes=31),
    dict(learning_rate=0.05, max_iter=150, max_leaf_nodes=15, min_samples_leaf=500),
    dict(learning_rate=0.03, max_iter=300, max_leaf_nodes=15, min_samples_leaf=1000),
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, use_unit=False),
    dict(learning_rate=0.05, max_iter=150, max_leaf_nodes=15, min_samples_leaf=500, use_unit=False),
    dict(learning_rate=0.03, max_iter=300, max_leaf_nodes=15, min_samples_leaf=1000, use_unit=False),
    # Друге (останнє) розширення: ще сильніша регуляризація. Далі сітку не розширюємо,
    # щоб не перенавчитися на validation.
    dict(learning_rate=0.03, max_iter=300, max_leaf_nodes=7, min_samples_leaf=2000),
    dict(learning_rate=0.03, max_iter=500, max_leaf_nodes=15, min_samples_leaf=2000),
]
LOGREG_C = [0.01, 0.1, 1.0]


# ---------------------------------------------------------------- калібрування
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """method: 'none' | 'platt' (логістична на logit p) | 'isotonic'."""

    def __init__(self, method):
        self.method = method

    def fit(self, p, y):
        if self.method == "platt":
            self.m = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        return self

    def predict(self, p):
        if self.method == "platt":
            return self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.m.predict(p)
        return p


def choose_calibration(p_val, y_val, t_val) -> tuple[str, dict]:
    """Метод обираємо крос-фітингом на двох часових половинах validation (без test)."""
    mid = np.sort(t_val)[len(t_val) // 2]
    first = t_val < mid
    scores = {}
    for m in ("none", "platt", "isotonic"):
        ll = []
        for fit_mask in (first, ~first):
            c = Calibrator(m).fit(p_val[fit_mask], y_val[fit_mask])
            ll.append(log_loss(y_val[~fit_mask], np.clip(c.predict(p_val[~fit_mask]), 1e-6, 1 - 1e-6)))
        scores[m] = float(np.mean(ll))
    return min(scores, key=scores.get), scores


def enforce_monotone(preds: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], dict]:
    """P(H=1) <= P(H=3) <= P(H=6) за побудовою цілі. Повертає виправлені прогнози і частку порушень."""
    hs = sorted(preds)
    out, viol = {hs[0]: preds[hs[0]]}, {}
    for a, b in zip(hs, hs[1:]):
        viol[f"p{b}<p{a}"] = float(np.mean(preds[b] < out[a]))
        out[b] = np.maximum(preds[b], out[a])
    return out, viol


def service_predict(df: pd.DataFrame, models: dict) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Модель сервісу: клімат для західних областей, GBM (калібрований, монотонний за H) — для решти.

    Захід переведено на клімат, бо GBM там не переміг клімат (reports/model_report.md, §2а).
    Клімат монотонний за H за побудовою (перевірено на validation: 0 порушень), тож маршрутизація
    не ламає P1 <= P3 <= P6. Повертає прогнози за H і маску «прогноз з клімату».
    """
    hs = sorted(models)
    gbm = {h: models[h]["gbm_cal"].predict(
        models[h]["gbm"].predict_proba(df[models[h]["gbm_features"]])[:, 1]) for h in hs}
    gbm, _ = enforce_monotone(gbm)
    west = df.oblast.isin(WESTERN_OBLASTS).to_numpy()
    out = {h: np.where(west, models[h]["climate"].predict(df), gbm[h]) for h in hs}
    return out, west
