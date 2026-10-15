"""Ознаки для (одиниця, t). Лише інформація, доступна в момент t.

Правило доступності: подія відома, якщо started_at < t; її кінець відомий, лише якщо
finished_at < t. Інакше тривога в момент t вважається активною, а її тривалість невідомою.
Усі вікна напіввідкриті: [t - w, t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import KYIV_CITY

KYIV_TZ = "Europe/Kyiv"
COUNT_WINDOWS_H = (3, 6, 24, 72, 168)
CAP_H = 24 * 30  # обрізання для "годин з останньої події"

# Сусідство областей (спільний сухопутний кордон), задано вручну. Перевірки: симетрія і
# зв'язність — tests/test_leakage.py; узгодженість з даними — reports/model_report.md.
OBLAST_ADJACENCY = {
    "Волинська область": ["Рівненська область", "Львівська область"],
    "Рівненська область": ["Волинська область", "Львівська область", "Тернопільська область",
                           "Хмельницька область", "Житомирська область"],
    "Львівська область": ["Волинська область", "Рівненська область", "Тернопільська область",
                          "Івано-Франківська область", "Закарпатська область"],
    "Закарпатська область": ["Львівська область", "Івано-Франківська область"],
    "Івано-Франківська область": ["Львівська область", "Закарпатська область",
                                  "Тернопільська область", "Чернівецька область"],
    "Тернопільська область": ["Львівська область", "Рівненська область", "Хмельницька область",
                              "Чернівецька область", "Івано-Франківська область"],
    "Чернівецька область": ["Івано-Франківська область", "Тернопільська область",
                            "Хмельницька область", "Вінницька область"],
    "Хмельницька область": ["Рівненська область", "Житомирська область", "Вінницька область",
                            "Чернівецька область", "Тернопільська область"],
    "Житомирська область": ["Рівненська область", "Хмельницька область", "Вінницька область",
                            "Київська область"],
    "Вінницька область": ["Житомирська область", "Хмельницька область", "Чернівецька область",
                          "Одеська область", "Кіровоградська область", "Черкаська область",
                          "Київська область"],
    "Київська область": ["Житомирська область", "Вінницька область", "Черкаська область",
                         "Полтавська область", "Чернігівська область", KYIV_CITY],
    KYIV_CITY: ["Київська область"],
    "Чернігівська область": ["Київська область", "Полтавська область", "Сумська область"],
    "Сумська область": ["Чернігівська область", "Полтавська область", "Харківська область"],
    "Полтавська область": ["Київська область", "Чернігівська область", "Сумська область",
                           "Харківська область", "Дніпропетровська область",
                           "Кіровоградська область", "Черкаська область"],
    "Черкаська область": ["Київська область", "Вінницька область", "Кіровоградська область",
                          "Полтавська область"],
    "Кіровоградська область": ["Черкаська область", "Вінницька область", "Одеська область",
                               "Миколаївська область", "Дніпропетровська область",
                               "Полтавська область"],
    "Одеська область": ["Вінницька область", "Кіровоградська область", "Миколаївська область"],
    "Миколаївська область": ["Одеська область", "Кіровоградська область",
                             "Дніпропетровська область", "Херсонська область"],
    "Херсонська область": ["Миколаївська область", "Дніпропетровська область",
                           "Запорізька область", "Автономна Республіка Крим"],
    "Дніпропетровська область": ["Полтавська область", "Харківська область", "Донецька область",
                                 "Запорізька область", "Херсонська область",
                                 "Миколаївська область", "Кіровоградська область"],
    "Запорізька область": ["Дніпропетровська область", "Донецька область", "Херсонська область"],
    "Харківська область": ["Сумська область", "Полтавська область", "Дніпропетровська область",
                           "Донецька область", "Луганська область"],
    "Донецька область": ["Дніпропетровська область", "Запорізька область", "Харківська область",
                         "Луганська область"],
    "Луганська область": ["Харківська область", "Донецька область"],
    "Автономна Республіка Крим": ["Херсонська область"],
}

WESTERN_OBLASTS = {"Волинська область", "Рівненська область", "Львівська область",
                   "Тернопільська область", "Івано-Франківська область", "Закарпатська область",
                   "Чернівецька область"}


def _ns_series(col: pd.Series) -> np.ndarray:
    return col.dt.as_unit("ns").astype("int64").to_numpy()


def _per_unit(starts: np.ndarray, ends: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """Ознаки однієї одиниці. starts/ends — ns, події одиниці (не перетинаються після злиття)."""
    H = 3600 * 10**9
    if len(starts) == 0:  # у короткому вікні сервісу одиниця може не мати жодної події
        zeros = np.zeros(len(t), int)
        return {"active": np.zeros(len(t), bool), "active_elapsed_h": np.full(len(t), np.nan),
                "h_since_last_start": np.full(len(t), float(CAP_H)),
                "h_since_last_end": np.full(len(t), float(CAP_H)), "ended_last_1h": np.zeros(len(t), bool),
                **{f"starts_{w}h": zeros for w in COUNT_WINDOWS_H}, "starts_1h": zeros}
    o = np.argsort(starts)
    st, fn = starts[o], ends[o]
    n_known = np.searchsorted(st, t, side="left")             # події зі started_at < t
    k = n_known - 1
    has = k >= 0
    last_start = np.where(has, st[np.clip(k, 0, None)], 0)
    run_max_end = np.maximum.accumulate(fn) if len(fn) else fn
    # Активна: є відома подія, кінець якої ще не настав (finished_at >= t => невідомий).
    active = has & (run_max_end[np.clip(k, 0, None)] >= t) if len(fn) else np.zeros(len(t), bool)
    ends_sorted = np.sort(fn)
    m = np.searchsorted(ends_sorted, t, side="left") - 1      # кінці < t (відомі)
    last_end = np.where(m >= 0, ends_sorted[np.clip(m, 0, None)], 0)
    out = {
        "active": active,
        "active_elapsed_h": np.where(active, (t - last_start) / H, np.nan),
        "h_since_last_start": np.where(has, np.minimum((t - last_start) / H, CAP_H), CAP_H),
        "h_since_last_end": np.where(m >= 0, np.minimum((t - last_end) / H, CAP_H), CAP_H),
        "ended_last_1h": (m >= 0) & (t - last_end <= H),
    }
    for w in COUNT_WINDOWS_H:
        out[f"starts_{w}h"] = n_known - np.searchsorted(st, t - w * H, side="left")
    out["starts_1h"] = n_known - np.searchsorted(st, t - H, side="left")
    return out


def compute_features(events: pd.DataFrame, hromada: pd.DataFrame, units: list[str],
                     times: pd.DatetimeIndex) -> pd.DataFrame:
    """Повна матриця ознак для всіх units × times.

    events/hromada можуть містити майбутнє: функція сама відкидає все, що недоступне в t
    (це перевіряють тести, підставляючи змінене майбутнє).
    """
    t = times.as_unit("ns").asi8
    frames = []
    for u in units:
        e = events[events.unit == u]
        f = _per_unit(_ns_series(e.started_at), _ns_series(e.finished_at), t)
        h = hromada[hromada.unit == u]
        fh = _per_unit(_ns_series(h.started_at), _ns_series(h.finished_at), t) if len(h) else None
        f["hromada_starts_24h"] = fh["starts_24h"] if fh else np.zeros(len(t), int)
        f["hromada_active"] = fh["active"] if fh else np.zeros(len(t), bool)
        frames.append(pd.DataFrame({"unit": u, "t": times, **f}))
    F = pd.concat(frames, ignore_index=True)
    F["oblast"] = F.unit.str.split("|").str[0]

    # Агрегати по країні / області / сусідах — з тих самих ознак у момент t (без самої одиниці).
    touched = F.active | (F.starts_1h > 0) | F.ended_last_1h
    g_t = F.groupby("t")
    F["country_active_other"] = g_t.active.transform("sum") - F.active
    F["country_starts_1h_other"] = g_t.starts_1h.transform("sum") - F.starts_1h
    F["country_starts_6h_other"] = g_t.starts_6h.transform("sum") - F.starts_6h
    ob = F.assign(touched=touched).groupby(["t", "oblast"]).agg(
        ob_active=("active", "sum"), ob_units=("active", "size"), ob_touched=("touched", "max"))
    F["oblasts_touched_1h"] = F.t.map(ob.groupby(level="t").ob_touched.sum())  # "хвиля атаки"
    F = F.join(ob[["ob_active", "ob_units"]], on=["t", "oblast"])
    F["same_oblast_active_share"] = np.where(
        F.ob_units > 1, (F.ob_active - F.active) / (F.ob_units - 1).clip(lower=1), np.nan)
    share = (ob.ob_active / ob.ob_units).rename("share").reset_index()
    nb_rows = [(a, b) for a, bs in OBLAST_ADJACENCY.items() for b in bs]
    nb = pd.DataFrame(nb_rows, columns=["oblast", "neighbor"])
    nbs = nb.merge(share.rename(columns={"oblast": "neighbor"}), on="neighbor")
    agg = nbs.groupby(["t", "oblast"]).share.agg(neighbors_active_share="mean",
                                                 neighbors_any_active=lambda s: (s > 0).sum())
    F = F.join(agg, on=["t", "oblast"])
    F[["neighbors_active_share", "neighbors_any_active"]] = F[
        ["neighbors_active_share", "neighbors_any_active"]].fillna(0)

    local = F.t.dt.tz_convert(KYIV_TZ)
    F["hour"] = local.dt.hour
    F["dow"] = local.dt.dayofweek
    F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * F.hour / 24), np.cos(2 * np.pi * F.hour / 24)
    F["dow_sin"], F["dow_cos"] = np.sin(2 * np.pi * F.dow / 7), np.cos(2 * np.pi * F.dow / 7)
    return F.drop(columns=["ob_active", "ob_units", "ended_last_1h"])


FEATURES_GBM = [
    "unit_code", "hour", "dow", "active", "active_elapsed_h", "h_since_last_start",
    "h_since_last_end", "starts_1h", "starts_3h", "starts_6h", "starts_24h", "starts_72h", "starts_168h",
    "hromada_starts_24h", "hromada_active", "country_active_other", "country_starts_1h_other",
    "country_starts_6h_other", "oblasts_touched_1h", "same_oblast_active_share",
    "neighbors_active_share", "neighbors_any_active",
]
FEATURES_LOGREG = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "active", "log_starts_6h", "log_starts_24h",
    "log_starts_72h", "log_starts_168h", "log_h_since_last_start", "country_active_other",
    "oblasts_touched_1h", "neighbors_active_share",
]


def add_derived(F: pd.DataFrame, units: list[str]) -> pd.DataFrame:
    F = F.copy()
    F["unit_code"] = F.unit.map({u: i for i, u in enumerate(sorted(units))}).astype(int)
    for w in COUNT_WINDOWS_H:
        F[f"log_starts_{w}h"] = np.log1p(F[f"starts_{w}h"])
    F["log_h_since_last_start"] = np.log1p(F.h_since_last_start)
    F["active"] = F.active.astype(int)
    F["hromada_active"] = F.hromada_active.astype(int)
    return F


def build_feature_table() -> pd.DataFrame:
    """labels.parquet + ознаки -> data/processed/dataset.parquet (окремо по кожному сегменту)."""
    import json
    from src.data import PROCESSED_DIR
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    events = pd.read_parquet(PROCESSED_DIR / "events.parquet")
    hrom = pd.read_parquet(PROCESSED_DIR / "hromada_events.parquet")
    units = json.loads((PROCESSED_DIR / "coverage.json").read_text())["units"]
    parts = []
    for seg, lab in labels.groupby("segment"):
        times = pd.DatetimeIndex(lab.t.unique()).sort_values()
        F = compute_features(events[events.source == seg], hrom[hrom.source == seg], units, times)
        parts.append(lab.merge(F, on=["unit", "t"], how="left", validate="1:1"))
    D = add_derived(pd.concat(parts, ignore_index=True), units)
    D.to_parquet(PROCESSED_DIR / "dataset.parquet", index=False)
    return D
