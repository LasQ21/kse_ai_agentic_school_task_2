"""Синтетичний зразок даних для тестів на чистому клоні (без data/ і models/).

Події вигадані: випадкові «хвилі» тривог по всіх 105 одиницях (перелік — tests/fixtures/units.json),
у форматі сирого CSV Vadimkin. Далі той самий шлях, що й у пайплайні: src.data.clean_events /
hromada_events -> src.split.build_label_grid / assign_split -> src.features.compute_features /
add_derived -> модель сервісу тієї ж структури, що й scripts/export_service_model.py.
Числа якості на зразку нічого не означають; він лише перевіряє код.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import KYIV_CITY, OBLAST_AS_UNIT, clean_events, hromada_events, load_vadimkin
from src.features import WESTERN_OBLASTS, add_derived, compute_features
from src.models import Calibrator, Climate, ClimateWave, gbm_features, make_gbm, make_logreg
from src.split import HORIZONS, SPLITS, assign_split, build_label_grid
from src.state_features import add_state_features
from src.state_models import (FEATURES_A_LOGREG, FEATURES_B_LOGREG, MedianRule, SurvivalTable,
                              add_state_derived, gbm_a_features, gbm_b_features, make_gbm_a, make_gbm_b,
                              make_logreg_b)
from src.state_targets import STATE_HORIZONS, build_state_labels

UNITS = json.loads((Path(__file__).parent / "fixtures" / "units.json").read_text())
# Сегмент: 7 діб розігріву, train 25.05–13.06, val 15.06–30.06 (межі SPLITS["V"]).
SEG_START, SEG_END = pd.Timestamp("2026-05-18", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
GBM_PARAMS = dict(learning_rate=0.03, max_iter=300, max_leaf_nodes=7, min_samples_leaf=2000)
CALIBRATION = {1: "platt", 3: "platt", 6: "none"}  # як обрано на validation для справжньої моделі


def raw_events(seed: int = 0) -> pd.DataFrame:
    """Сирі події у форматі CSV Vadimkin: районні, обласні (розгортаються на райони), громадські."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(SEG_START, SEG_END, freq="h", inclusive="left")
    oblasts = sorted({u.split("|")[0] for u in UNITS})
    rate = {o: (0.01 if o in WESTERN_OBLASTS else 0.05) for o in oblasts}
    rows = []

    def add(oblast, raion, hromada, level, start, minutes):
        rows.append({"oblast": oblast, "raion": raion, "hromada": hromada, "level": level,
                     "started_at": start, "finished_at": start + pd.Timedelta(minutes=minutes),
                     "source": "official"})

    for t in hours:
        wave = rng.random() < 0.15                       # година «хвилі»: ризик вищий скрізь
        for u in UNITS:
            o, r = u.split("|")
            if rng.random() < rate[o] * (6 if wave else 1):
                start = t + pd.Timedelta(seconds=int(rng.integers(0, 3600)))
                # рідкі довгі тривоги (≥ 25 год), щоб мітка «триває ще 24 год» мала обидва класи
                minutes = int(rng.integers(1500, 2400)) if rng.random() < 0.01 else int(rng.integers(10, 180))
                if o == KYIV_CITY:
                    add(o, None, None, "oblast", start, minutes)
                elif o in OBLAST_AS_UNIT:
                    add(o, None, None, "oblast", start, minutes)
                elif rng.random() < 0.1:
                    add(o, r, f"громада {r}", "hromada", start, minutes)
                else:
                    add(o, r, None, "raion", start, minutes)
    # Кожен район має хоч одну районну подію: офлайн перелік районів для обласних тривог береться
    # з даних, а сервіс — з переліку одиниць; так вони збігаються.
    for u in UNITS:
        o, r = u.split("|")
        if r != "*":
            add(o, r, None, "raion", SEG_START + pd.Timedelta(hours=1), 15)
    # Сирі райони для областей-одиниць (у переліку одиниць їх немає).
    for o in OBLAST_AS_UNIT:
        add(o, "Синтетичний район", None, "raion", SEG_START + pd.Timedelta(hours=2), 15)
    df = pd.DataFrame(rows)
    for c in ("started_at", "finished_at"):
        df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    return df


def build(root: Path, seed: int = 0) -> dict:
    """Пише в root: raw/vadimkin/official_data_uk_sample.csv, processed/dataset.parquet,
    models/service_model.joblib. Повертає шляхи."""
    raw_csv = root / "raw" / "vadimkin" / "official_data_uk_sample.csv"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_events(seed).to_csv(raw_csv, index=False)
    v = load_vadimkin(raw_csv)

    ev, _ = clean_events(v, SEG_START, SEG_END)
    hrom = hromada_events(v, SEG_START, SEG_END)
    assert sorted(ev.unit.unique()) == sorted(UNITS), "у зразку мають бути всі 105 одиниць"
    coverage = {"segments": [{"source": "vadimkin", "start": SEG_START.isoformat(), "end": SEG_END.isoformat()}],
                "suspect_end_intervals": [], "units": sorted(UNITS)}
    labels = build_label_grid(ev, coverage)
    labels["split_V"] = assign_split(labels, SPLITS["V"])
    times = pd.DatetimeIndex(labels.t.unique()).sort_values()
    F = compute_features(ev, hrom, coverage["units"], times)
    D = add_derived(labels.merge(F, on=["unit", "t"], how="left", validate="1:1"), coverage["units"])
    (root / "processed").mkdir(exist_ok=True)
    dataset = root / "processed" / "dataset.parquet"
    D.to_parquet(dataset, index=False)

    tr, va = D[D.split_V == "train"], D[D.split_V == "val"]
    feats = gbm_features(True)
    models = {}
    for h in HORIZONS:
        y_tr, y_va = tr[f"y_{h}"].to_numpy(int), va[f"y_{h}"].to_numpy(int)
        gbm = make_gbm(**GBM_PARAMS).fit(tr[feats], y_tr)
        cal = Calibrator(CALIBRATION[h]).fit(gbm.predict_proba(va[feats])[:, 1], y_va)
        models[h] = {"climate": Climate().fit(tr, y_tr), "gbm": gbm, "gbm_features": feats, "gbm_cal": cal}
    meta = {"sample": True, "units": coverage["units"], "train_range": [str(tr.t.min()), str(tr.t.max())],
            "routing": {"climate": sorted(WESTERN_OBLASTS), "gbm": "усі інші одиниці"}}
    model = root / "models" / "service_model.joblib"
    model.parent.mkdir(exist_ok=True)
    joblib.dump({"models": models, "meta": meta}, model)
    dataset_state, state_model = build_state(root, ev, coverage, D)
    return {"raw_csv": raw_csv, "dataset": dataset, "service_model": model,
            "dataset_state": dataset_state, "state_model": state_model}


def build_state(root: Path, ev: pd.DataFrame, coverage: dict, D: pd.DataFrame):
    """Цілі за станом на зразку тим самим шляхом (src.state_targets / src.state_features) і
    невелика модель станів тієї ж структури, що й scripts/train_state.py (GBM, Platt)."""
    L, _ = build_state_labels(ev, coverage)
    X = L.merge(D.drop(columns=["segment", "split_V"]), on=["unit", "t"], how="left", validate="1:1")
    X = add_state_derived(add_state_features(X, ev))
    path = root / "processed" / "dataset_state.parquet"
    X.to_parquet(path, index=False)
    K = X[X.keep]
    A, B = K[K.state == "A"], K[K.state == "B"]
    trA, vaA, trB, vaB = (d[d.split_S == p] for d in (A, B) for p in ("train", "val"))
    durs = pd.DataFrame({"oblast": ev.oblast, "duration_h": (ev.finished_at - ev.started_at).dt.total_seconds() / 3600})

    def fam(tr, va, y, feats_gbm, make_gbm_fn, feats_lr, make_lr_fn):
        g = make_gbm_fn(**GBM_PARAMS).fit(tr[feats_gbm], tr[y].astype(int))
        cal = Calibrator("platt").fit(g.predict_proba(va[feats_gbm])[:, 1], va[y].astype(int))
        lr = make_lr_fn(1.0).fit(tr[feats_lr], tr[y].astype(int))
        return {"gbm": g, "gbm_features": feats_gbm, "logreg": lr, "logreg_features": feats_lr, "cal": cal,
                "climate": Climate().fit(tr, tr[y].astype(int))}

    a24 = fam(trA, vaA, "y_start_24", gbm_a_features(True), make_gbm_a, FEATURES_A_LOGREG, make_logreg)
    a24["climate_wave"] = ClimateWave(a24["climate"]).fit(trA, trA.y_start_24.astype(int))
    b = {}
    for h in STATE_HORIZONS:
        b[h] = fam(trB, vaB, f"y_cont_{h}", gbm_b_features(True), make_gbm_b, FEATURES_B_LOGREG, make_logreg_b)
        b[h].update(survival=SurvivalTable().fit(trB, trB[f"y_cont_{h}"]), median_rule=MedianRule(h).fit(durs))
    model = root / "models" / "state_models.joblib"
    joblib.dump({"A24": a24, "B": b, "family_A": "gbm", "family_B": "gbm", "meta": {"sample": True}}, model)
    return path, model
