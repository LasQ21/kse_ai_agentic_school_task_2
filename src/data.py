"""Завантаження сирих даних про тривоги.

Джерела (перевірені вручну 2026-10-15):
- Vadimkin/ukrainian-air-raid-sirens-dataset, official_data_uk.csv — історія з 2022-03-15,
  час у UTC, рівні oblast / raion / hromada. Офіційний файл перестав оновлюватися 2026-09-08.
- alerts.in.ua API: /v1/regions/{uid}/alerts/month_ago.json — історія лише за останній місяць,
  окремий ліміт 2 запити на хвилину. Тому кожне вивантаження зберігаємо як окремий знімок
  і ніколи не перезаписуємо: з часом знімки накопичують історію довшу за місяць.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

VADIMKIN_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
    "main/datasets/official_data_uk.csv"
)
ALERTS_IN_UA_API = "https://api.alerts.in.ua/v1"

# UID областей з документації devs.alerts.in.ua (розділ "Location UID").
OBLAST_UIDS = {
    3: "Хмельницька область", 4: "Вінницька область", 5: "Рівненська область",
    8: "Волинська область", 9: "Дніпропетровська область", 10: "Житомирська область",
    11: "Закарпатська область", 12: "Запорізька область", 13: "Івано-Франківська область",
    14: "Київська область", 15: "Кіровоградська область", 16: "Луганська область",
    17: "Миколаївська область", 18: "Одеська область", 19: "Полтавська область",
    20: "Сумська область", 21: "Тернопільська область", 22: "Харківська область",
    23: "Херсонська область", 24: "Черкаська область", 25: "Чернігівська область",
    26: "Чернівецька область", 27: "Львівська область", 28: "Донецька область",
    29: "Автономна Республіка Крим", 30: "м. Севастополь", 31: "м. Київ",
}
HISTORY_MIN_INTERVAL_S = 31  # ліміт history-ендпоінта: 2 запити на хвилину

log = logging.getLogger(__name__)


def _token() -> str:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("ALERTS_IN_UA_TOKEN")
    if not token:
        raise RuntimeError("ALERTS_IN_UA_TOKEN не задано (.env)")
    return token


def download_vadimkin(dest: Path | None = None) -> Path:
    """Завантажує офіційний датасет; ім'я файлу містить дату завантаження."""
    dest = dest or RAW_DIR / "vadimkin" / f"official_data_uk_{datetime.now(timezone.utc):%Y%m%d}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(VADIMKIN_URL, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    log.info("vadimkin: %s (%d bytes)", dest, len(r.content))
    return dest


def fetch_history(uid: int, token: str, retries: int = 3) -> dict:
    """Історія тривог області за ~30 діб з alerts.in.ua; на 429 чекає 65 с і повторює."""
    url = f"{ALERTS_IN_UA_API}/regions/{uid}/alerts/month_ago.json"
    for attempt in range(retries):
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        if r.status_code == 429:
            log.warning("uid=%s: 429, чекаю 65 с (спроба %d)", uid, attempt + 1)
            time.sleep(65)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"uid={uid}: перевищено ліміт після {retries} спроб")


def archive_alerts_in_ua_history(uids: list[int] | None = None) -> Path:
    """Зберігає знімок місячної історії по всіх областях у data/raw/alerts_in_ua/<timestamp>/."""
    token = _token()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RAW_DIR / "alerts_in_ua" / stamp
    out.mkdir(parents=True, exist_ok=True)
    for i, uid in enumerate(uids or list(OBLAST_UIDS)):
        if i:
            time.sleep(HISTORY_MIN_INTERVAL_S)
        payload = fetch_history(uid, token)
        payload["_fetched_at"] = datetime.now(timezone.utc).isoformat()
        (out / f"history_{uid}.json").write_text(json.dumps(payload, ensure_ascii=False))
        log.info("uid=%s %s: %d подій", uid, OBLAST_UIDS.get(uid), len(payload.get("alerts", [])))
    return out


# ---------------------------------------------------------------------------
# Очищення. Кожне правило обґрунтоване в reports/data_assumptions.md.
# ---------------------------------------------------------------------------
UTC = "UTC"
RAION_ERA_START = pd.Timestamp("2025-11-10", tz=UTC)   # з цього тижня >=99% тривог районні
VADIMKIN_RELIABLE_END = pd.Timestamp("2026-08-29", tz=UTC)  # з 08-29 збій офіційного каналу
MERGE_GAP_S = 60          # перерва <= 60 с між тривогами одного району = та сама тривога
STUCK_ALERT_H = 96        # тривоги довші за 96 год — "завислі" (кінець невідомий)
KYIV_CITY = "м. Київ"
# Постійні тривоги / окупація / відсутність різниці між районами (Донецька).
EXCLUDED_OBLASTS = {"Луганська область", "Автономна Республіка Крим", "м. Севастополь",
                    "Донецька область"}
# Області, де всі райони мають ідентичні тривоги (Jaccard = 1.00) -> одна одиниця на область.
OBLAST_AS_UNIT = {"Запорізька область", "Тернопільська область"}

# Рівні alerts.in.ua: red = "Повітряна тривога", yellow = "Попередня загроза".
# Vadimkin (канал @air_alert_ua) жовтого рівня не має. Яка комбінація відповідає офіційному
# каналу — НЕ встановлено (немає періоду перетину джерел). Тимчасово беремо обидва:
# сумарна частота найближча до очікуваної (див. reports/data_assumptions.md, п. 7).
ALERTS_IN_UA_LEVELS = {"red", "yellow"}

# Перейменування районів 2024 р.: у Vadimkin старі назви, в alerts.in.ua — нові.
RAION_ALIASES = {
    "Володимир-Волинський район": "Володимирський район",
    "Червоноградський район": "Шептицький район",
    "Красноградський район": "Берестинський район",
}

EVENT_COLS = ["oblast", "raion", "hromada", "level", "started_at", "finished_at", "source",
              "censored"]


def normalize_name(s):
    """Уніфікує апострофи й пробіли в назвах (’ ʼ ` ' -> ')."""
    if not isinstance(s, str):
        return s
    s = re.sub(r"[’ʼ`‘]", "'", s)
    s = re.sub(r"\s+", " ", s).strip()
    return RAION_ALIASES.get(s, s)


def load_vadimkin(path: Path) -> pd.DataFrame:
    """CSV Vadimkin -> EVENT_COLS: час у UTC, нормалізовані назви, source='vadimkin'."""
    df = pd.read_csv(path)
    for c in ("started_at", "finished_at"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    for c in ("oblast", "raion", "hromada"):
        df[c] = df[c].map(normalize_name)
    df["source"] = "vadimkin"
    df["censored"] = False
    return df[EVENT_COLS]


def load_alerts_in_ua(snapshot_dirs: list[Path]) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Об'єднує знімки history_*.json. Повертає події та межі гарантованого покриття [start, end).

    Покриття = найпізніший (fetch_time - 30 діб) серед усіх файлів найранішого знімку, щоб не
    вважати "тихими" години, які просто не потрапили у вікно month_ago. Час запиту береться з
    поля _fetched_at (пишеться з 2026-10-15), для старіших файлів — з mtime файлу.
    """
    recs, coverage_by_snapshot, fetch_times = [], [], []
    for d in sorted(snapshot_dirs):
        file_cov, fetch_times = [], []
        for f in sorted(d.glob("history_*.json")):
            payload = json.loads(f.read_text())
            fetched = (pd.Timestamp(payload["_fetched_at"]) if "_fetched_at" in payload
                       else pd.Timestamp(f.stat().st_mtime, unit="s", tz=UTC))
            file_cov.append(fetched - pd.Timedelta(days=30))
            fetch_times.append(fetched)
            for a in payload.get("alerts", []):
                recs.append({**a, "_fetched_at": fetched})
        coverage_by_snapshot.append(max(file_cov))
    # Знімки мають бути не рідше ніж раз на 30 діб, інакше між ними діра.
    coverage_start = min(coverage_by_snapshot)
    # Кінець = найраніший запит останнього знімку (після нього стан частини областей невідомий).
    coverage_end = min(fetch_times)  # fetch_times тут — файли останнього знімку
    return normalize_alerts_in_ua(pd.DataFrame(recs)), coverage_start, coverage_end


def normalize_alerts_in_ua(raw: pd.DataFrame) -> pd.DataFrame:
    """Записи API alerts.in.ua (+ колонка _fetched_at — час, на який відомий стан) -> EVENT_COLS.

    Спільна для архіву (load_alerts_in_ua) і живих даних (src/live.py), щоб не було skew.
    """
    if raw.empty:
        return pd.DataFrame(columns=EVENT_COLS)
    # Той самий id у кількох знімках -> беремо найсвіжіший стан.
    raw = raw.sort_values("_fetched_at").drop_duplicates("id", keep="last")
    raw = raw[(raw.alert_type == "air_raid") & raw.alert_level.isin(ALERTS_IN_UA_LEVELS)]
    level = raw.location_type.replace({"city": "hromada"})
    is_raion = raw.location_type == "raion"
    out = pd.DataFrame({
        "oblast": raw.location_oblast.map(normalize_name),
        "raion": raw.location_title.where(is_raion, raw.get("location_raion")).map(normalize_name),
        "hromada": raw.location_title.where(level == "hromada").map(normalize_name),
        "level": level,
        "started_at": pd.to_datetime(raw.started_at, utc=True, format="ISO8601"),
        "finished_at": pd.to_datetime(raw.finished_at, utc=True, format="ISO8601"),
        "source": "alerts_in_ua",
    })
    # Активна зараз тривога: кінець невідомий -> правостороннє цензурування моментом знімку.
    out["censored"] = out.finished_at.isna()
    out["finished_at"] = out.finished_at.fillna(pd.to_datetime(raw._fetched_at, utc=True, format="ISO8601"))
    out.loc[out.level == "oblast", "raion"] = None
    return out[EVENT_COLS].reset_index(drop=True)


def _merge_intervals(g: pd.DataFrame, gap_s: int) -> pd.DataFrame:
    g = g.sort_values("started_at")
    new_block = g.started_at > (g.finished_at.cummax().shift() + pd.Timedelta(seconds=gap_s))
    block = new_block.cumsum()
    return g.groupby(block).agg(
        started_at=("started_at", "min"), finished_at=("finished_at", "max"),
        source=("source", "first"), censored=("censored", "max"), n_merged=("started_at", "size"),
    )


def assign_unit(df: pd.DataFrame) -> pd.Series:
    """Одиниця прогнозу: 'область|район'; для м. Київ і OBLAST_AS_UNIT — 'область|*'."""
    whole = df.oblast.isin(OBLAST_AS_UNIT | {KYIV_CITY})
    return df.oblast + "|" + df.raion.where(~whole, "*").fillna("*")


def expand_oblast_alerts(df: pd.DataFrame, raions_by_oblast: dict | None = None) -> pd.DataFrame:
    """Обласна тривога (крім м. Київ) = тривога в кожному районі області.

    Перелік районів: явний (сервіс — з переліку одиниць моделі, бо у короткому вікні живих даних
    частина районів може не мати подій) або, за замовчуванням, з районних подій у самих даних.
    """
    if raions_by_oblast is None:
        raions = df[df.level == "raion"].groupby("oblast").raion.unique()
    else:
        raions = pd.Series({o: list(r) for o, r in raions_by_oblast.items()}, dtype=object)
        raions.index.name = "oblast"
    obl = df[(df.level == "oblast") & (df.oblast != KYIV_CITY) & df.oblast.isin(raions.index)]
    if obl.empty:
        return df
    expanded = obl.drop(columns="raion").merge(
        raions.explode().rename("raion").reset_index(), on="oblast")
    expanded["level"] = "raion"
    return pd.concat([df.drop(obl.index), expanded[df.columns]], ignore_index=True)


def raions_from_units(units: list[str]) -> dict[str, list[str]]:
    """'Область|Район' -> {Область: [Район, ...]}; для одиниць 'Область|*' — ['*']."""
    out: dict[str, list[str]] = {}
    for u in units:
        o, r = u.split("|", 1)
        out.setdefault(o, []).append(r)
    return out


def _coerce_times(df: pd.DataFrame) -> pd.DataFrame:
    """Час -> datetime64[us, UTC]. pandas 3: concat [us] і [ns] дає object, а рядки без дробових
    секунд парсяться як [s] — тоді запис межі вікна з мікросекундами падає (знайдено тестом API)."""
    df = df.copy()
    for c in ("started_at", "finished_at"):
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
        df[c] = df[c].dt.as_unit("us")
    return df


def clean_events(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                 raions_by_oblast: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Події цільового рівня (район; м. Київ — місто) у вікні [start, end).

    Повертає таблицю з колонками unit, oblast, raion, started_at, finished_at, source,
    censored, suspect_end, n_merged і словник з лічильниками кожного кроку очищення.
    """
    rep = {"rows_in": len(df)}
    df = _coerce_times(df).drop_duplicates()
    rep["exact_duplicates_dropped"] = rep["rows_in"] - len(df)
    df = df[(df.started_at < end) & (df.finished_at > start) & ~df.oblast.isin(EXCLUDED_OBLASTS)]
    n = len(df)
    df = expand_oblast_alerts(df, raions_by_oblast)
    rep["rows_added_by_oblast_expansion"] = len(df) - n
    target = (df.level == "raion") | ((df.level == "oblast") & (df.oblast == KYIV_CITY))
    df = df[target].copy()
    rep["target_level_rows_in_window"] = len(df)
    # Тривоги, що триватимуть за межею вікна, обрізаємо: їх кінець для нас невідомий.
    cut = df.finished_at > end
    df.loc[cut, "finished_at"] = end
    df.loc[cut, "censored"] = True
    rep["right_censored_at_window_end"] = int(cut.sum())
    dur = df.finished_at - df.started_at
    rep["nonpositive_duration_dropped"] = int((dur <= pd.Timedelta(0)).sum())
    df = df[dur > pd.Timedelta(0)]
    df["unit"] = assign_unit(df)
    n_before = len(df)
    merged = (df.groupby(["unit", "oblast"], group_keys=True)
                .apply(_merge_intervals, gap_s=MERGE_GAP_S, include_groups=False)
                .reset_index(level=[0, 1]).reset_index(drop=True))
    rep["rows_merged_away_gap_or_collapse"] = n_before - len(merged)
    merged["raion"] = merged.unit.str.split("|").str[1]
    merged["suspect_end"] = (merged.finished_at - merged.started_at) > pd.Timedelta(hours=STUCK_ALERT_H)
    rep["suspect_end_events"] = int(merged.suspect_end.sum())
    rep["events_out"] = len(merged)
    rep["units"] = merged.unit.nunique()
    cols = ["unit", "oblast", "raion", "started_at", "finished_at", "source", "censored",
            "suspect_end", "n_merged"]
    return merged[cols].sort_values(["started_at", "unit"]).reset_index(drop=True), rep


def hromada_events(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Тривоги громад — лише для ознак (не для цілі)."""
    df = _coerce_times(df)
    h = df[(df.level == "hromada") & (df.started_at < end) & (df.finished_at > start)
           & ~df.oblast.isin(EXCLUDED_OBLASTS)].drop_duplicates().copy()
    h["unit"] = assign_unit(h)
    return h[["unit", "oblast", "raion", "hromada", "started_at", "finished_at", "source",
              "censored"]].sort_values("started_at").reset_index(drop=True)


def build_processed(vadimkin_csv: Path | None = None) -> dict:
    """Сирі файли -> data/processed/{events,hromada_events}.parquet + coverage.json.

    coverage.json фіксує інтервали, де дані надійні; усе поза ними — діра, на якій не можна
    ні рахувати мітки, ні вважати, що тривог не було.
    """
    vadimkin_csv = vadimkin_csv or max((RAW_DIR / "vadimkin").glob("official_data_uk_*.csv"))
    v = load_vadimkin(vadimkin_csv)
    snaps = sorted(p for p in (RAW_DIR / "alerts_in_ua").iterdir() if p.is_dir())
    a, a_start, a_end = load_alerts_in_ua(snaps)

    ev_v, rep_v = clean_events(v, RAION_ERA_START, VADIMKIN_RELIABLE_END)
    ev_a, rep_a = clean_events(a, a_start, a_end)
    events = pd.concat([ev_v, ev_a], ignore_index=True)
    hrom = pd.concat([hromada_events(v, RAION_ERA_START, VADIMKIN_RELIABLE_END),
                      hromada_events(a, a_start, a_end)], ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(PROCESSED_DIR / "events.parquet", index=False)
    hrom.to_parquet(PROCESSED_DIR / "hromada_events.parquet", index=False)
    coverage = {
        "segments": [
            {"source": "vadimkin", "start": RAION_ERA_START.isoformat(),
             "end": VADIMKIN_RELIABLE_END.isoformat()},
            {"source": "alerts_in_ua", "start": a_start.isoformat(), "end": a_end.isoformat()},
        ],
        "suspect_end_intervals": [
            {"unit": r.unit, "start": r.started_at.isoformat(), "end": r.finished_at.isoformat()}
            for r in events[events.suspect_end].itertuples()
        ],
        "units": sorted(events.unit.unique()),
        "inputs": {"vadimkin": vadimkin_csv.name, "alerts_in_ua_snapshots": [s.name for s in snaps]},
        "cleaning_report": {"vadimkin": rep_v, "alerts_in_ua": rep_a},
    }
    (PROCESSED_DIR / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=1, default=str))
    return coverage
