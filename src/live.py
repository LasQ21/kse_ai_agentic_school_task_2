"""Живі дані: джерело тривог -> SQLite (data/live/alerts.sqlite).

Схема сховища — сирі записи у форматі alerts.in.ua; у події їх перетворює та сама
src.data.normalize_alerts_in_ua, що й архів (без копіпасту).

Активний ендпоінт віддає лише активні тривоги. Якщо тривога зникла з відповіді, її кінець
приблизно = момент опитування, коли її вперше не було (похибка <= інтервал опитування);
такі записи позначені finished_observed=1. Точний кінець можна підтягнути з історії
(reconcile_with_history), коли дозволяє ліміт API.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data import (ALERTS_IN_UA_API, OBLAST_UIDS, RAW_DIR, ROOT, fetch_history,
                      normalize_alerts_in_ua)

LIVE_DB = ROOT / "data" / "live" / "alerts.sqlite"
OPEN_END = "2100-01-01T00:00:00+00:00"
RAW_FIELDS = ["id", "location_title", "location_type", "location_oblast", "location_raion",
              "alert_type", "alert_level", "started_at", "finished_at"]
log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AlertSource(Protocol):
    """Джерело активних тривог для воркера."""
    name: str

    def fetch_active(self) -> list[dict]:
        """Активні тривоги у форматі записів alerts.in.ua (поля RAW_FIELDS)."""


class AlertsInUaSource:
    """alerts.in.ua: активні тривоги й історія; токен лише з .env / середовища."""
    name = "alerts_in_ua"

    def __init__(self, token: str | None = None):
        load_dotenv(ROOT / ".env")
        self.token = token or os.environ.get("ALERTS_IN_UA_TOKEN")
        if not self.token:
            raise RuntimeError("ALERTS_IN_UA_TOKEN не задано (.env)")

    def fetch_active(self) -> list[dict]:
        r = requests.get(f"{ALERTS_IN_UA_API}/alerts/active.json",
                         headers={"Authorization": f"Bearer {self.token}"}, timeout=30)
        r.raise_for_status()
        return r.json().get("alerts", [])

    def fetch_history(self, uid: int) -> list[dict]:
        return fetch_history(uid, self.token).get("alerts", [])


class UkraineAlarmSource:
    """Офіційне API (api.ukrainealarm.com) — місце під ключ UKRAINEALARM_API_KEY.

    Ключа немає й не буде (рішення 2026-10-16), тому формат відповіді не перевірено і мапінг
    у RAW_FIELDS не реалізовано навмисно.
    """
    name = "ukrainealarm"

    def __init__(self):
        load_dotenv(ROOT / ".env")
        self.key = os.environ.get("UKRAINEALARM_API_KEY")

    def fetch_active(self) -> list[dict]:
        raise NotImplementedError(
            "UkraineAlarmSource: потрібен ключ UKRAINEALARM_API_KEY і перевірка реальної структури "
            "відповіді перед мапінгом у формат RAW_FIELDS")


def make_source(name: str | None = None) -> AlertSource:
    """Джерело за назвою (аргумент або ALERT_SOURCE); за замовчуванням alerts_in_ua."""
    name = name or os.environ.get("ALERT_SOURCE", "alerts_in_ua")
    return {"alerts_in_ua": AlertsInUaSource, "ukrainealarm": UkraineAlarmSource}[name]()


class LiveStore:
    """SQLite-сховище сирих записів alerts.in.ua, журнал опитувань і помилок."""
    def __init__(self, path: Path = LIVE_DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                source TEXT NOT NULL, id TEXT NOT NULL,
                location_title TEXT, location_type TEXT, location_oblast TEXT, location_raion TEXT,
                alert_type TEXT, alert_level TEXT, started_at TEXT NOT NULL, finished_at TEXT,
                finished_observed INTEGER DEFAULT 0, first_seen TEXT, last_seen TEXT,
                PRIMARY KEY (source, id));
            CREATE TABLE IF NOT EXISTS polls (
                polled_at TEXT, source TEXT, kind TEXT, ok INTEGER, n INTEGER, error TEXT);
            """)

    def _conn(self):
        return sqlite3.connect(self.path)

    def _upsert(self, c, source, a, seen_at, exact_end: bool):
        vals = {k: (str(a.get(k)) if a.get(k) is not None else None) for k in RAW_FIELDS}
        c.execute("""
            INSERT INTO alerts (source, id, location_title, location_type, location_oblast,
                location_raion, alert_type, alert_level, started_at, finished_at,
                finished_observed, first_seen, last_seen)
            VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?)
            ON CONFLICT(source, id) DO UPDATE SET
                alert_level=excluded.alert_level, last_seen=excluded.last_seen,
                finished_at=CASE WHEN ? THEN excluded.finished_at ELSE alerts.finished_at END,
                finished_observed=CASE WHEN ? THEN 0 ELSE alerts.finished_observed END
            """, (source, vals["id"], vals["location_title"], vals["location_type"],
                  vals["location_oblast"], vals["location_raion"], vals["alert_type"],
                  vals["alert_level"], vals["started_at"], vals["finished_at"], seen_at, seen_at,
                  int(exact_end), int(exact_end)))

    def record_active(self, source: str, alerts: list[dict], polled_at: str) -> int:
        """Оновлює активні; ті, що були відкриті й зникли, закриває моментом цього опитування."""
        ids = {str(a["id"]) for a in alerts}
        with self._conn() as c:
            for a in alerts:
                self._upsert(c, source, a, polled_at, exact_end=False)
                # Тривога "мигнула" (зникла на одне опитування і повернулася) -> знову відкрита.
                c.execute("""UPDATE alerts SET finished_at=NULL, finished_observed=0
                             WHERE source=? AND id=? AND finished_observed=1""", (source, str(a["id"])))
            open_ids = [r[0] for r in c.execute(
                "SELECT id FROM alerts WHERE source=? AND finished_at IS NULL", (source,))]
            closed = [i for i in open_ids if i not in ids]
            c.executemany("UPDATE alerts SET finished_at=?, finished_observed=1 WHERE source=? AND id=?",
                          [(polled_at, source, i) for i in closed])
            c.execute("INSERT INTO polls VALUES (?,?,?,?,?,?)", (polled_at, source, "active", 1, len(alerts), None))
        return len(closed)

    def record_history(self, source: str, alerts: list[dict], fetched_at: str):
        """Історія має точні кінці — вони переписують наближені з опитувань."""
        with self._conn() as c:
            for a in alerts:
                self._upsert(c, source, a, fetched_at, exact_end=a.get("finished_at") is not None)
            c.execute("INSERT INTO polls VALUES (?,?,?,?,?,?)", (fetched_at, source, "history", 1, len(alerts), None))

    def record_error(self, source: str, kind: str, err: str):
        """Неуспішне опитування -> журнал polls (текст помилки обрізано до 500 символів)."""
        with self._conn() as c:
            c.execute("INSERT INTO polls VALUES (?,?,?,?,?,?)", (_now(), source, kind, 0, 0, err[:500]))

    def import_archive(self, archive_dirs: list[Path] | None = None):
        """Знімки data/raw/alerts_in_ua/* -> сховище (стартова історія для 168-годинних ознак)."""
        dirs = archive_dirs or sorted(p for p in (RAW_DIR / "alerts_in_ua").iterdir() if p.is_dir())
        for d in dirs:
            for f in sorted(d.glob("history_*.json")):
                payload = json.loads(f.read_text())
                fetched = payload.get("_fetched_at") or datetime.fromtimestamp(
                    f.stat().st_mtime, timezone.utc).isoformat()
                self.record_history("alerts_in_ua", payload.get("alerts", []), fetched)

    def last_ok_poll(self, kind: str = "active") -> pd.Timestamp | None:
        """Час останнього успішного опитування (за ним сервіс рахує вік даних)."""
        with self._conn() as c:
            r = c.execute("SELECT MAX(polled_at) FROM polls WHERE ok=1 AND kind=?", (kind,)).fetchone()[0]
        return pd.Timestamp(r) if r else None

    def raw_events(self, source: str = "alerts_in_ua") -> pd.DataFrame:
        """Події у форматі EVENT_COLS. Відкриті тривоги мають кінець OPEN_END (див. нижче)."""
        with self._conn() as c:
            raw = pd.read_sql_query("SELECT * FROM alerts WHERE source=?", c, params=(source,))
        if raw.empty:
            return normalize_alerts_in_ua(raw)
        # Відкрита на момент останнього опитування тривога має НЕВІДОМИЙ кінець: ставимо далеке
        # майбутнє, а clean_events обріже його моментом прогнозу (-> "активна в t"). Якщо ж
        # опитування зупинилося, це видно з data_age / stale у прогнозі, а не з "закінчених" тривог.
        raw["_fetched_at"] = raw.last_seen.where(raw.finished_at.notna(), OPEN_END)
        return normalize_alerts_in_ua(raw)


def poll_once(store: LiveStore, source: AlertSource) -> dict:
    """Одне опитування активних тривог; помилка записується в журнал, а не кидається."""
    polled_at = _now()
    try:
        alerts = source.fetch_active()
    except Exception as e:  # мережа / ліміт / ключ — фіксуємо, не падаємо
        store.record_error(source.name, "active", repr(e))
        log.warning("poll failed: %r", e)
        return {"ok": False, "error": repr(e)}
    closed = store.record_active(source.name, alerts, polled_at)
    return {"ok": True, "polled_at": polled_at, "active": len(alerts), "closed": closed}


def reconcile_with_history(store: LiveStore, source: AlertsInUaSource, uids=None, pause_s: float = 31):
    """Точні кінці з історії (ліміт 2 запити/хв -> ~14 хв на всі області)."""
    for i, uid in enumerate(uids or list(OBLAST_UIDS)):
        if i:
            time.sleep(pause_s)
        try:
            store.record_history(source.name, source.fetch_history(uid), _now())
        except Exception as e:
            store.record_error(source.name, "history", f"uid={uid}: {e!r}")


def run_poller(interval_s: int = 90, source_name: str | None = None):
    """Нескінченний цикл poll_once (для scripts/live_poll.py; воркер — scripts/live_worker.py)."""
    store, source = LiveStore(), make_source(source_name)
    while True:
        log.info("poll: %s", poll_once(store, source))
        time.sleep(interval_s)
