"""Живе сховище в Postgres (Neon): те саме, що LiveStore на SQLite, але доступне і воркеру, і Vercel.

Воркер пише сюди (scripts/live_worker.py), webhook бота на Vercel читає (src/service.live_forecaster).
DSN — лише змінна середовища DATABASE_URL (для Neon краще адреса з `-pooler`). Семантика
record_active / record_history збігається з LiveStore; різниця лише в синтаксисі SQL.
"""
from __future__ import annotations

import pandas as pd

from src.data import normalize_alerts_in_ua
from src.live import OPEN_END, RAW_FIELDS, _now

COLUMNS = ["source", "id", *RAW_FIELDS[1:], "finished_observed", "first_seen", "last_seen"]
# Тривоги старші за це й закриті сервісу не потрібні (LOOKBACK = 31 доба); відкриті читаємо завжди.
READ_DAYS = 45


class PgLiveStore:
    """Postgres-сховище сирих записів alerts.in.ua, журнал опитувань і помилок."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        with self._conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                source TEXT NOT NULL, id TEXT NOT NULL,
                location_title TEXT, location_type TEXT, location_oblast TEXT, location_raion TEXT,
                alert_type TEXT, alert_level TEXT, started_at TEXT NOT NULL, finished_at TEXT,
                finished_observed INTEGER DEFAULT 0, first_seen TEXT, last_seen TEXT,
                PRIMARY KEY (source, id))""")
            c.execute("""
            CREATE TABLE IF NOT EXISTS polls (
                polled_at TEXT, source TEXT, kind TEXT, ok INTEGER, n INTEGER, error TEXT)""")
            c.execute("CREATE INDEX IF NOT EXISTS polls_ok ON polls (kind, ok, polled_at)")

    def _conn(self):
        import psycopg
        return psycopg.connect(self.dsn, connect_timeout=15)

    @staticmethod
    def _row(source, a, seen_at):
        vals = {k: (str(a.get(k)) if a.get(k) is not None else None) for k in RAW_FIELDS}
        return (source, vals["id"], vals["location_title"], vals["location_type"], vals["location_oblast"],
                vals["location_raion"], vals["alert_type"], vals["alert_level"], vals["started_at"],
                vals["finished_at"], seen_at, seen_at)

    _UPSERT = """
        INSERT INTO alerts (source, id, location_title, location_type, location_oblast, location_raion,
            alert_type, alert_level, started_at, finished_at, finished_observed, first_seen, last_seen)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s,%s)
        ON CONFLICT (source, id) DO UPDATE SET
            alert_level=EXCLUDED.alert_level, last_seen=EXCLUDED.last_seen,
            finished_at=CASE WHEN %s THEN EXCLUDED.finished_at ELSE alerts.finished_at END,
            finished_observed=CASE WHEN %s THEN 0 ELSE alerts.finished_observed END"""

    def _upsert_many(self, c, source, alerts, seen_at, exact):
        """exact: bool або функція alert -> bool (чи кінець у записі точний)."""
        rows = []
        for a in alerts:
            e = bool(exact(a) if callable(exact) else exact)
            rows.append((*self._row(source, a, seen_at), e, e))
        c.cursor().executemany(self._UPSERT, rows)

    def record_active(self, source: str, alerts: list[dict], polled_at: str) -> int:
        """Оновлює активні; ті, що були відкриті й зникли, закриває моментом цього опитування."""
        ids = [str(a["id"]) for a in alerts]
        with self._conn() as c:
            self._upsert_many(c, source, alerts, polled_at, False)
            # Тривога "мигнула" (зникла на одне опитування і повернулася) -> знову відкрита.
            c.execute("""UPDATE alerts SET finished_at=NULL, finished_observed=0
                         WHERE source=%s AND id = ANY(%s) AND finished_observed=1""", (source, ids))
            closed = c.execute("""UPDATE alerts SET finished_at=%s, finished_observed=1
                                  WHERE source=%s AND finished_at IS NULL AND NOT (id = ANY(%s))""",
                               (polled_at, source, ids)).rowcount
            c.execute("INSERT INTO polls VALUES (%s,%s,%s,%s,%s,%s)",
                      (polled_at, source, "active", 1, len(alerts), None))
        return closed

    def record_history(self, source: str, alerts: list[dict], fetched_at: str):
        """Історія має точні кінці — вони переписують наближені з опитувань."""
        with self._conn() as c:
            self._upsert_many(c, source, alerts, fetched_at, lambda a: a.get("finished_at") is not None)
            c.execute("INSERT INTO polls VALUES (%s,%s,%s,%s,%s,%s)",
                      (fetched_at, source, "history", 1, len(alerts), None))

    def record_error(self, source: str, kind: str, err: str):
        """Неуспішне опитування -> журнал polls (текст помилки обрізано до 500 символів)."""
        with self._conn() as c:
            c.execute("INSERT INTO polls VALUES (%s,%s,%s,%s,%s,%s)", (_now(), source, kind, 0, 0, err[:500]))

    def import_archive(self, archive_dirs=None):
        """Знімки data/raw/alerts_in_ua/* -> сховище (так само, як LiveStore.import_archive)."""
        from src.live import LiveStore
        LiveStore.import_archive(self, archive_dirs)

    def last_ok_poll(self, kind: str = "active") -> pd.Timestamp | None:
        """Час останнього успішного опитування (за ним сервіс рахує вік даних)."""
        with self._conn() as c:
            r = c.execute("SELECT MAX(polled_at) FROM polls WHERE ok=1 AND kind=%s", (kind,)).fetchone()[0]
        return pd.Timestamp(r) if r else None

    def raw_events(self, source: str = "alerts_in_ua") -> pd.DataFrame:
        """Події у форматі EVENT_COLS; відкриті тривоги мають кінець OPEN_END (див. LiveStore)."""
        cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=READ_DAYS)).strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as c:
            rows = c.execute(f"SELECT {', '.join(COLUMNS)} FROM alerts "
                             "WHERE source=%s AND (started_at >= %s OR finished_at IS NULL)",
                             (source, cutoff)).fetchall()
        raw = pd.DataFrame(rows, columns=COLUMNS)
        if raw.empty:
            return normalize_alerts_in_ua(raw)
        raw["_fetched_at"] = raw.last_seen.where(raw.finished_at.notna(), OPEN_END)
        return normalize_alerts_in_ua(raw)
