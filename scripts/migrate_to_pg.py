"""Одноразове перенесення data/live/alerts.sqlite у Postgres (Neon): історія для 31-добових ознак.

  python scripts/migrate_to_pg.py          # DATABASE_URL із .env / середовища

Повторний запуск безпечний (upsert за (source, id)); журнал опитувань переноситься лише для
останнього успішного "active", щоб не роздувати таблицю.
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.live import LIVE_DB, open_store  # noqa: E402
from src.live_pg import COLUMNS, PgLiveStore  # noqa: E402

BATCH = 1000

if __name__ == "__main__":
    store = open_store()
    if not isinstance(store, PgLiveStore):
        raise SystemExit("DATABASE_URL не задано (.env або змінна середовища)")
    src = sqlite3.connect(LIVE_DB)
    rows = src.execute(f"SELECT {', '.join(COLUMNS)} FROM alerts").fetchall()
    sql = (f"INSERT INTO alerts ({', '.join(COLUMNS)}) VALUES ({', '.join(['%s'] * len(COLUMNS))}) "
           "ON CONFLICT (source, id) DO UPDATE SET "
           + ", ".join(f"{c}=EXCLUDED.{c}" for c in COLUMNS[2:]))
    with store._conn() as c:
        for i in range(0, len(rows), BATCH):
            c.cursor().executemany(sql, rows[i:i + BATCH])
        last = src.execute("SELECT MAX(polled_at) FROM polls WHERE ok=1 AND kind='active'").fetchone()[0]
        if last:
            c.execute("INSERT INTO polls VALUES (%s,'alerts_in_ua','active',1,0,NULL)", (last,))
    print(f"перенесено {len(rows)} тривог; останнє успішне опитування: {last}")
