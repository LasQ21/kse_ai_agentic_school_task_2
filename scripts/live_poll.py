"""Живі дані в data/live/alerts.sqlite (або в Postgres, якщо задано DATABASE_URL).

  python scripts/live_poll.py --import-archive   # стартова історія з data/raw/alerts_in_ua/*
  python scripts/live_poll.py --once             # одне опитування активних тривог
  python scripts/live_poll.py --interval 90      # опитування раз на 90 с (Ctrl+C — зупинка)
  python scripts/live_poll.py --reconcile        # точні кінці з історії (~14 хв, ліміт 2 запити/хв)

Джерело — ALERT_SOURCE у .env (alerts_in_ua за замовчуванням; ukrainealarm — коли буде ключ).
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live import make_source, open_store, poll_once, reconcile_with_history, run_poller  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--import-archive", action="store_true")
    g.add_argument("--once", action="store_true")
    g.add_argument("--interval", type=int)
    g.add_argument("--reconcile", action="store_true")
    ap.add_argument("--source")
    a = ap.parse_args()
    store = open_store()
    if a.import_archive:
        store.import_archive()
        print("imported; events:", len(store.raw_events()))
    elif a.once:
        print(json.dumps(poll_once(store, make_source(a.source)), ensure_ascii=False))
    elif a.reconcile:
        reconcile_with_history(store, make_source(a.source))
    else:
        run_poller(a.interval, a.source)
