"""Воркер живих даних: опитування активних тривог + регулярний знімок історії (окремий процес).

  python scripts/live_worker.py            # працює, доки не зупинять (SIGTERM / Ctrl+C)
  python scripts/live_worker.py --health   # exit 0, якщо останнє успішне опитування свіже

Налаштування (змінні середовища / .env):
  POLL_INTERVAL_S=60     опитування /alerts/active.json
  ARCHIVE_EVERY_H=24     знімок місячної історії (~14 хв через ліміт 2 запити/хв) + імпорт у SQLite;
                         історія в API лише за 30 діб, тож без цього в архіві з'явиться діра
  HEALTH_MAX_AGE_S=300   поріг для --health
Знімок історії йде у фоновому потоці, опитування при цьому не зупиняється.
Стан: data/live/worker_status.json.
"""
import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import RAW_DIR, archive_alerts_in_ua_history  # noqa: E402
from src.live import LIVE_DB, LiveStore, make_source, poll_once  # noqa: E402

STATUS = LIVE_DB.parent / "worker_status.json"
log = logging.getLogger("live_worker")


def env(name, default):
    return type(default)(os.environ.get(name, default))


def last_snapshot_time():
    d = RAW_DIR / "alerts_in_ua"
    snaps = sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []
    return pd.Timestamp(datetime.strptime(snaps[-1], "%Y%m%dT%H%M%SZ"), tz="UTC") if snaps else None


class Worker:
    def __init__(self):
        self.store, self.source = LiveStore(), make_source()
        self.poll_s, self.archive_every = env("POLL_INTERVAL_S", 60), pd.Timedelta(hours=env("ARCHIVE_EVERY_H", 24.0))
        self.stop = threading.Event()
        self.archive_thread = None
        self.status = {"started_at": datetime.now(timezone.utc).isoformat(), "source": self.source.name,
                       "last_poll": None, "last_archive": str(last_snapshot_time()), "archive_running": False,
                       "errors": []}

    def save_status(self):
        # знімок міг зробити й інший процес (run_pipeline / archive_alerts) — беремо з диска
        self.status["last_archive"] = str(last_snapshot_time())
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(self.status, ensure_ascii=False, indent=1, default=str))

    def _archive(self):
        self.status["archive_running"] = True
        try:
            snap = archive_alerts_in_ua_history()
            self.store.import_archive([snap])
            self.status["last_archive"] = snap.name
            log.info("archive snapshot %s imported", snap.name)
        except Exception as e:  # мережа / ліміт — спробуємо в наступному циклі
            self.status["errors"] = (self.status["errors"] + [f"archive: {e!r}"])[-20:]
            log.exception("archive failed")
        finally:
            self.status["archive_running"] = False

    def maybe_archive(self):
        last = last_snapshot_time()
        due = last is None or pd.Timestamp.now(tz="UTC") - last >= self.archive_every
        if due and not (self.archive_thread and self.archive_thread.is_alive()):
            self.archive_thread = threading.Thread(target=self._archive, daemon=True)
            self.archive_thread.start()

    def run(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self.stop.set())
        log.info("worker started: poll every %ss, archive every %s", self.poll_s, self.archive_every)
        while not self.stop.is_set():
            r = poll_once(self.store, self.source)
            if r["ok"]:
                self.status["last_poll"] = r["polled_at"]
            else:
                self.status["errors"] = (self.status["errors"] + [f"poll: {r['error']}"])[-20:]
            self.maybe_archive()
            self.save_status()
            self.stop.wait(self.poll_s)
        log.info("worker stopped")


def health() -> int:
    last = LiveStore().last_ok_poll()
    max_age = env("HEALTH_MAX_AGE_S", 300)
    age = None if last is None else (pd.Timestamp.now(tz="UTC") - last).total_seconds()
    ok = age is not None and age <= max_age
    print(json.dumps({"ok": ok, "last_poll": str(last), "age_s": age}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--health", action="store_true")
    a = ap.parse_args()
    sys.exit(health()) if a.health else Worker().run()
