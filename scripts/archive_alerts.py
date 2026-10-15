"""Знімок місячної історії alerts.in.ua (запускати щодня, поки триває проєкт)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import archive_alerts_in_ua_history  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(archive_alerts_in_ua_history())
