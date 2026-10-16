"""Діри в даних alerts.in.ua: що не покрито ні знімками історії, ні безперервним опитуванням.

Історія (month_ago) покриває для області [запит - 30 діб, запит]. Опитування покриває відрізок
між сусідніми успішними опитуваннями, якщо вони ближче за MAX_POLL_GAP (тривогу, коротшу за цей
проміжок, можна пропустити). Виводить непокриті інтервали довші за MIN_GAP по кожній області.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import OBLAST_UIDS, RAW_DIR  # noqa: E402
from src.live import LIVE_DB  # noqa: E402

MAX_POLL_GAP = pd.Timedelta(minutes=3)
MIN_GAP = pd.Timedelta(minutes=5)


def merge(iv):
    """Об'єднує інтервали, що перетинаються."""
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


if __name__ == "__main__":
    with sqlite3.connect(LIVE_DB) as c:
        polls = pd.to_datetime(pd.read_sql_query(
            "SELECT polled_at FROM polls WHERE ok=1 AND kind='active' ORDER BY polled_at", c).polled_at,
            utc=True, format="ISO8601")
    poll_iv = [[a, b] for a, b in zip(polls[:-1], polls[1:]) if b - a <= MAX_POLL_GAP]
    now = pd.Timestamp.now(tz="UTC")
    snaps = sorted(p for p in (RAW_DIR / "alerts_in_ua").iterdir() if p.is_dir())
    print("знімки історії:", [s.name for s in snaps], "| успішних опитувань:", len(polls),
          "| останнє:", polls.max() if len(polls) else None)
    worst = []
    for uid, name in OBLAST_UIDS.items():
        iv = []
        for s in snaps:
            f = s / f"history_{uid}.json"
            if f.exists():
                p = json.loads(f.read_text())
                t = pd.Timestamp(p["_fetched_at"]) if "_fetched_at" in p else pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
                iv.append([t - pd.Timedelta(days=30), t])
        cov = merge(iv + poll_iv)
        start = cov[0][0]
        gaps = [(a[1], b[0]) for a, b in zip(cov[:-1], cov[1:]) if b[0] - a[1] > MIN_GAP]
        tail = now - cov[-1][1]
        worst.append((name, start, gaps, tail))
    print("покриття починається (найпізніше серед областей):", max(w[1] for w in worst))
    any_gap = False
    for name, start, gaps, tail in worst:
        if gaps:
            any_gap = True
            for a, b in gaps:
                print(f"ДІРА {name}: {a} -> {b} ({(b - a)})")
    print("внутрішніх дір > 5 хв:", "є" if any_gap else "немає")
    print("непокритий хвіст до now (макс. по областях):", max(w[3] for w in worst))
