"""GET /api/forecast на Vercel: прогноз усіх районів для сайту (JSON).

Дані — Postgres (DATABASE_URL), який наповнює воркер; моделі — models/*.joblib у бандлі функції.
Відповідь кешується на краю CDN на 60 с (s-maxage), тож база й моделі не навантажуються кожним відвідувачем.
Помилка джерела даних -> 503 без подробиць (тип винятку лише в лозі).
"""
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bot import ForecastCache  # noqa: E402
from src.service import live_forecaster  # noqa: E402
from src.site_api import build_payload  # noqa: E402

log = logging.getLogger("forecast")
CACHE = ForecastCache(live_forecaster)


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict, cache: str = "no-store"):
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        try:
            payload = build_payload(CACHE)
        except Exception as e:
            log.warning("forecast failed: %s", type(e).__name__)
            return self._send(503, {"error": "forecast_unavailable"})
        self._send(200, payload, "public, s-maxage=60, stale-while-revalidate=120")
