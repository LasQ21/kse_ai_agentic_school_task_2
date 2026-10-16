"""Локальний перегляд сайту: public/ як статика + /api/forecast з живого сховища воркера (SQLite або Postgres).

python scripts/dev_site.py [порт]   (за замовчуванням 5173). Слухає лише 127.0.0.1.
"""
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bot import ForecastCache  # noqa: E402
from src.service import live_forecaster  # noqa: E402
from src.site_api import build_payload  # noqa: E402

CACHE = ForecastCache(live_forecaster)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] != "/api/forecast":
            return super().do_GET()
        try:
            body, code = build_payload(CACHE), 200
        except Exception as e:
            body, code = {"error": "forecast_unavailable", "type": type(e).__name__}, 503
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5173
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT / "public")))
    print(f"http://127.0.0.1:{port}")
    srv.serve_forever()
