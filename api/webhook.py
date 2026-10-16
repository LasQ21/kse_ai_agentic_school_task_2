"""Webhook Telegram-бота на Vercel: POST /api/webhook -> python-telegram-bot -> src.bot.Bot.

Дані: Postgres (DATABASE_URL), який наповнює воркер (scripts/live_worker.py). Моделі — models/*.joblib
у бандлі функції. Змінні середовища: TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET, DATABASE_URL.
Без TELEGRAM_WEBHOOK_SECRET функція відмовляє (503): інакше будь-хто міг би слати їй підроблені оновлення.
"""
import asyncio
import hmac
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bot import Bot, ForecastCache, build_application, configure_logging  # noqa: E402
from src.service import live_forecaster  # noqa: E402

log = logging.getLogger("webhook")
MAX_BODY = 1_000_000
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
configure_logging(TOKEN or None)
# Один Bot на екземпляр функції: кеш прогнозу й ліміт запитів живуть, доки екземпляр теплий.
BOT = Bot(ForecastCache(live_forecaster))


async def process(data: dict):
    from telegram import Update
    # Application створюється на кожен запит: asyncio.run дає новий цикл подій, а клієнт
    # python-telegram-bot прив'язаний до того, у якому створений.
    app = build_application(TOKEN, BOT)
    async with app:
        await app.process_update(Update.de_json(data, app.bot))


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: str = ""):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        self._send(200, "ok")

    def do_POST(self):
        # Тіло читаємо до перевірок: відповідь без прочитаного тіла рве з'єднання на боці клієнта.
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            n = 0
        if not 0 < n <= MAX_BODY:
            return self._send(400, "bad length")
        body = self.rfile.read(n)
        if not TOKEN or not SECRET:
            return self._send(503, "not configured")
        given = self.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not hmac.compare_digest(given.encode(), SECRET.encode()):
            return self._send(403, "forbidden")
        try:
            data = json.loads(body)
        except ValueError:
            return self._send(400, "bad json")
        try:
            asyncio.run(process(data))
        except Exception as e:  # 200, щоб Telegram не повторював оновлення; у лог — лише тип винятку
            log.warning("update failed: %s", type(e).__name__)
        self._send(200, "ok")
