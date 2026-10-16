"""Telegram-бот прогнозу: long polling, без вебхука й сервера.

  python scripts/run_bot.py

Токен — лише TELEGRAM_BOT_TOKEN у .env / середовищі. Дані — SQLite воркера
(scripts/live_worker.py); якщо воркер не працює, бот прямо каже, що дані застарілі.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bot import Bot, ForecastCache, build_application, configure_logging, read_token  # noqa: E402
from src.service import live_forecaster  # noqa: E402

if __name__ == "__main__":
    token = read_token()
    configure_logging(token)
    app = build_application(token, Bot(ForecastCache(live_forecaster)))
    app.run_polling(allowed_updates=["message", "callback_query"], drop_pending_updates=True)
