"""Реєструє (або знімає) webhook бота в Telegram. Токен і секрет — лише з .env / середовища.

  python scripts/set_webhook.py https://<проєкт>.vercel.app/api/webhook
  python scripts/set_webhook.py --delete        # повернутися до long polling (scripts/run_bot.py)
  python scripts/set_webhook.py --info

Потрібні TELEGRAM_BOT_TOKEN і (для реєстрації) TELEGRAM_WEBHOOK_SECRET — той самий, що у змінних
середовища Vercel. Токен і секрет не друкуються.
"""
import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ["message", "callback_query"]


def call(token: str, method: str, **params) -> dict:
    r = requests.post(f"https://api.telegram.org/bot{token}/{method}", json=params, timeout=30)
    return r.json()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--info", action="store_true")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not token:
        sys.exit("TELEGRAM_BOT_TOKEN не задано (.env або змінна середовища)")
    if a.delete:
        res = call(token, "deleteWebhook", drop_pending_updates=True)
    elif a.info:
        res = call(token, "getWebhookInfo")
        res["result"] = {k: v for k, v in res.get("result", {}).items() if k != "ip_address"}
    else:
        if not a.url or not a.url.startswith("https://"):
            sys.exit("потрібна https-адреса webhook, напр. https://<проєкт>.vercel.app/api/webhook")
        if len(secret) < 16:
            sys.exit("TELEGRAM_WEBHOOK_SECRET не задано або закороткий (потрібно >= 16 символів A-Za-z0-9_-)")
        res = call(token, "setWebhook", url=a.url, secret_token=secret, allowed_updates=ALLOWED,
                   drop_pending_updates=True, max_connections=5)
    print(res)
