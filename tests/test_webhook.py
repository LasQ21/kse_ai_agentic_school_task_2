"""Webhook на Vercel без мережі: секрет, розмір тіла, обробка оновлення через python-telegram-bot.

Токен і секрет синтетичні. Мережа Telegram підмінена: Bot._post
записує виклики API (getMe, sendMessage) замість відправки."""
import importlib
import json
import threading
from http.client import HTTPConnection
from http.server import HTTPServer

import pytest

FAKE_TOKEN = "123456789:" + "B" * 35
SECRET = "s" * 24
UPDATE = {"update_id": 1, "message": {
    "message_id": 5, "date": 1760000000, "chat": {"id": 7, "type": "private"},
    "from": {"id": 7, "is_bot": False, "first_name": "t"}, "text": "/help",
    "entities": [{"type": "bot_command", "offset": 0, "length": 5}]}}


def load_module(monkeypatch, token=FAKE_TOKEN, secret=SECRET):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", secret)
    import api.webhook as w
    return importlib.reload(w)


def request(module, method="POST", headers=None, body=b""):
    srv = HTTPServer(("127.0.0.1", 0), module.handler)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    c = HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    c.request(method, "/api/webhook", body=body, headers=headers or {})
    r = c.getresponse()
    out = r.status, r.read().decode()
    c.close()
    srv.server_close()
    return out


def post(module, payload=UPDATE, secret=SECRET):
    body = json.dumps(payload).encode()
    h = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
    if secret is not None:
        h["X-Telegram-Bot-Api-Secret-Token"] = secret
    return request(module, headers=h, body=body)


def test_get_is_health(monkeypatch):
    assert request(load_module(monkeypatch), "GET") == (200, "ok")


def test_wrong_or_missing_secret_rejected_and_not_processed(monkeypatch):
    w = load_module(monkeypatch)
    called = []
    monkeypatch.setattr(w, "process", lambda data: called.append(data))
    assert post(w, secret="x" * 24)[0] == 403
    assert post(w, secret=None)[0] == 403
    assert not called


def test_unconfigured_secret_fails_closed(monkeypatch):
    assert post(load_module(monkeypatch, secret=""))[0] == 503


def test_bad_body_rejected(monkeypatch):
    w = load_module(monkeypatch)
    h = {"Content-Length": "3", "X-Telegram-Bot-Api-Secret-Token": SECRET}
    assert request(w, headers=h, body=b"{{{")[0] == 400
    assert request(w, headers={"Content-Length": "0", "X-Telegram-Bot-Api-Secret-Token": SECRET})[0] == 400


def test_help_command_is_answered_via_telegram_api(monkeypatch):
    import telegram

    w = load_module(monkeypatch)
    sent = []

    async def fake_post(self, endpoint, data=None, **kw):
        sent.append((endpoint, data))
        if endpoint == "getMe":
            return {"id": 1, "is_bot": True, "first_name": "b", "username": "b_bot"}
        return {"message_id": 6, "date": 1760000001, "chat": {"id": 7, "type": "private"}}

    monkeypatch.setattr(telegram.Bot, "_post", fake_post)
    assert post(w)[0] == 200
    assert [e for e, _ in sent] == ["getMe", "sendMessage"]
    assert "/region" in sent[1][1]["text"]


def test_processing_error_still_returns_200_without_leaking(monkeypatch, caplog):
    w = load_module(monkeypatch)

    async def boom(data):
        raise RuntimeError(f"token {FAKE_TOKEN} leaked")

    monkeypatch.setattr(w, "process", boom)
    assert post(w)[0] == 200
    assert FAKE_TOKEN not in caplog.text
