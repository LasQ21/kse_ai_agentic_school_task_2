"""Єдина точка входу Vercel (pyproject.toml, [tool.vercel] entrypoint): збірка Python вимагає один застосунок.

GET /api/forecast -> JSON прогнозу для сайту (api/forecast.py); усе інше — як у api/webhook.py
(POST — webhook Telegram із секретом, GET — «ok»). Статика public/ віддається до цієї функції.
"""
from api.forecast import handler as ForecastHandler
from api.webhook import handler as WebhookHandler


class handler(WebhookHandler):
    _send_json = ForecastHandler._send_json   # do_GET прогнозу викликається з цього класу

    def do_GET(self):
        if self.path.split("?")[0] == "/api/forecast":
            return ForecastHandler.do_GET(self)
        return super().do_GET()
