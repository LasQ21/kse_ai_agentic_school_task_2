"""Бот без мережі: хендлери з підробленим Update, фальшивий (синтетичний) токен, форматування,
обмеження частоти й довжини вводу, токен не потрапляє в лог, помилки без трейсбека."""
import asyncio
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from src.bot import (DISCLAIMER, MAX_INPUT, Bot, ForecastCache, RateLimiter, RedactFilter, build_application,
                     configure_logging, fmt_duration, fmt_p, format_forecast)
from src.service import Forecast

FAKE_TOKEN = "123456789:" + "A" * 35   # синтетичний, не похідний від жодного реального
NOW = pd.Timestamp("2026-10-16T12:00:00Z")


def fc(region, state="A", elapsed=None, mode="gbm", stale=False, age=30.0, p=(0.05, 0.2, 0.4, 0.8)):
    probs = dict(zip((1, 3, 6, 24), p))
    counts = {"p0": {h: 1 - v for h, v in probs.items()}, "p1": {h: v / 2 for h, v in probs.items()},
              "p2plus": {h: v / 2 for h, v in probs.items()}}
    return Forecast(region=region, requested_for=NOW, as_of=NOW - pd.Timedelta(hours=3) if stale else NOW,
                    probabilities=probs, active_now=state == "B", model=mode, data_last_update=NOW,
                    data_age_s=3 * 3600 if stale else age, stale=stale, state=state, elapsed_h=elapsed,
                    mode=mode, counts=counts)


FORECASTS = {f.region: f for f in [
    fc("Харківська область|Харківський район", "B", 1 + 20 / 60, p=(0.02, 0.3, 0.6, 0.9)),
    fc("Харківська область|Ізюмський район"),
    fc("Львівська область|Львівський район", mode="climate", p=(0.01, 0.03, 0.05, 0.2)),
    fc("м. Київ|*", p=(0.1, 0.25, 0.4, 0.7)),
]}


class FakeForecaster:
    meta = {"train_range": ["2025-11-17", "2026-06-13 23:00"]}
    state = {"meta": {"code_version": "abc1234"}, "family_any24": "gbm", "family_q": "gbm"}

    def __init__(self, stale=False, fail=False):
        self.stale, self.fail = stale, fail

    def forecast_many(self, now):
        if self.fail:
            raise RuntimeError("boom /Users/x/secret " + FAKE_TOKEN)
        return list(FORECASTS.values()), {"stale": self.stale, "data_age_s": 30.0, "as_of": now,
                                          "data_last_update": now}


class Msg:
    def __init__(self, text=""):
        self.text, self.replies = text, []

    async def reply_text(self, text, reply_markup=None):
        self.replies.append((text, reply_markup))


class Query:
    def __init__(self, data):
        self.data, self.edits = data, []

    async def answer(self):
        pass

    async def edit_message_text(self, text, reply_markup=None):
        self.edits.append((text, reply_markup))


def update(text="", user=1, data=None):
    m = Msg(text)
    return SimpleNamespace(effective_user=SimpleNamespace(id=user), effective_message=m,
                           callback_query=Query(data) if data else None), m


def make_bot(**kw):
    return Bot(ForecastCache(lambda: FakeForecaster(**kw)))


def run(coro):
    return asyncio.run(coro)


def test_rounding_and_duration():
    assert [fmt_p(x) for x in (0.004, 0.034, 0.096, 0.12, 0.33, 0.996)] == ["<1%", "3%", "10%", "10%", "35%", ">99%"]
    assert fmt_duration(1 + 20 / 60) == "1 год 20 хв" and fmt_duration(0.2) == "12 хв"


def test_format_states_mode_disclaimer():
    b = format_forecast(FORECASTS["Харківська область|Харківський район"])
    assert "Тривога йде 1 год 20 хв" in b and "лише після відбою" in b
    assert "за 24 год: 90%" in b and "дві й більше" in b and b.endswith(DISCLAIMER)
    a = format_forecast(FORECASTS["Харківська область|Ізюмський район"])
    assert "Тривоги зараз немає." in a
    w = format_forecast(FORECASTS["Львівська область|Львівський район"])
    assert "базовий кліматичний рівень" in w


def test_stale_is_said_plainly():
    s = format_forecast(fc("м. Київ|*", stale=True))
    assert "Дані застаріли" in s and "не на зараз" in s


def test_handlers_start_top_status_about():
    bot = make_bot()
    for h in (bot.start, bot.top, bot.status, bot.about, bot.help):
        u, m = update()
        run(h(u, None))
        assert m.replies and m.replies[0][0]
    u, m = update()
    run(bot.top(u, None))
    assert m.replies[0][0].splitlines()[2].startswith("1. Харківська область, Харківський район")


def test_region_flow_and_excluded():
    bot = make_bot()
    u, m = update()
    run(bot.region(u, None))
    kb = m.replies[0][1].inline_keyboard
    names = [b.text for row in kb for b in row]
    datas = {b.text: b.callback_data for row in kb for b in row}
    assert "Харківська" in names and "Донецька" in names
    u, _ = update(data=datas["Харківська"])
    run(bot.callback(u, None))
    text, markup = u.callback_query.edits[0]
    raion = [b for row in markup.inline_keyboard for b in row if b.text == "Харківський район"][0]
    u2, _ = update(data=raion.callback_data)
    run(bot.callback(u2, None))
    assert "Тривога йде" in u2.callback_query.edits[0][0]
    u3, _ = update(data=datas["Донецька"])
    run(bot.callback(u3, None))
    assert "без прогнозу" in u3.callback_query.edits[0][0]
    u4, _ = update(data="u:9999")          # вигаданий індекс не ламає бота
    run(bot.callback(u4, None))


def test_text_search_input_limit_and_unknown():
    bot = make_bot()
    u, m = update("ізюм")
    run(bot.text(u, None))
    assert "Ізюмський" in m.replies[0][0]
    u, m = update("х" * (MAX_INPUT + 1))
    run(bot.text(u, None))
    assert "задовгий" in m.replies[0][0]
    u, m = update("Донецька")
    run(bot.text(u, None))
    assert "без прогнозу" in m.replies[0][0]


def test_rate_limit_per_user():
    r = RateLimiter(limit=3, window=60)
    assert all(r.allow(1, now=t) for t in (0, 1, 2)) and not r.allow(1, now=3) and r.allow(2, now=3)
    assert r.allow(1, now=61)
    bot = Bot(ForecastCache(lambda: FakeForecaster()), RateLimiter(limit=1, window=60))
    for i in range(2):
        u, m = update(user=7)
        run(bot.help(u, None))
    assert "Забагато запитів" in m.replies[0][0]


def test_error_handler_no_traceback_no_token(caplog):
    bot = make_bot(fail=True)
    u, m = update()
    with pytest.raises(RuntimeError) as e:
        run(bot.top(u, None))
    caplog.set_level(logging.WARNING)
    run(bot.error(u, SimpleNamespace(error=e.value)))
    reply = m.replies[0][0]
    assert "Traceback" not in reply and FAKE_TOKEN not in reply and "/Users/" not in reply
    assert FAKE_TOKEN not in caplog.text and "boom" not in caplog.text


def test_token_never_in_logs(caplog):
    caplog.set_level(logging.DEBUG)
    f = configure_logging(FAKE_TOKEN)
    caplog.handler.addFilter(f)
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("telegram").level == logging.WARNING
    logging.getLogger("httpx").info("POST https://api.telegram.org/bot%s/getUpdates", FAKE_TOKEN)
    logging.getLogger("bot").warning("url https://api.telegram.org/bot%s/sendMessage", FAKE_TOKEN)
    logging.getLogger("other").error("other token 987654321:" + "B" * 35)
    assert FAKE_TOKEN not in caplog.text and "B" * 35 not in caplog.text and "<token>" in caplog.text
    r = logging.LogRecord("x", logging.INFO, "", 0, "t=%s", (FAKE_TOKEN,), None)
    RedactFilter(FAKE_TOKEN).filter(r)
    assert FAKE_TOKEN not in r.getMessage()


def test_build_application_offline():
    app = build_application(FAKE_TOKEN, make_bot())
    names = {c for h in app.handlers[0] for c in getattr(h, "commands", ())}
    assert {"start", "help", "about", "status", "top", "region"} <= names
