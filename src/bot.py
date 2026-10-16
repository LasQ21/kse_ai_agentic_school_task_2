"""Telegram-бот прогнозу (python-telegram-bot, long polling). Рахує лише через src.service.

Команди: /start, /region, /top, /status, /about, /help; також пошук району текстом.
Безпека:
- токен лише з середовища (TELEGRAM_BOT_TOKEN), у логах маскується;
- логери httpx / httpcore / urllib3 / telegram — WARNING (URL запиту містить токен);
- тексти повідомлень і id користувачів не логуються;
- обмеження частоти на користувача й довжини вводу;
- помилки в чаті — без трейсбека.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.data import EXCLUDED_OBLASTS, KYIV_CITY, OBLAST_AS_UNIT
from src.service import Forecast, HORIZONS_BOT

log = logging.getLogger("bot")

DISCLAIMER = "Це статистичний прогноз, а не офіційне оповіщення. Орієнтуйтеся на офіційні сигнали тривоги."
EXCLUDED_REASON = {
    "Донецька область": "райони майже весь час під тривогою й не відрізняються між собою",
    "Луганська область": "окупована територія",
    "Автономна Республіка Крим": "окупована територія",
    "м. Севастополь": "окупована територія",
}
MAX_INPUT = 64            # символів у текстовому запиті
RATE_LIMIT = (10, 60.0)   # не більше 10 запитів за 60 с на користувача
CACHE_S = 60              # прогноз для всіх районів перераховується не частіше
TOP_N = 10
TOP_H = 3                 # /top сортує за P(нова тривога за 3 год)
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "telegram", "telegram.ext", "apscheduler")


# ---------------------------------------------------------------- логування
class RedactFilter(logging.Filter):
    """Маскує токен бота (і будь-що схоже на токен) у тексті запису логу."""

    TOKEN_RE = re.compile(r"\d{6,12}:[A-Za-z0-9_-]{30,}")

    def __init__(self, token: str | None = None):
        super().__init__()
        self.token = token

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        red = self.TOKEN_RE.sub("<token>", msg.replace(self.token, "<token>") if self.token else msg)
        if red != msg:
            record.msg, record.args = red, ()
        return True


def configure_logging(token: str | None = None, level=logging.INFO):
    """Свій формат, шумні бібліотеки — WARNING, фільтр токена на кожному обробнику."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    f = RedactFilter(token)
    for h in logging.getLogger().handlers:
        h.addFilter(f)
    return f


def read_token() -> str:
    """TELEGRAM_BOT_TOKEN із середовища (або .env). Значення ніде не друкується."""
    from dotenv import load_dotenv
    from src.data import ROOT
    load_dotenv(ROOT / ".env")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задано (.env або змінна середовища)")
    return token


# ---------------------------------------------------------------- форматування
def fmt_p(p: float) -> str:
    """Округлення: < 1% → «<1%»; до 10% — до 1 п.п.; від 10% — до 5 п.п.; > 99% → «>99%»."""
    if p is None or not math.isfinite(p):
        return "—"
    pct = 100 * p
    if pct < 1:
        return "<1%"
    if pct > 99:
        return ">99%"
    if pct < 10:
        return f"{round(pct):.0f}%"
    return f"{min(95, 5 * round(pct / 5)):.0f}%"


def fmt_duration(hours: float) -> str:
    """1.33 -> «1 год 20 хв»; 0.2 -> «12 хв»."""
    m = max(0, int(round(hours * 60)))
    h, m = divmod(m, 60)
    return f"{h} год {m} хв" if h else f"{m} хв"


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "невідомо (опитувань ще не було)"
    if seconds < 90:
        return f"{seconds:.0f} с тому"
    return f"{fmt_duration(seconds / 3600)} тому"


UK_ALPHABET = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UK_RANK = {c: i + 1 for i, c in enumerate(UK_ALPHABET)}


def uk_key(label: str) -> tuple:
    """Ключ українського алфавітного порядку (а б в г ґ д е є ж з и і ї й …).

    Сортування за кодами Юнікоду ставить і/ї/є/ґ після «я», тому «Івано-Франківська» опиняється
    першою, а «Білоцерківський» — після «Бучанський». Префікс «м. » ігнорується («м. Київ» — на К);
    не-літери (дефіс, пробіл, апостроф) йдуть перед літерами.
    """
    t = label.lower().replace("’", "'")
    if t.startswith("м. "):
        t = t[3:]
    return tuple(UK_RANK.get(c, 0) for c in t) + (len(t),), t


def region_title(unit: str) -> str:
    o, r = unit.split("|", 1)
    return o if r == "*" else f"{o}, {r}"


def mode_text(f: Forecast) -> str:
    if f.mode == "climate":
        return "базовий кліматичний рівень (частота тривог у цьому районі в цю годину тижня)"
    return f"модель {f.mode.upper()}" if f.mode == "gbm" else f"моделі: {f.mode}"


def format_forecast(f: Forecast, tz: str = "Europe/Kyiv") -> str:
    lines = [region_title(f.region), ""]
    if f.stale:
        when = "оновлень ще не було" if f.data_last_update is None else \
            f"останнє оновлення {fmt_age(f.data_age_s)}"
        lines += [f"⚠️ Дані застаріли: {when}. Прогноз — станом на "
                  f"{pd.Timestamp(f.as_of).tz_convert(tz):%d.%m %H:%M}, а не на зараз.", ""]
    if f.state == "B":
        lines.append(f"Тривога йде {fmt_duration(f.elapsed_h or 0)}. Нова тривога можлива лише після відбою.")
    else:
        lines.append("Тривоги зараз немає.")
    lines += ["", "Ймовірність, що почнеться нова тривога:"]
    lines += [f"• за {h} год: {fmt_p(f.probabilities.get(h))}" for h in HORIZONS_BOT if h in f.probabilities]
    if f.counts:
        lines += ["", "Скільки нових тривог:"]
        for h in (6, 24):
            c = f.counts
            lines.append(f"• за {h} год: жодної {fmt_p(c['p0'][h])}, одна {fmt_p(c['p1'][h])}, "
                         f"дві й більше {fmt_p(c['p2plus'][h])}")
    lines += ["", f"Режим: {mode_text(f)}.",
              f"Дані alerts.in.ua: {fmt_age(f.data_age_s) if f.data_last_update is not None else 'невідомо'}.",
              "", DISCLAIMER]
    return "\n".join(lines)


def format_top(fs: list[Forecast], status: dict) -> str:
    fs = sorted(fs, key=lambda f: f.probabilities.get(TOP_H, 0), reverse=True)[:TOP_N]
    lines = [f"Найвищий ризик нової тривоги за {TOP_H} год:", ""]
    for i, f in enumerate(fs, 1):
        now = " (тривога йде)" if f.state == "B" else ""
        clim = " [кліматичний рівень]" if f.mode == "climate" else ""
        p = " / ".join(fmt_p(f.probabilities.get(h)) for h in HORIZONS_BOT)
        lines.append(f"{i}. {region_title(f.region)}{now}{clim}: {p}")
    lines += ["", "Ймовірності за 1 / 3 / 6 / 24 год."]
    if status.get("stale"):
        lines.append(f"⚠️ Дані застаріли: останнє оновлення {fmt_age(status.get('data_age_s'))}.")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


ABOUT = (
    "Як це працює\n\n"
    "Модель навчена на офіційному каналі «Повітряна тривога» (листопад 2025 — серпень 2026) "
    "на рівні районів. Живі дані беруться з alerts.in.ua, де визначення тривоги інше: крім "
    "повітряної тривоги там є «попередня загроза», і бот враховує обидва рівні. Тому ймовірності — "
    "орієнтир, а не точна оцінка.\n\n"
    "Що прогнозується: чи почнеться нова тривога в районі за 1, 3, 6 і 24 год, і скільки нових "
    "тривог буде. Якщо тривога вже йде, нова можлива лише після відбою.\n\n"
    "Обмеження:\n"
    "• західні області (Волинська, Рівненська, Львівська, Тернопільська, Івано-Франківська, "
    "Закарпатська, Чернівецька) — базовий кліматичний рівень: модель там не перевершила частоти тривог;\n"
    "• Донецька, Луганська області, АР Крим і Севастополь — без прогнозу;\n"
    "• період перевірки (серпень 2026) переглядали кілька разів під час розробки (старі цілі двічі, "
    "нові один раз), тож чистої незалежної перевірки немає;\n"
    "• у живих даних кінці тривог наближені (момент опитування), старти — точні.\n\n" + DISCLAIMER)
HELP = ("/region — вибрати область і район\n/top — райони з найвищим ризиком\n"
        "/status — вік даних і модель\n/about — методологія й обмеження\n"
        "Або напишіть назву району чи області (до 64 символів).")


# ---------------------------------------------------------------- дані
@dataclass
class ForecastCache:
    """Прогноз для всіх районів, перерахований не частіше CACHE_S (forecaster.forecast_many)."""
    forecaster_fn: Callable[[], object]
    ttl: float = CACHE_S
    _at: float = -1e9
    _value: tuple | None = None
    _fc: object | None = None

    def get(self) -> tuple[dict[str, Forecast], dict]:
        if self._value is None or time.monotonic() - self._at > self.ttl:
            if self._fc is None:
                self._fc = self.forecaster_fn()
            fs, status = self._fc.forecast_many(pd.Timestamp.now(tz="UTC"))
            self._value, self._at = ({f.region: f for f in fs}, status), time.monotonic()
        return self._value

    def meta(self) -> dict:
        if self._fc is None:
            self._fc = self.forecaster_fn()
        st = getattr(self._fc, "state", None) or {}
        return {"old": self._fc.meta, "bot": st.get("meta", {}), "family_any24": st.get("family_any24"),
                "family_q": st.get("family_q")}


@dataclass
class RateLimiter:
    limit: int = RATE_LIMIT[0]
    window: float = RATE_LIMIT[1]
    hits: dict = field(default_factory=lambda: defaultdict(deque))

    def allow(self, user_id: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        q = self.hits[user_id]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


# ---------------------------------------------------------------- хендлери
class Bot:
    """Хендлери без прив'язки до мережі: працюють з будь-яким об'єктом із полями Update."""

    def __init__(self, cache: ForecastCache, limiter: RateLimiter | None = None):
        self.cache, self.limiter = cache, limiter or RateLimiter()

    async def _forecasts(self):
        return await asyncio.to_thread(self.cache.get)

    def _allowed(self, update) -> bool:
        u = getattr(update, "effective_user", None)
        return self.limiter.allow(u.id if u else 0)

    async def _reply(self, update, text, markup=None):
        msg = update.effective_message
        await msg.reply_text(text, reply_markup=markup)

    async def _limited(self, update) -> bool:
        if self._allowed(update):
            return False
        await self._reply(update, "Забагато запитів. Спробуйте за хвилину.")
        return True

    async def start(self, update, context):
        if await self._limited(update):
            return
        await self._reply(update, "Прогноз ризику повітряної тривоги по районах України.\n\n"
                                  f"{DISCLAIMER}\n\n{HELP}")

    async def help(self, update, context):
        if await self._limited(update):
            return
        await self._reply(update, HELP)

    async def about(self, update, context):
        if await self._limited(update):
            return
        await self._reply(update, ABOUT)

    async def status(self, update, context):
        if await self._limited(update):
            return
        fs, st = await self._forecasts()
        m = self.cache.meta()
        b = m["bot"]
        lines = ["Стан даних: " + ("⚠️ застарілі" if st["stale"] else "свіжі"),
                 f"Останнє оновлення alerts.in.ua: {fmt_age(st['data_age_s'])}",
                 f"Районів з прогнозом: {len(fs)}; зараз під тривогою: {sum(f.state == 'B' for f in fs.values())}",
                 "", "Модель 1/3/6 год: GBM, навчена на даних до "
                 f"{str(m['old'].get('train_range', ['', '?'])[1])[:10]} (захід — клімат)"]
        if b:
            lines.append(f"Модель 24 год і лічильник: {m['family_any24']} / {m['family_q']}, "
                         f"версія коду {b.get('code_version', '?')}")
        await self._reply(update, "\n".join(lines))

    async def top(self, update, context):
        if await self._limited(update):
            return
        fs, st = await self._forecasts()
        await self._reply(update, format_top(list(fs.values()), st))

    def _oblasts(self, fs: dict) -> list[str]:
        return sorted({u.split("|")[0] for u in fs} | set(EXCLUDED_OBLASTS), key=uk_key)

    def oblast_keyboard(self, fs: dict):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        obl = self._oblasts(fs)
        rows = [[InlineKeyboardButton(o.replace(" область", ""), callback_data=f"o:{i}")
                 for i, o in enumerate(obl)][j:j + 2] for j in range(0, len(obl), 2)]
        return InlineKeyboardMarkup(rows)

    def raion_keyboard(self, fs: dict, oblast: str):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        units = sorted((u for u in fs if u.startswith(oblast + "|")), key=lambda u: uk_key(u.split("|")[1]))
        idx = {u: i for i, u in enumerate(sorted(fs))}
        btn = [InlineKeyboardButton(u.split("|")[1], callback_data=f"u:{idx[u]}") for u in units]
        return InlineKeyboardMarkup([btn[j:j + 2] for j in range(0, len(btn), 2)])

    async def region(self, update, context):
        if await self._limited(update):
            return
        fs, _ = await self._forecasts()
        await self._reply(update, "Оберіть область:", self.oblast_keyboard(fs))

    async def callback(self, update, context):
        q = update.callback_query
        await q.answer()
        if not self._allowed(update):
            await q.edit_message_text("Забагато запитів. Спробуйте за хвилину.")
            return
        fs, _ = await self._forecasts()
        data = (q.data or "")[:16]
        kind, _, num = data.partition(":")
        if not num.isdigit():
            return
        i = int(num)
        if kind == "o":
            obl = self._oblasts(fs)
            if i >= len(obl):
                return
            o = obl[i]
            if o in EXCLUDED_OBLASTS:
                await q.edit_message_text(f"{o}: без прогнозу — {EXCLUDED_REASON.get(o, 'виключено')}.")
            elif o in OBLAST_AS_UNIT | {KYIV_CITY}:
                await q.edit_message_text(format_forecast(fs[f"{o}|*"]))
            else:
                await q.edit_message_text(f"{o}: оберіть район:", reply_markup=self.raion_keyboard(fs, o))
        elif kind == "u":
            units = sorted(fs)
            if i < len(units):
                await q.edit_message_text(format_forecast(fs[units[i]]))

    async def text(self, update, context):
        if await self._limited(update):
            return
        raw = (update.effective_message.text or "").strip()
        if len(raw) > MAX_INPUT:
            await self._reply(update, f"Запит задовгий (більше {MAX_INPUT} символів). Напишіть назву району.")
            return
        query = raw.lower().replace("’", "'")
        if len(query) < 3:
            await self._reply(update, "Напишіть щонайменше 3 літери назви району чи області, або /region.")
            return
        for o in EXCLUDED_OBLASTS:
            if query in o.lower():
                await self._reply(update, f"{o}: без прогнозу — {EXCLUDED_REASON.get(o, 'виключено')}.")
                return
        fs, _ = await self._forecasts()
        hits = sorted((u for u in fs if query in u.lower()), key=lambda u: uk_key(region_title(u)))
        if len(hits) == 1:
            await self._reply(update, format_forecast(fs[hits[0]]))
        elif not hits:
            await self._reply(update, "Не знайшов такого району. Спробуйте /region.")
        else:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            idx = {u: i for i, u in enumerate(sorted(fs))}
            btn = [[InlineKeyboardButton(region_title(u), callback_data=f"u:{idx[u]}")] for u in hits[:12]]
            await self._reply(update, f"Знайдено {len(hits)}; оберіть:", InlineKeyboardMarkup(btn))

    async def error(self, update, context):
        """Помилка: у лог — лише тип винятку, у чат — коротке повідомлення без трейсбека."""
        err = getattr(context, "error", None)
        log.warning("handler error: %s", type(err).__name__)
        msg = getattr(update, "effective_message", None) if update is not None else None
        if msg is not None:
            await msg.reply_text("Не вдалося отримати прогноз. Спробуйте пізніше.")


def build_application(token: str, bot: Bot):
    """Application з усіма хендлерами (мережі не торкається, доки не запущено polling)."""
    from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, MessageHandler,
                              filters)
    app = Application.builder().token(token).build()
    for name in ("start", "help", "about", "status", "top", "region"):
        app.add_handler(CommandHandler(name, getattr(bot, name)))
    app.add_handler(CallbackQueryHandler(bot.callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.text))
    app.add_error_handler(bot.error)
    return app
