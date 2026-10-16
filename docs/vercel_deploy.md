# Бот на Vercel (webhook) + Neon Postgres

Vercel виконує короткі функції без диска, тому живе сховище винесено в Postgres:

```
alerts.in.ua --(воркер, у вас/Docker)--> Neon Postgres <--(читає)-- Vercel: /api/webhook <-- Telegram
```

Воркер (`scripts/live_worker.py`) лишається поза Vercel: Cron на тарифі Hobby спрацьовує раз на добу,
а дані мусять оновлюватись щохвилини (інакше за 10 хв прогноз стає «застарілим»).

## Одноразове налаштування

1. **Neon** (neon.tech, безкоштовно): створити проєкт, скопіювати connection string з `-pooler` у хості.
2. **`.env`** (не в git): `DATABASE_URL=...`, `TELEGRAM_WEBHOOK_SECRET=<випадкові 32 символи A-Za-z0-9_->`.
   Секрет можна згенерувати так: `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
3. **Перенести історію** з SQLite: `python scripts/migrate_to_pg.py` (повторний запуск безпечний).
4. **Воркер на Postgres**: з `DATABASE_URL` у `.env` `python scripts/live_worker.py` пише вже в Neon
   (без `DATABASE_URL` — як раніше, в SQLite). Зупиніть старий воркер і бота з long polling.
5. **Vercel** — окремий проєкт на корінь репозиторію (Root Directory порожній; заглушка `site/` — інший проєкт).
   Environment Variables: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`, `DATABASE_URL`.
6. **Деплой з CLI** (не через GitHub): `models/` у `.gitignore`, тож у git їх немає; `.vercelignore`
   замінює `.gitignore` для CLI й вмикає їх у бандл.
   `npx vercel --prod` у корені репозиторію.
7. **Webhook**: `python scripts/set_webhook.py https://<проєкт>.vercel.app/api/webhook`
   (`--info` — стан, `--delete` — повернутися до `scripts/run_bot.py`).

Поки webhook встановлено, `run_polling` не працює (Telegram віддає оновлення лише одному споживачу).

## Обмеження

- Функція без секрету webhook відповідає 503, із неправильним — 403.
- Кеш прогнозу й ліміт запитів живуть у теплому екземплярі функції, тож ліміт «10 за хвилину» наближений.
- Перший запит після простою повільніший (завантаження sklearn і моделей, пробудження Neon).
- Не перевірено на самому Vercel (немає CLI й акаунта в сесії): локально перевірено webhook-обробник на
  справжніх моделях і локальному Postgres, прогнози з Postgres збігаються з SQLite до 0.0.

## Сайт (public/ + /api/forecast)

Той самий Vercel-проєкт віддає статику з `public/` і функцію `api/forecast.py` (JSON для мапи; CDN-кеш 60 с).
Додаткових змінних не треба: функція читає ті ж `DATABASE_URL` і `models/`. Мапа
(`public/geo/map.json`) збирається `python scripts/build_map.py` (качає межі з UNOSAT; `--cache` зберігає
копію). Локальний перегляд на живих даних воркера: `python scripts/dev_site.py` → http://127.0.0.1:5173.
Розгорнуто й перевірено на Vercel (16.10.2026). Єдина точка входу — `api/index.py`; заглушка `site/` — окремий проєкт, сайтом не користується.
