#!/usr/bin/env bash
# Локальний запуск: воркер живих даних (alerts.in.ua -> data/live/alerts.sqlite) і Telegram-бот
# (long polling; токен — TELEGRAM_BOT_TOKEN у .env). Логи — data/logs/.
#
#   scripts/local_stack.sh start
#   scripts/local_stack.sh status
#   scripts/local_stack.sh stop
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=data/run; LOG=data/logs; mkdir -p "$RUN" "$LOG"

alive() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }

start_worker() {
  nohup .venv/bin/python scripts/live_worker.py >> "$LOG/worker.log" 2>&1 & echo $! > "$RUN/worker.pid"
  echo "worker запущено (pid $!)"
}
start_bot() {
  nohup .venv/bin/python scripts/run_bot.py >> "$LOG/bot.log" 2>&1 & echo $! > "$RUN/bot.pid"
  echo "bot запущено (pid $!)"
}
stop_one() {
  if alive "$1"; then kill -TERM "$(cat "$RUN/$1.pid")"; echo "$1 зупинено"; else echo "$1 не працював"; fi
  rm -f "$RUN/$1.pid"
}

case "${1:-}" in
  start)
    alive worker && echo "worker вже працює (pid $(cat $RUN/worker.pid))" || start_worker
    alive bot && echo "bot вже працює (pid $(cat $RUN/bot.pid))" || start_bot ;;
  stop)
    stop_one bot; stop_one worker ;;
  status)
    for n in worker bot; do alive $n && echo "$n: працює (pid $(cat $RUN/$n.pid))" || echo "$n: зупинено"; done
    .venv/bin/python scripts/live_worker.py --health || true ;;
  *) echo "використання: $0 start|stop|status"; exit 2 ;;
esac
