#!/usr/bin/env bash
# Локальний запуск: воркер живих даних (alerts.in.ua -> data/live/alerts.sqlite).
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
stop_one() {
  if alive "$1"; then kill -TERM "$(cat "$RUN/$1.pid")"; echo "$1 зупинено"; else echo "$1 не працював"; fi
  rm -f "$RUN/$1.pid"
}

case "${1:-}" in
  start)
    alive worker && echo "worker вже працює (pid $(cat $RUN/worker.pid))" || start_worker ;;
  stop)
    stop_one worker ;;
  status)
    alive worker && echo "worker: працює (pid $(cat $RUN/worker.pid))" || echo "worker: зупинено"
    .venv/bin/python scripts/live_worker.py --health || true ;;
  *) echo "використання: $0 start|stop|status"; exit 2 ;;
esac
