#!/usr/bin/env bash
# Локальне розгортання: воркер живих даних + API на 127.0.0.1, а також статичний знімок для сайту.
# Тунель (cloudflared) запускається окремо вручну й сюди не входить.
#
#   scripts/local_stack.sh start     # воркер + API (127.0.0.1:8000, CORS лише для сайту)
#   scripts/local_stack.sh status
#   scripts/local_stack.sh stop
#   scripts/local_stack.sh snapshot  # новий site/public/snapshot.json однією командою (~15 хв)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=data/run; LOG=data/logs; mkdir -p "$RUN" "$LOG"
SITE_ORIGIN="${CORS_ALLOW_ORIGINS:-https://kse-air-alert-risk.vercel.app}"

alive() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }

start_worker() {
  nohup .venv/bin/python scripts/live_worker.py >> "$LOG/worker.log" 2>&1 & echo $! > "$RUN/worker.pid"
  echo "worker запущено (pid $!)"
}
start_api() {
  CORS_ALLOW_ORIGINS="$SITE_ORIGIN" nohup .venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8000 \
    --no-server-header >> "$LOG/api.log" 2>&1 & echo $! > "$RUN/api.pid"
  echo "api запущено на http://127.0.0.1:8000 (pid $!), CORS: $SITE_ORIGIN"
}
stop_one() {
  if alive "$1"; then kill -TERM "$(cat "$RUN/$1.pid")"; echo "$1 зупинено"; else echo "$1 не працював"; fi
  rm -f "$RUN/$1.pid"
}

case "${1:-}" in
  start)
    alive worker && echo "worker вже працює (pid $(cat $RUN/worker.pid))" || start_worker
    alive api && echo "api вже працює (pid $(cat $RUN/api.pid))" || start_api ;;
  stop)
    stop_one api; stop_one worker ;;
  status)
    for n in worker api; do alive $n && echo "$n: працює (pid $(cat $RUN/$n.pid))" || echo "$n: зупинено"; done
    .venv/bin/python scripts/live_worker.py --health || true ;;
  snapshot)
    started=""
    alive worker || { start_worker; started="$started worker"; }
    alive api || { start_api; started="$started api"; }
    echo "знімок історії alerts.in.ua (закриває діри; ~14 хв через ліміт API)..."
    .venv/bin/python scripts/archive_alerts.py > "$LOG/archive_snapshot.log" 2>&1
    .venv/bin/python scripts/live_poll.py --import-archive
    for _ in $(seq 1 36); do .venv/bin/python scripts/live_worker.py --health >/dev/null 2>&1 && break; sleep 5; done
    .venv/bin/python scripts/check_live_coverage.py | tail -2
    .venv/bin/python scripts/make_snapshot.py
    (cd site && node build.js)
    for n in $started; do stop_one "$n"; done
    echo "готово. Опублікувати: cd site && npx vercel deploy --prod --yes" ;;
  *) echo "використання: $0 start|stop|status|snapshot"; exit 2 ;;
esac
