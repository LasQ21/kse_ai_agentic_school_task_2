#!/usr/bin/env bash
# Автозапуск воркера живих даних на macOS (launchd): стартує при вході в систему і перезапускається,
# якщо процес упав або його вбили. Токени — лише в .env, у plist їх немає.
#
#   scripts/install_launchd.sh install     # створити агента й запустити
#   scripts/install_launchd.sh status      # чи завантажений і чи свіжі дані
#   scripts/install_launchd.sh uninstall   # зупинити й прибрати
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LABEL="com.kse.air-alert.worker"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

case "${1:-}" in
  install)
    mkdir -p "$ROOT/data/logs" "$ROOT/data/run" "$HOME/Library/LaunchAgents"
    # ручний воркер (local_stack.sh) зупиняємо, щоб не було двох опитувачів
    if [ -f data/run/worker.pid ] && kill -0 "$(cat data/run/worker.pid)" 2>/dev/null; then
      kill -TERM "$(cat data/run/worker.pid)"; sleep 2
    fi
    rm -f data/run/worker.pid
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$ROOT/.venv/bin/python</string>
    <string>$ROOT/scripts/live_worker.py</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$ROOT/data/logs/worker.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/logs/worker.log</string>
</dict>
</plist>
EOF
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "агент $LABEL встановлено й запущено" ;;
  status)
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state|pid|last exit" || echo "не завантажений"
    .venv/bin/python scripts/live_worker.py --health || true ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "агент прибрано" ;;
  *) echo "використання: $0 install|status|uninstall"; exit 2 ;;
esac
