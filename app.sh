#!/usr/bin/env bash
#
# app.sh — start/stop the airfoil-surrogate web app.
#
#   ./app.sh          start   → http://127.0.0.1:8000
#   ./app.sh --kill   stop
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000
APP_LOG=/tmp/turbulent-flows-app.log

KILL=0
for arg in "$@"; do
  case "$arg" in
    --kill) KILL=1 ;;
    -h|--help)
      echo "usage: ./app.sh [--kill]"
      echo "  (no flag)  start  → http://127.0.0.1:$PORT"
      echo "  --kill     stop the app"
      exit 0 ;;
    *) echo "unknown argument: $arg (try --help)"; exit 1 ;;
  esac
done

# Kill whatever is listening on $1, identified by PORT only (macOS/Linux via lsof).
kill_port() {
  local port="$1" pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  fi
  [ -z "$pids" ] && return 1
  kill $pids 2>/dev/null || true
  if kill -0 $pids 2>/dev/null; then
    sleep 1
    kill -9 $pids 2>/dev/null || true
  fi
  return 0
}

if [ "$KILL" = 1 ]; then
  echo "stopping app..."
  kill_port "$PORT" && echo "  stopped app on :$PORT" || echo "  nothing running on :$PORT"
  exit 0
fi

echo "stopping any running instance..."
kill_port "$PORT" && sleep 1 || true

cd "$REPO_DIR"
echo "starting app on http://127.0.0.1:$PORT ..."
PYTHONPATH=src:src/app nohup uv run python src/app/app.py --host 127.0.0.1 --port "$PORT" \
  >"$APP_LOG" 2>&1 &

up=0
for _ in $(seq 1 40); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/healthz" 2>/dev/null; then up=1; break; fi
  sleep 1
done
if [ "$up" != 1 ]; then
  echo "ERROR: app did not start — last lines of $APP_LOG:"; tail -n 20 "$APP_LOG"; exit 1
fi
echo "  app is up.  http://127.0.0.1:$PORT   (log: $APP_LOG)"
