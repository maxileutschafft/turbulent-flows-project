#!/usr/bin/env bash
#
# app.sh — start/stop the airfoil-surrogate web app (with an optional public tunnel).
#
#   ./app.sh                   start locally only  → http://127.0.0.1:8000, auto device (cuda > mps > cpu)
#   ./app.sh --device mps      start, forcing a specific inference device (cuda | mps | cpu)
#   ./app.sh --public          start + Cloudflare tunnel → prints a public https URL
#   ./app.sh --kill            stop the app (and tunnel)
#
# The inference device is chosen once here at startup — there is no in-UI
# device switch. Both processes are launched detached with nohup, so they
# survive logout. The app always binds to localhost; --public exposes it
# only via the tunnel.
#
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000
APP_LOG=/tmp/turbulent-flows-app.log
CF_LOG=/tmp/turbulent-flows-cloudflared.log
CF_PID=/tmp/turbulent-flows-cf.pid     # PID file for the Cloudflare tunnel
DEVICE=""

PUBLIC=0
KILL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --public) PUBLIC=1; shift ;;
    --kill) KILL=1; shift ;;
    --device)
      DEVICE="${2:-}"
      [ -z "$DEVICE" ] && { echo "error: --device requires a value (cuda|mps|cpu)"; exit 1; }
      shift 2 ;;
    --device=*) DEVICE="${1#*=}"; shift ;;
    -h|--help)
      echo "usage: ./app.sh [--device cuda|mps|cpu] [--public] [--kill]"
      echo "  (no flag)       start locally  → http://127.0.0.1:$PORT, auto device (cuda > mps > cpu)"
      echo "  --device <dev>  start, forcing a specific inference device"
      echo "  --public        start + public Cloudflare tunnel URL"
      echo "  --kill          stop the app and tunnel"
      exit 0 ;;
    *) echo "unknown argument: $1 (try --help)"; exit 1 ;;
  esac
done

case "$DEVICE" in
  ""|cuda|mps|cpu) ;;
  *) echo "error: --device must be one of cuda, mps, cpu (got '$DEVICE')"; exit 1 ;;
esac

# Stop the tunnel (by its recorded PID) and the app (by whoever holds the port).
# No process-name pattern matching, so it can never hit an unrelated process.

# Kill whatever is listening on $1, identified by PORT only. Portable across
# macOS/Linux (lsof), Linux (ss/fuser), and Windows Git Bash (netstat+taskkill).
# Returns 0 if it killed something, 1 otherwise.
kill_port() {
  local port="$1" pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  elif command -v ss >/dev/null 2>&1; then
    pids="$(ss -ltnpH "sport = :${port}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)"
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 && return 0 || return 1
  elif command -v netstat >/dev/null 2>&1 && command -v taskkill >/dev/null 2>&1; then
    # Windows (Git Bash / MSYS)
    pids="$(netstat -ano 2>/dev/null | grep -E "[:.]${port}[[:space:]]+.*LISTENING" | awk '{print $NF}' | sort -u || true)"
    [ -z "$pids" ] && return 1
    local p; for p in $pids; do taskkill //F //PID "$p" >/dev/null 2>&1 || true; done
    return 0
  fi
  [ -z "$pids" ] && return 1
  # graceful SIGTERM; SIGKILL only if the process is still alive after 1 s
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  # shellcheck disable=SC2086
  if kill -0 $pids 2>/dev/null; then
    sleep 1
    kill -9 $pids 2>/dev/null || true
  fi
  return 0
}

stop_all() {
  if [ -f "$CF_PID" ]; then
    kill "$(cat "$CF_PID" 2>/dev/null)" 2>/dev/null && echo "  stopped tunnel" || true
    rm -f "$CF_PID"
  fi
  kill_port "$PORT" && echo "  stopped app on :$PORT" || true
}

# --- shutdown mode ---------------------------------------------------------
if [ "$KILL" = 1 ]; then
  echo "shutting down..."
  stop_all
  echo "done."
  exit 0
fi

# --- start (restart cleanly: stop any existing instance first) -------------
echo "stopping any running instance..."
stop_all
sleep 1

cd "$REPO_DIR"
DEVICE_ARGS=()
[ -n "$DEVICE" ] && DEVICE_ARGS=(--device "$DEVICE")
echo "starting app on http://127.0.0.1:$PORT ...${DEVICE:+ (device: $DEVICE)}"
PYTHONPATH=src:src/app nohup uv run python src/app/app.py --host 127.0.0.1 --port "$PORT" "${DEVICE_ARGS[@]}" \
  >"$APP_LOG" 2>&1 &

# wait for the server to answer /healthz
up=0
for _ in $(seq 1 40); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/healthz" 2>/dev/null; then up=1; break; fi
  sleep 1
done
if [ "$up" != 1 ]; then
  echo "ERROR: app did not start — last lines of $APP_LOG:"; tail -n 20 "$APP_LOG"; exit 1
fi
echo "  app is up.  local: http://127.0.0.1:$PORT   (log: $APP_LOG)"

# --- optional public tunnel ------------------------------------------------
if [ "$PUBLIC" = 1 ]; then
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "ERROR: cloudflared not found in ~/.local/bin — install it first." >&2
    exit 1
  fi
  echo "starting public Cloudflare tunnel..."
  nohup cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" >"$CF_LOG" 2>&1 &
  echo $! > "$CF_PID"
  url=""
  for _ in $(seq 1 30); do
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | head -1 || true)
    [ -n "$url" ] && break
    sleep 1
  done
  if [ -z "$url" ]; then
    echo "ERROR: tunnel URL not found — last lines of $CF_LOG:"; tail -n 20 "$CF_LOG"; exit 1
  fi
  echo ""
  echo "  ============================================================"
  echo "    PUBLIC URL:  $url"
  echo "  ============================================================"
  echo "    No login — share only with people you trust."
  echo "    URL changes on each --public restart.   (log: $CF_LOG)"
  echo ""
fi
