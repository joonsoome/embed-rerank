#!/usr/bin/env bash
# Background server runner for embed-rerank
# Loads .env, activates .venv, starts uvicorn in background with PID + logs
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PIDFILE="${PIDFILE:-/tmp/embed-rerank.pid}"
LOGFILE="${LOGFILE:-/tmp/embed-rerank.log}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
# Watchdog control
WATCHDOG="${WATCHDOG:-1}"              # set 0 to disable automatic watchdog launch
WATCHDOG_PIDFILE="${WATCHDOG_PIDFILE:-/tmp/embed-rerank.watchdog.pid}"
# Log rotation configuration
LOG_ROTATE_ENABLE="${LOG_ROTATE_ENABLE:-1}"   # set 0 to disable rotation
LOG_MAX_MB="${LOG_MAX_MB:-50}"               # rotate if current log exceeds this size (MB)
LOG_ROTATE_COUNT="${LOG_ROTATE_COUNT:-5}"     # number of old logs to keep

rotate_log_if_needed() {
  [[ "$LOG_ROTATE_ENABLE" == "1" ]] || return 0
  [[ -f "$LOGFILE" ]] || return 0
  # macOS stat differs from GNU; use portable approach
  if command -v stat >/dev/null 2>&1; then
    if stat -f%z "$LOGFILE" >/dev/null 2>&1; then
      bytes=$(stat -f%z "$LOGFILE")
    else
      bytes=$(stat -c%s "$LOGFILE")
    fi
  else
    bytes=$(wc -c < "$LOGFILE" 2>/dev/null || echo 0)
  fi
  max_bytes=$(( LOG_MAX_MB * 1024 * 1024 ))
  if (( bytes < max_bytes )); then
    return 0
  fi
  echo "Rotating log $LOGFILE (size $bytes bytes >= $max_bytes)" >&2
  # Shift old logs
  for (( i=LOG_ROTATE_COUNT; i>=1; i-- )); do
    prev=$(( i - 1 ))
    if (( prev == 0 )); then
      src="$LOGFILE"
    else
      src="${LOGFILE}.${prev}"
    fi
    dst="${LOGFILE}.${i}"
    if [[ -f "$src" ]]; then
      mv "$src" "$dst" 2>/dev/null || true
    fi
  done
  # Recreate fresh log file
  : > "$LOGFILE"
}

if [[ -f "$PIDFILE" ]]; then
  if kill -0 "$(cat "$PIDFILE")" >/dev/null 2>&1; then
    echo "Server already running (PID $(cat "$PIDFILE"))"
    exit 0
  else
    echo "Stale PID file. Removing." >&2
    rm -f "$PIDFILE"
  fi
fi

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "Virtualenv .venv not found. Create with: python -m venv .venv" >&2
  exit 2
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . .env
  set +a
fi

if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python; else echo "python not found" >&2; exit 3; fi

CMD=("$PY" -m uvicorn app.main:app --host "${HOST}" --port "${PORT}")

echo "Starting server in background: ${CMD[*]}" | tee -a "$LOGFILE"
rotate_log_if_needed
nohup "${CMD[@]}" >>"$LOGFILE" 2>&1 &
PID=$!
echo $PID >"$PIDFILE"
# detach
if command -v disown >/dev/null 2>&1; then disown "$PID" || true; fi

echo "Started (PID $PID). Logs: $LOGFILE"

# Launch watchdog (unless explicitly skipped or disabled)
if [[ "${WATCHDOG_SKIP:-0}" != "1" && "$WATCHDOG" == "1" ]]; then
  if [[ -f "$WATCHDOG_PIDFILE" ]] && kill -0 "$(cat "$WATCHDOG_PIDFILE")" 2>/dev/null; then
    echo "Watchdog already running (PID $(cat "$WATCHDOG_PIDFILE"))"
  else
    echo "Starting watchdog..." | tee -a "$LOGFILE"
  # Execute watchdog script directly (ensure script is executable). Quoting the
  # full path prevents word-splitting when REPO_ROOT contains spaces.
  nohup "$REPO_ROOT/tools/server-run-watchdog.sh" >> "$LOGFILE" 2>&1 &
    if command -v disown >/dev/null 2>&1; then disown $! || true; fi
  fi
fi
