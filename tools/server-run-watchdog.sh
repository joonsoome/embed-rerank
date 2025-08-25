#!/usr/bin/env bash
# Lightweight watchdog for embed-rerank server
# Periodically hits /health and restarts the server if unhealthy or dead.
# Designed to be started by server-run.sh. Avoid starting this manually unless needed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="${PIDFILE:-/tmp/embed-rerank.pid}"
WATCHDOG_PIDFILE="${WATCHDOG_PIDFILE:-/tmp/embed-rerank.watchdog.pid}"
LOGFILE="${LOGFILE:-/tmp/embed-rerank.watchdog.log}"

# Health check configuration
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
HEALTH_PATH="${HEALTH_PATH:-/health/}"  # FastAPI health router path
INTERVAL="${INTERVAL:-30}"               # seconds between checks
FAILURES_FOR_RESTART="${FAILURES_FOR_RESTART:-2}" # consecutive failures required
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3}"  # curl --max-time seconds

HEALTH_URL="http://${HOST}:${PORT}${HEALTH_PATH}"

# Prevent multiple watchdog instances
if [[ -f "$WATCHDOG_PIDFILE" ]]; then
  if kill -0 "$(cat "$WATCHDOG_PIDFILE")" 2>/dev/null; then
    echo "[watchdog] Already running (PID $(cat "$WATCHDOG_PIDFILE"))" | tee -a "$LOGFILE"
    exit 0
  else
    echo "[watchdog] Stale watchdog PID file. Removing." | tee -a "$LOGFILE"
    rm -f "$WATCHDOG_PIDFILE"
  fi
fi
echo $$ > "$WATCHDOG_PIDFILE"

echo "[watchdog] Started (PID $$). Monitoring ${HEALTH_URL} every ${INTERVAL}s (failures threshold=${FAILURES_FOR_RESTART})" | tee -a "$LOGFILE"

consecutive_failures=0

restart_server() {
  echo "[watchdog] Restart trigger at $(date -u +'%Y-%m-%dT%H:%M:%SZ') (failures=${consecutive_failures})." | tee -a "$LOGFILE"
  # Attempt graceful stop if PID exists
  if [[ -f "$PIDFILE" ]]; then
    srv_pid="$(cat "$PIDFILE")"
    if kill -0 "$srv_pid" 2>/dev/null; then
      echo "[watchdog] Sending SIGTERM to server PID $srv_pid" | tee -a "$LOGFILE"
      kill "$srv_pid" || true
      # wait up to 10s
      for i in {1..10}; do
        if kill -0 "$srv_pid" 2>/dev/null; then sleep 1; else break; fi
      done
      if kill -0 "$srv_pid" 2>/dev/null; then
        echo "[watchdog] Force killing server PID $srv_pid" | tee -a "$LOGFILE"
        kill -9 "$srv_pid" || true
      fi
    fi
  fi
  # Restart via server-run.sh but prevent spawning a second watchdog
  echo "[watchdog] Starting fresh server instance" | tee -a "$LOGFILE"
  # Use bash -lc with a single argument so paths with spaces are handled
  # correctly by the shell. This avoids splitting $REPO_ROOT on spaces.
  WATCHDOG_SKIP=1 bash -lc "exec '$REPO_ROOT/tools/server-run.sh'" >> "$LOGFILE" 2>&1 || {
    echo "[watchdog] Restart attempt failed" | tee -a "$LOGFILE"
  }
}

while true; do
  # If server process missing, count as failure (but only once per loop)
  server_alive=0
  if [[ -f "$PIDFILE" ]]; then
    if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      server_alive=1
    fi
  fi

  if [[ "$server_alive" -eq 1 ]] && curl -fsS --max-time "$REQUEST_TIMEOUT" "$HEALTH_URL" >/dev/null 2>&1; then
    if [[ "$consecutive_failures" -ne 0 ]]; then
      echo "[watchdog] Health restored after ${consecutive_failures} failure(s)" | tee -a "$LOGFILE"
    fi
    consecutive_failures=0
  else
    consecutive_failures=$((consecutive_failures + 1))
    echo "[watchdog] Health check failed (${consecutive_failures}/${FAILURES_FOR_RESTART})" | tee -a "$LOGFILE"
    if [[ "$consecutive_failures" -ge "$FAILURES_FOR_RESTART" ]]; then
      restart_server
      consecutive_failures=0
    fi
  fi
  sleep "$INTERVAL"
done
