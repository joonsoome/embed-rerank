#!/usr/bin/env bash
# Stop background server started by server-run.sh
set -euo pipefail

PIDFILE="${PIDFILE:-/tmp/embed-rerank.pid}"
WATCHDOG_PIDFILE="${WATCHDOG_PIDFILE:-/tmp/embed-rerank.watchdog.pid}"
GRACE=${GRACE:-8}

stop_watchdog() {
  local wpid
  if [[ -f "$WATCHDOG_PIDFILE" ]]; then
    wpid=$(cat "$WATCHDOG_PIDFILE" 2>/dev/null || true)
    if [[ -n "$wpid" ]] && kill -0 "$wpid" >/dev/null 2>&1; then
      echo "Stopping watchdog PID $wpid" >&2
      kill -TERM "$wpid" 2>/dev/null || true
      for i in 1 2 3; do
        if kill -0 "$wpid" >/dev/null 2>&1; then sleep 1; else break; fi
      done
      if kill -0 "$wpid" >/dev/null 2>&1; then
        echo "Force killing watchdog PID $wpid" >&2
        kill -KILL "$wpid" 2>/dev/null || true
        sleep 1
      fi
      if kill -0 "$wpid" >/dev/null 2>&1; then
        echo "Warning: watchdog PID $wpid still alive" >&2
      else
        echo "Watchdog stopped" >&2
      fi
    else
      echo "No active watchdog (stale or missing PID)" >&2
    fi
    rm -f "$WATCHDOG_PIDFILE" || true
  fi
}

# Stop watchdog first to prevent it from restarting the server during shutdown
stop_watchdog

if [[ ! -f "$PIDFILE" ]]; then
  echo "No PID file at $PIDFILE" >&2
  exit 1
fi
PID=$(cat "$PIDFILE")
if ! kill -0 "$PID" >/dev/null 2>&1; then
  echo "Process $PID not running. Removing stale PID file." >&2
  rm -f "$PIDFILE"
  exit 0
fi

echo "Stopping PID $PID (grace $GRACE s)" >&2
kill -TERM "$PID" || true
for i in $(seq 1 "$GRACE"); do
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo "Stopped." >&2
    rm -f "$PIDFILE"
    exit 0
  fi
  sleep 1
done

echo "Force killing PID $PID" >&2
kill -KILL "$PID" || true
sleep 1
if kill -0 "$PID" >/dev/null 2>&1; then
  echo "Failed to terminate PID $PID" >&2
  exit 2
fi
rm -f "$PIDFILE"
echo "Stopped and cleaned up."
