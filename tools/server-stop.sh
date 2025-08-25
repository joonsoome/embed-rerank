#!/usr/bin/env bash
# Stop background server started by server-run.sh
set -euo pipefail

PIDFILE="${PIDFILE:-/tmp/embed-rerank.pid}"
GRACE=${GRACE:-8}

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
