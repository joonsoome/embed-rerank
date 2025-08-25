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
nohup "${CMD[@]}" >>"$LOGFILE" 2>&1 &
PID=$!
echo $PID >"$PIDFILE"
# detach
if command -v disown >/dev/null 2>&1; then disown "$PID" || true; fi

echo "Started (PID $PID). Logs: $LOGFILE"
