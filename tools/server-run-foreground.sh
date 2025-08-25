#!/usr/bin/env bash
# Foreground server runner for embed-rerank
# Loads .env, activates .venv, runs uvicorn in foreground (Ctrl-C to stop)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"

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

exec "$PY" -m uvicorn app.main:app --host "${HOST}" --port "${PORT}"
