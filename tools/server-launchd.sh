#!/usr/bin/env bash
# LaunchAgent-only startup script
#  - launchd manages process lifecycle (restart / stdout & stderr redirection)
#  - No backgrounding, nohup, PID files, or watchdog required here
#  - Clear separation: server-run.sh (manual/dev, background + watchdog) vs server-launchd.sh (foreground for launchd)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Provide sane locale defaults (launchd often supplies a minimal environment)
export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

# Prepend project virtualenv to PATH to ensure correct python & scripts are found first
export PATH="$REPO_ROOT/.venv/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

VENV_PY="./.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "[launchd] .venv python not found. Create & install deps:\n  python -m venv .venv\n  . .venv/bin/activate\n  pip install -r requirements.txt" >&2
  exit 2
fi

# Load .env (if present) to populate environment variables
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . .env
  set +a
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"

echo "[launchd] Starting embed-rerank (HOST=${HOST} PORT=${PORT}) using $($VENV_PY -V 2>/dev/null || echo python)"

# Foreground exec so launchd can supervise the process directly
exec "$VENV_PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT"