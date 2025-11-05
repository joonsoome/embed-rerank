#!/usr/bin/env zsh

# 🍎 Apple MLX Embed-Rerank macOS Service Setup (robust)
# Safely generates a LaunchAgent plist from .env, omitting empty optionals,
# validates it, and reloads the service.

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
ok() { echo -e "${GREEN}✅ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; exit 1; }

# Basic paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="com.embed-rerank.server"
LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$LAUNCH_AGENT_DIR/$SERVICE_NAME.plist"
LAUNCHD_ENTRY="$SERVICE_NAME"
LAUNCH_SCRIPT="$PROJECT_DIR/tools/server-launchd.sh"

info "Preparing macOS LaunchAgent for embed-rerank"

[[ "$(uname -s)" == "Darwin" ]] || fail "This script only supports macOS"
[[ -d "$PROJECT_DIR" ]] || fail "Project directory not found: $PROJECT_DIR"
[[ -x "$LAUNCH_SCRIPT" ]] || fail "Launch script not found or not executable: $LAUNCH_SCRIPT"

# Load .env (export all), falling back to .env.example for defaults
ENV_FILE="$PROJECT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    info "Using .env: $ENV_FILE"
else
    ENV_FILE="$PROJECT_DIR/.env.example"
    [[ -f "$ENV_FILE" ]] || fail "Neither .env nor .env.example found at project root"
    warn "Using .env.example (no .env found)"
fi

# Temporarily allow unset vars while sourcing
set +u
set -a
source "$ENV_FILE"
set +a
set -u

# Defaults if not provided
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"
BACKEND="${BACKEND:-auto}"
RERANKER_BACKEND="${RERANKER_BACKEND:-auto}"
# Prefer bounded scores by default for schema compatibility; promote 'none' to 'sigmoid'
RERANK_SCORE_NORM_RAW="${RERANK_SCORE_NORM:-}"
if [[ -z "${RERANK_SCORE_NORM_RAW}" || "${RERANK_SCORE_NORM_RAW}" == "none" ]]; then
    RERANK_SCORE_NORM="sigmoid"
else
    RERANK_SCORE_NORM="${RERANK_SCORE_NORM_RAW}"
fi
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_FORMAT="${LOG_FORMAT:-json}"

# Ensure LaunchAgents dir
mkdir -p "$LAUNCH_AGENT_DIR"

# Helper to print <key>k</key><string>v</string> if v is non-empty
add_env_kv() {
    local k="$1"; shift
    local v="$1"; shift || true
    if [[ -n "${v}" ]]; then
        printf '\t\t<key>%s</key><string>%s</string>\n' "$k" "$v"
    fi
}

info "Writing plist: $PLIST_FILE"
cat > "$PLIST_FILE" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$SERVICE_NAME</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>$LAUNCH_SCRIPT</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <key>EnvironmentVariables</key>
    <dict>
        $(add_env_kv HOST "$HOST")
        $(add_env_kv PORT "$PORT")
        $(add_env_kv BACKEND "$BACKEND")
        $(add_env_kv RERANKER_BACKEND "$RERANKER_BACKEND")
        $(add_env_kv MODEL_NAME "${MODEL_NAME:-}")
        $(add_env_kv MODEL_PATH "${MODEL_PATH:-}")
        $(add_env_kv CROSS_ENCODER_MODEL "${CROSS_ENCODER_MODEL:-}")
        $(add_env_kv RERANKER_MODEL_ID "${RERANKER_MODEL_ID:-}")
        $(add_env_kv RERANKER_MODEL_NAME "${RERANKER_MODEL_NAME:-}")
        $(add_env_kv RERANK_MAX_SEQ_LEN "${RERANK_MAX_SEQ_LEN:-}")
        $(add_env_kv RERANK_BATCH_SIZE "${RERANK_BATCH_SIZE:-}")
        $(add_env_kv RERANK_POOLING "${RERANK_POOLING:-}")
        $(add_env_kv RERANK_SCORE_NORM "${RERANK_SCORE_NORM:-}")
        $(add_env_kv OPENAI_RERANK_AUTO_SIGMOID "${OPENAI_RERANK_AUTO_SIGMOID:-}")
        $(add_env_kv RELOAD "${RELOAD:-}")
        $(add_env_kv BATCH_SIZE "${BATCH_SIZE:-}")
        $(add_env_kv MAX_BATCH_SIZE "${MAX_BATCH_SIZE:-}")
        $(add_env_kv MAX_TEXTS_PER_REQUEST "${MAX_TEXTS_PER_REQUEST:-}")
        $(add_env_kv MAX_PASSAGES_PER_RERANK "${MAX_PASSAGES_PER_RERANK:-}")
        $(add_env_kv MAX_SEQUENCE_LENGTH "${MAX_SEQUENCE_LENGTH:-}")
        $(add_env_kv DEVICE_MEMORY_FRACTION "${DEVICE_MEMORY_FRACTION:-}")
        $(add_env_kv REQUEST_TIMEOUT "${REQUEST_TIMEOUT:-}")
        $(add_env_kv DEFAULT_AUTO_TRUNCATE "${DEFAULT_AUTO_TRUNCATE:-}")
        $(add_env_kv DEFAULT_TRUNCATION_STRATEGY "${DEFAULT_TRUNCATION_STRATEGY:-}")
        $(add_env_kv DEFAULT_MAX_TOKENS_OVERRIDE "${DEFAULT_MAX_TOKENS_OVERRIDE:-}")
        $(add_env_kv DEFAULT_RETURN_PROCESSING_INFO "${DEFAULT_RETURN_PROCESSING_INFO:-}")
        $(add_env_kv DIMENSION_STRATEGY "${DIMENSION_STRATEGY:-}")
        $(add_env_kv OUTPUT_EMBEDDING_DIMENSION "${OUTPUT_EMBEDDING_DIMENSION:-}")
        $(add_env_kv LOG_LEVEL "$LOG_LEVEL")
        $(add_env_kv LOG_FORMAT "$LOG_FORMAT")
        $(add_env_kv PYTHONPATH "$PROJECT_DIR")
    </dict>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/tmp/embed-rerank.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/embed-rerank.err</string>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
EOF

# Validate plist
if /usr/bin/plutil -lint "$PLIST_FILE" >/dev/null; then
    ok "Plist validated"
else
    fail "Plist validation failed: $PLIST_FILE"
fi

# Reload service
info "Reloading LaunchAgent $SERVICE_NAME"
launchctl unload "$PLIST_FILE" >/dev/null 2>&1 || true
if launchctl load "$PLIST_FILE"; then
    ok "Service loaded"
else
    fail "Failed to load service"
fi

# Brief wait and health check
sleep 2
BASE_URL="http://${HOST}:${PORT}"
info "Health check: ${BASE_URL}/health/"
if curl -fsS "${BASE_URL}/health/" >/dev/null; then
    ok "Service is responding: ${BASE_URL}"
    info "Docs: ${BASE_URL}/docs"
else
    warn "Service not responding yet; it may still be starting. Check logs: tail -f /tmp/embed-rerank.log"
fi

echo
info "Service management shortcuts:"
echo "  Start:   launchctl load $PLIST_FILE"
echo "  Stop:    launchctl unload $PLIST_FILE"
echo "  Restart: launchctl unload $PLIST_FILE && launchctl load $PLIST_FILE"
echo "  Status:  launchctl list | grep $SERVICE_NAME"
echo "  Logs:    tail -f /tmp/embed-rerank.log"
echo "  Errors:  tail -f /tmp/embed-rerank.err"
echo
ok "LaunchAgent setup complete"
