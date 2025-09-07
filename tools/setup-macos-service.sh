#!/bin/bash

# 🍎 Apple MLX Embed-Rerank macOS Service Setup
# Automatically creates LaunchAgent from .env.example configuration

set -euo pipefail

# Color output for better UX
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; exit 1; }

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="com.embed-rerank.server"
LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$LAUNCH_AGENT_DIR/$SERVICE_NAME.plist"

info "🚀 Setting up macOS LaunchAgent for Apple MLX Embed-Rerank Service"

# Check if we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    error "This script is designed for macOS only"
fi

# Check if project directory exists
if [[ ! -d "$PROJECT_DIR" ]]; then
    error "Project directory not found: $PROJECT_DIR"
fi

# Check if .env.example exists
ENV_EXAMPLE="$PROJECT_DIR/.env.example"
if [[ ! -f "$ENV_EXAMPLE" ]]; then
    error ".env.example not found: $ENV_EXAMPLE"
fi

info "📁 Project directory: $PROJECT_DIR"
info "📋 Using configuration from: $ENV_EXAMPLE"

# Parse .env.example for configuration
declare -A CONFIG
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    if [[ $key =~ ^[[:space:]]*# ]] || [[ -z "$key" ]]; then
        continue
    fi
    # Remove leading/trailing whitespace and quotes
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs | sed 's/^["'"'"']//;s/["'"'"']$//')
    if [[ -n "$key" && -n "$value" ]]; then
        CONFIG["$key"]="$value"
    fi
done < "$ENV_EXAMPLE"

# Set default values from .env.example or fallbacks
HOST="${CONFIG[HOST]:-0.0.0.0}"
PORT="${CONFIG[PORT]:-9000}"
BACKEND="${CONFIG[BACKEND]:-auto}"
MODEL_NAME="${CONFIG[MODEL_NAME]:-mlx-community/Qwen3-Embedding-4B-4bit-DWQ}"
MODEL_PATH="${CONFIG[MODEL_PATH]:-}"
CROSS_ENCODER_MODEL="${CONFIG[CROSS_ENCODER_MODEL]:-}"
RELOAD="${CONFIG[RELOAD]:-false}"
BATCH_SIZE="${CONFIG[BATCH_SIZE]:-32}"
MAX_BATCH_SIZE="${CONFIG[MAX_BATCH_SIZE]:-128}"
MAX_TEXTS_PER_REQUEST="${CONFIG[MAX_TEXTS_PER_REQUEST]:-100}"
MAX_PASSAGES_PER_RERANK="${CONFIG[MAX_PASSAGES_PER_RERANK]:-1000}"
MAX_SEQUENCE_LENGTH="${CONFIG[MAX_SEQUENCE_LENGTH]:-512}"
DEVICE_MEMORY_FRACTION="${CONFIG[DEVICE_MEMORY_FRACTION]:-0.8}"
REQUEST_TIMEOUT="${CONFIG[REQUEST_TIMEOUT]:-300}"
DEFAULT_AUTO_TRUNCATE="${CONFIG[DEFAULT_AUTO_TRUNCATE]:-true}"
DEFAULT_TRUNCATION_STRATEGY="${CONFIG[DEFAULT_TRUNCATION_STRATEGY]:-smart_truncate}"
DEFAULT_MAX_TOKENS_OVERRIDE="${CONFIG[DEFAULT_MAX_TOKENS_OVERRIDE]:-}"
DEFAULT_RETURN_PROCESSING_INFO="${CONFIG[DEFAULT_RETURN_PROCESSING_INFO]:-false}"
LOG_LEVEL="${CONFIG[LOG_LEVEL]:-INFO}"
LOG_FORMAT="${CONFIG[LOG_FORMAT]:-json}"

info "🔧 Configuration loaded:"
info "   Host: $HOST"
info "   Port: $PORT"
info "   Backend: $BACKEND"
info "   Model: $MODEL_NAME"
info "   Auto Truncate: $DEFAULT_AUTO_TRUNCATE"
info "   Truncation Strategy: $DEFAULT_TRUNCATION_STRATEGY"

# Check if service is already running
if launchctl list | grep -q "$SERVICE_NAME"; then
    warning "Service $SERVICE_NAME is already loaded"
    read -p "Do you want to restart it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "🔄 Stopping existing service..."
        launchctl unload "$PLIST_FILE" 2>/dev/null || true
    else
        info "ℹ️  Keeping existing service running"
        exit 0
    fi
fi

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENT_DIR"

# Find Python executable in virtual environment or system
PYTHON_EXEC=""
if [[ -f "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python"
    info "🐍 Using virtual environment Python: $PYTHON_EXEC"
elif [[ -f "$PROJECT_DIR/venv/bin/python" ]]; then
    PYTHON_EXEC="$PROJECT_DIR/venv/bin/python"
    info "🐍 Using virtual environment Python: $PYTHON_EXEC"
else
    PYTHON_EXEC="$(which python3 || which python)"
    info "🐍 Using system Python: $PYTHON_EXEC"
fi

if [[ ! -x "$PYTHON_EXEC" ]]; then
    error "Python executable not found or not executable: $PYTHON_EXEC"
fi

# Create the plist file
info "📝 Creating LaunchAgent plist file..."

cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>$SERVICE_NAME</string>
	<key>ProgramArguments</key>
	<array>
		<string>$PYTHON_EXEC</string>
		<string>-m</string>
		<string>uvicorn</string>
		<string>app.main:app</string>
		<string>--host</string>
		<string>$HOST</string>
		<string>--port</string>
		<string>$PORT</string>
	</array>
	<key>WorkingDirectory</key>
	<string>$PROJECT_DIR</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>HOST</key><string>$HOST</string>
		<key>PORT</key><string>$PORT</string>
		<key>BACKEND</key><string>$BACKEND</string>
		<key>MODEL_NAME</key><string>$MODEL_NAME</string>
		<key>MODEL_PATH</key><string>$MODEL_PATH</string>
		<key>CROSS_ENCODER_MODEL</key><string>$CROSS_ENCODER_MODEL</string>
		<key>RELOAD</key><string>$RELOAD</string>
		<key>BATCH_SIZE</key><string>$BATCH_SIZE</string>
		<key>MAX_BATCH_SIZE</key><string>$MAX_BATCH_SIZE</string>
		<key>MAX_TEXTS_PER_REQUEST</key><string>$MAX_TEXTS_PER_REQUEST</string>
		<key>MAX_PASSAGES_PER_RERANK</key><string>$MAX_PASSAGES_PER_RERANK</string>
		<key>MAX_SEQUENCE_LENGTH</key><string>$MAX_SEQUENCE_LENGTH</string>
		<key>DEVICE_MEMORY_FRACTION</key><string>$DEVICE_MEMORY_FRACTION</string>
		<key>REQUEST_TIMEOUT</key><string>$REQUEST_TIMEOUT</string>
		<key>DEFAULT_AUTO_TRUNCATE</key><string>$DEFAULT_AUTO_TRUNCATE</string>
		<key>DEFAULT_TRUNCATION_STRATEGY</key><string>$DEFAULT_TRUNCATION_STRATEGY</string>
		<key>DEFAULT_MAX_TOKENS_OVERRIDE</key><string>$DEFAULT_MAX_TOKENS_OVERRIDE</string>
		<key>DEFAULT_RETURN_PROCESSING_INFO</key><string>$DEFAULT_RETURN_PROCESSING_INFO</string>
		<key>LOG_LEVEL</key><string>$LOG_LEVEL</string>
		<key>LOG_FORMAT</key><string>$LOG_FORMAT</string>
		<key>PYTHONPATH</key><string>$PROJECT_DIR</string>
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

success "📝 Created plist file: $PLIST_FILE"

# Load the service
info "🚀 Loading LaunchAgent service..."
if launchctl load "$PLIST_FILE"; then
    success "🎉 Service loaded successfully!"
else
    error "Failed to load service"
fi

# Wait a moment for service to start
sleep 3

# Check service status
info "🔍 Checking service status..."
if launchctl list | grep -q "$SERVICE_NAME"; then
    success "✅ Service is running!"
    
    # Test the service
    info "🧪 Testing service health..."
    if curl -s -f "http://$HOST:$PORT/health/" > /dev/null; then
        success "🎯 Service is responding correctly!"
        info "📍 Service URL: http://$HOST:$PORT"
        info "📊 Health check: http://$HOST:$PORT/health/"
        info "📚 API docs: http://$HOST:$PORT/docs"
    else
        warning "⚠️  Service is running but not responding yet (may still be starting up)"
        info "📍 Service URL: http://$HOST:$PORT"
        info "📋 Check logs: tail -f /tmp/embed-rerank.log"
        info "📋 Check errors: tail -f /tmp/embed-rerank.err"
    fi
else
    warning "⚠️  Service may not be running properly"
    info "📋 Check status: launchctl list | grep $SERVICE_NAME"
    info "📋 Check logs: tail -f /tmp/embed-rerank.err"
fi

# Service management instructions
echo
info "🛠️  Service Management Commands:"
echo "   ▶️  Start:   launchctl load $PLIST_FILE"
echo "   ⏹️  Stop:    launchctl unload $PLIST_FILE"
echo "   🔄 Restart: launchctl unload $PLIST_FILE && launchctl load $PLIST_FILE"
echo "   📋 Status:  launchctl list | grep $SERVICE_NAME"
echo "   📝 Logs:    tail -f /tmp/embed-rerank.log"
echo "   ❌ Errors:  tail -f /tmp/embed-rerank.err"
echo

success "🎉 macOS LaunchAgent setup complete!"
info "🚀 Your Apple MLX Embed-Rerank service is now running automatically!"
