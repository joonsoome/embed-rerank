#!/bin/bash

# 🧪 GitHub CI Tests Runner - Apple MLX Embed-Rerank
# Locally execute the same tests that run in GitHub Actions CI

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

info "🧪 Starting GitHub CI Test Suite Locally"
info "📁 Project directory: $PROJECT_DIR"

cd "$PROJECT_DIR"

# Check for virtual environment
if [[ ! -d ".venv" && ! -d "venv" ]]; then
    error "Virtual environment not found. Please create with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi

# Activate virtual environment
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
    info "🐍 Using .venv virtual environment"
elif [[ -d "venv" ]]; then
    source venv/bin/activate
    info "🐍 Using venv virtual environment"
fi

# Set CI environment variables
export BACKEND=torch
export LOG_LEVEL=ERROR
export CI=true
export PYTHONPATH=.

info "🔧 Environment configured for CI testing"

# Test 1: Import Tests
info "📦 Running import tests..."
python -c "from app.main import app; print('✅ App import successful')" || error "App import failed"
python -c "from app.backends.factory import BackendFactory; print('✅ Backend factory import successful')" || error "Backend factory import failed"
python -c "from app.models.requests import EmbedRequest; print('✅ Models import successful')" || error "Models import failed"
success "📦 All imports successful"

# Test 2: Core CI Tests
info "🧪 Running core CI tests..."
if pytest tests/test_ci_quick.py -v --tb=short; then
    success "🧪 All core CI tests passed"
else
    error "Core CI tests failed"
fi

# Test 3: Code Quality (Optional - informational only)
info "🎨 Checking code quality..."

echo "📝 Black formatting check:"
if black --check --line-length 120 app/ tests/ 2>/dev/null; then
    success "📝 Black formatting: OK"
else
    warning "📝 Black formatting: Some files need formatting (run: black --line-length 120 app/ tests/)"
fi

echo "🔤 isort import sorting check:"
if isort --check-only --profile black app/ tests/ 2>/dev/null; then
    success "🔤 isort import sorting: OK"
else
    warning "🔤 isort import sorting: Some files need sorting (run: isort --profile black app/ tests/)"
fi

echo "🔍 flake8 linting check:"
if flake8 app/ tests/ --max-line-length=120 --extend-ignore=E203,W503 --exclude=__pycache__ 2>/dev/null; then
    success "🔍 flake8 linting: OK"
else
    warning "🔍 flake8 linting: Some issues found (see above)"
fi

# Test 4: Basic functionality test (if no server running)
info "🚀 Testing basic functionality..."
if ! curl -s -f http://localhost:9000/health/ > /dev/null 2>&1; then
    info "🔧 Starting temporary test server..."
    
    # Start server in background
    python -m uvicorn app.main:app --host 127.0.0.1 --port 18080 &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 5
    
    # Test basic endpoints
    if curl -s -f http://127.0.0.1:18080/health/ > /dev/null; then
        success "🚀 Basic server functionality: OK"
    else
        warning "🚀 Basic server functionality: Could not connect"
    fi
    
    # Clean up
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
else
    info "🚀 Server already running, skipping basic functionality test"
fi

# Summary
echo
success "🎉 GitHub CI Test Suite Complete!"
info "📊 Test Summary:"
echo "   ✅ Import tests: PASSED"
echo "   ✅ Core CI tests: PASSED"
echo "   ⚠️  Code quality: CHECK WARNINGS ABOVE"
echo "   ✅ Basic functionality: TESTED"
echo
info "💡 To fix code quality issues:"
echo "   black --line-length 120 app/ tests/"
echo "   isort --profile black app/ tests/"
echo "   flake8 app/ tests/ --max-line-length=120 --extend-ignore=E203,W503"
echo
success "🚀 Ready for GitHub CI!"
