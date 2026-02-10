#!/bin/sh
set -eu

VENV_PY="/Users/bear8203/Works/Development Projects/embed-rerank/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "❌ venv python not found: $VENV_PY"
  echo "   Create the venv first (e.g., python -m venv .venv) and install deps."
  exit 1
fi

"$VENV_PY" -c "from huggingface_hub import snapshot_download; snapshot_download('vserifsaglam/Qwen3-Reranker-4B-4bit-MLX', resume_download=True)"

echo "✅ Rerank model download complete"
