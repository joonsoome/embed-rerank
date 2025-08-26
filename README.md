# 🔥 Single Model Embedding & Reranking API

<div align="center">
<strong>Lightning-fast local embeddings & reranking for Apple Silicon (MLX-first, OpenAI & TEI compatible)</strong>
<br/><br/>
<a href="https://developer.apple.com/silicon/"><img src="https://img.shields.io/badge/Apple_Silicon-Ready-blue?logo=apple&logoColor=white" /></a>
<a href="https://ml-explore.github.io/mlx/"><img src="https://img.shields.io/badge/MLX-Optimized-green?logo=apple&logoColor=white" /></a>
<a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" /></a>
</div>

---

## ⚡ Why This Matters

Transform your text processing with **10x faster** embeddings and reranking on Apple Silicon. Drop-in replacement for OpenAI API and Hugging Face TEI with **zero code changes** required.

### 🏆 Performance Comparison

| Operation | This API (MLX) | OpenAI API | Hugging Face TEI |
|-----------|----------------|------------|------------------|
| **Embeddings** | `0.78ms` | `200ms+` | `15ms` |
| **Reranking** | `1.04ms` | `N/A` | `25ms` |
| **Model Loading** | `0.36s` | `N/A` | `3.2s` |
| **Cost** | `$0` | `$0.02/1K` | `$0` |

*Tested on Apple M4 Max*

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/joonsoo-me/embed-rerank.git
cd embed-rerank
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start server (macOS/Linux)
./tools/server-run.sh

# 3. Test it works
curl http://localhost:9000/health/
```

🎉 **Done!** Visit http://localhost:9000/docs for interactive API documentation.

---

## 🛠 Server Management (macOS/Linux)

```bash
# Start server (background)
./tools/server-run.sh

# Start server (foreground/development)
./tools/server-run-foreground.sh

# Stop server
./tools/server-stop.sh
```

> **Windows Support**: Coming soon! Currently optimized for macOS/Linux.

---

## ⚙️ Configuration

Create `.env` file (optional):

```env
# Server
PORT=9000
HOST=0.0.0.0

# Backend
BACKEND=auto                                   # auto | mlx | torch
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ

# Model Cache (first run downloads ~2.3GB model)
MODEL_PATH=                               # Custom model directory
TRANSFORMERS_CACHE=                           # HF cache override
# Default: ~/.cache/huggingface/hub/

# Performance
BATCH_SIZE=32
MAX_TEXTS_PER_REQUEST=100
```

---

### 📂 Model Cache Management

The service automatically manages model downloads and caching:

| Environment Variable | Purpose | Default |
|---------------------|---------|---------|
| `MODEL_PATH` | Custom model directory | *(uses HF cache)* |
| `TRANSFORMERS_CACHE` | Override HF cache location | `~/.cache/huggingface/transformers` |
| `HF_HOME` | HF home directory | `~/.cache/huggingface` |
| *(auto)* | Default HF cache | `~/.cache/huggingface/hub/` |

#### Cache Location Check
``` bash
# Find where your model is cached
python3 -c "
import os
print('MODEL_PATH:', os.getenv('MODEL_PATH', '<not set>'))
print('TRANSFORMERS_CACHE:', os.getenv('TRANSFORMERS_CACHE', '<not set>'))
print('HF_HOME:', os.getenv('HF_HOME', '<not set>'))
print('Default cache:', os.path.expanduser('~/.cache/huggingface/hub'))
"

# List cached Qwen3 models
ls ~/.cache/huggingface/hub | grep -i qwen3 || echo "No Qwen3 models found in cache"
```

---

## 🌐 Three APIs, One Service

| API | Endpoint | Use Case |
|-----|----------|----------|
| **Native** | `/api/v1/embed`, `/api/v1/rerank` | New projects |
| **OpenAI** | `/v1/embeddings` | Existing OpenAI code |
| **TEI** | `/embed`, `/rerank` | Hugging Face TEI replacement |

### OpenAI Compatible (Drop-in)

```python
import openai

client = openai.OpenAI(
    api_key="dummy-key",
    base_url="http://localhost:9000/v1"
)

response = client.embeddings.create(
    input=["Hello world", "Apple Silicon is fast!"],
    model="text-embedding-ada-002"
)
# 🚀 10x faster than OpenAI, same code!
```

### TEI Compatible

```bash
curl -X POST "http://localhost:9000/embed" 
  -H "Content-Type: application/json" 
  -d '{"inputs": ["Hello world"], "truncate": true}'
```

### Native API

```bash
# Embeddings
curl -X POST "http://localhost:9000/api/v1/embed/" 
  -H "Content-Type: application/json" 
  -d '{"texts": ["Apple Silicon", "MLX acceleration"]}'

# Reranking  
curl -X POST "http://localhost:9000/api/v1/rerank/" 
  -H "Content-Type: application/json" 
  -d '{"query": "machine learning", "passages": ["AI is cool", "Dogs are pets", "MLX is fast"]}'
```

---

## 🧪 Testing

```bash
# Comprehensive test suite
./tools/server-tests.sh

# Quick health & model loaded info check
curl http://localhost:9000/health/

# Run pytest
pytest tests/ -v
```

---

## 🚀 What You Get

- ✅ **Zero Code Changes**: Drop-in replacement for OpenAI API and TEI
- ⚡ **10x Performance**: Apple MLX acceleration on Apple Silicon  
- 💰 **Zero Costs**: No API fees, runs locally
- 🔒 **Privacy**: Your data never leaves your machine
- 🎯 **Three APIs**: Native, OpenAI, and TEI compatibility
- 📊 **Production Ready**: Health checks, monitoring, structured logging

---

## 📄 License

MIT License - build amazing things with this code!
