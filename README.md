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

🎉 **Done!** Visit [http://localhost:9000/docs](http://localhost:9000/docs) for interactive API documentation.

---

## 🛠 Server Management (macOS/Linux)

```bash
# Start server (background)
./tools/server-run.sh

# Start server (foreground/development)
./tools/server-run-foreground.sh

# Stop server
./tools/server-stop.sh

# Run comprehensive tests
./tools/server-tests.sh
```

> **Windows Support**: Coming soon! Currently optimized for macOS/Linux.

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

## 🧪 Testing

```bash
# Comprehensive test suite
./tools/server-tests.sh

# Quick health check
curl http://localhost:9000/health/

# Check model cache location
python3 -c "import os; print('Cache:', os.path.expanduser('~/.cache/huggingface/hub'))"

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

---

## 🎯 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/joonsoo-me/embed-rerank.git
cd embed-rerank
python -m venv .venv && source .venv/bin/activate

# 2. Install and run
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 9000

# 3. Test it works
curl http://localhost:9000/health/
```

🎉 **Done!** Visit [http://localhost:9000/docs](http://localhost:9000/docs) for interactive API documentation.

---

## 🌐 Three APIs, One Service

Your service speaks **three languages fluently** - use whichever fits your project:

| API Standard | Embedding Endpoint | Reranking Endpoint | Use Case |
|-------------|-------------------|-------------------|----------|
| **🏠 Native** | `/api/v1/embed` | `/api/v1/rerank` | New projects, full control |
| **🤖 OpenAI** | `/v1/embeddings` | - | Existing OpenAI code |
| **🔄 TEI** | `/embed` | `/rerank` | Hugging Face TEI replacement |

### OpenAI Compatible (Drop-in Replacement)

```python
import openai

# Just change the base_url - everything else stays the same!
client = openai.OpenAI(
    api_key="dummy-key",  # Not used, but required
    base_url="http://localhost:9000/v1"
)

response = client.embeddings.create(
    input=["Hello world", "Apple Silicon is fast!"],
    model="text-embedding-ada-002"
)

print(f"Generated {len(response.data)} embeddings")
# 🚀 10x faster than OpenAI, same exact code!
```

### TEI Compatible (Hugging Face Replacement)

```bash
# Before: Hugging Face TEI
curl -X POST "https://api-inference.huggingface.co/models/..." 
     -H "Authorization: Bearer $HF_TOKEN" 
     -d '{"inputs": ["Hello world"]}'

# After: Your Apple MLX service  
curl -X POST "http://localhost:9000/embed" 
     -d '{"inputs": ["Hello world"], "truncate": true}'

# Same API, 10x faster, $0 cost! 🎯
```

### Native API (Full Control)

```bash
# Embeddings
curl -X POST "http://localhost:9000/api/v1/embed/" 
     -H "Content-Type: application/json" 
     -d '{"texts": ["Apple Silicon", "MLX acceleration"]}'

# Reranking  
curl -X POST "http://localhost:9000/api/v1/rerank/" 
     -H "Content-Type: application/json" 
     -d '{
       "query": "machine learning",
       "passages": ["AI is cool", "Dogs are pets", "MLX is fast"]
     }'
```

---

## ⚙️ Configuration

Create `.env` file for customization:

```env
```env
BACKEND=auto                                   # auto | mlx | torch
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ
HOST=0.0.0.0
PORT=9000
BATCH_SIZE=32
MAX_TEXTS_PER_REQUEST=100
LOG_LEVEL=INFO
LOG_FORMAT=json

# Model Cache (optional overrides)
MODEL_PATH=/path/to/custom/model/dir      # Custom model location
TRANSFORMERS_CACHE=/path/to/hf/cache          # HF transformers cache
HF_HOME=/path/to/hf/home                      # HF home directory
```

**Model Storage Priority**:
1. `MODEL_PATH` (if set) → Use custom directory
2. `TRANSFORMERS_CACHE` (if set) → Use custom HF cache
3. `HF_HOME/hub` (if HF_HOME set) → Use custom HF home
4. `~/.cache/huggingface/hub/` → Default location

Runtime precedence: request params > headers > env defaults.
```

### 📂 Model Cache Management

The service automatically manages model downloads and caching:

| Environment Variable | Purpose | Default |
|---------------------|---------|---------|
| `MODEL_PATH` | Custom model directory | *(uses HF cache)* |
| `TRANSFORMERS_CACHE` | Override HF cache location | `~/.cache/huggingface/transformers` |
| `HF_HOME` | HF home directory | `~/.cache/huggingface` |
| *(auto)* | Default HF cache | `~/.cache/huggingface/hub/` |

**First Run**: Downloads `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` (~2.3GB) to cache  
**Subsequent Runs**: Loads from cached files (sub-second startup)

**Cache Location Check**:
```bash
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

## 🧪 Testing

```bash
# Test all endpoints (comprehensive)
./tools/server-tests.sh

# Quick health check with model cache info
curl http://localhost:9000/health/

# Check where models are stored
python3 -c "
import os
print('Current model cache settings:')
print('  MLX_MODEL_PATH:', os.getenv('MLX_MODEL_PATH', '<default>'))
print('  TRANSFORMERS_CACHE:', os.getenv('TRANSFORMERS_CACHE', '<default>'))
print('  HF_HOME:', os.getenv('HF_HOME', '<default>'))
print('  Default cache:', os.path.expanduser('~/.cache/huggingface/hub'))
"

# Run pytest
pytest tests/ -v
```

---

## 🚀 What You Get

- **✅ Zero Code Changes**: Drop-in replacement for OpenAI API and TEI
- **⚡ 10x Performance**: Apple MLX acceleration on Apple Silicon  
- **💰 Zero Costs**: No API fees, runs locally
- **🔒 Privacy**: Your data never leaves your machine
- **🎯 Three APIs**: Native, OpenAI, and TEI compatibility
- **📊 Production Ready**: Health checks, monitoring, structured logging
