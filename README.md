# 🔥 Single Model Embedding & Reranking API

**Lightning-fast text embeddings and document reranking powered by Apple Silicon & MLX**

<div align="center">

[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Ready-blue?logo=apple&logoColor=white)](https://developer.apple.com/silicon/)
[![MLX Framework](https://img.shields.io/b  print('  MODEL_PATH:', os.getenv('MODEL_PATH', '<default>'))dge/MLX-Optimized-green?logo=apple&logoColor=white)](https://ml-explore.github.io/mlx/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

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

# � Single Model Embedding & Reranking API

<div align="center">
<strong>Lightning-fast local embeddings & reranking for Apple Silicon (MLX-first, OpenAI & TEI compatible)</strong>
<br/><br/>
<a href="https://developer.apple.com/silicon/"><img src="https://img.shields.io/badge/Apple_Silicon-Ready-blue?logo=apple&logoColor=white" /></a>
<a href="https://ml-explore.github.io/mlx/"><img src="https://img.shields.io/badge/MLX-Optimized-green?logo=apple&logoColor=white" /></a>
<a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" /></a>
</div>

---

## 📌 TL;DR
One lightweight service. One quantized model. Three fully compatible APIs (Native / OpenAI / TEI). Sub‑millisecond embedding + reranking on Apple Silicon using MLX with automatic fallback to PyTorch (MPS) or CPU.
✅ **OpenAI Migration**: "Same OpenAI SDK code, but local and lightning fast"  
---

## ⚡ Why It Matters

| Benefit | What It Means |
|---------|---------------|
| 10x+ speed | Local MLX inference: no network latency |
| Zero cost | No paid API calls – everything stays on-device |
| Private by design | Your data never leaves your machine |
| True drop‑in | Works with existing OpenAI SDK & TEI clients |
| Single model | Embedding + rerank = simpler deployment |

### 🏆 Performance Snapshot (M4 Max)

| Metric | MLX | PyTorch (MPS) | OpenAI Remote* | Hugging Face TEI* |
|--------|-----|---------------|----------------|------------------|
| Embedding latency (2 texts) | 0.78 ms | 35 ms | 200 ms+ | 15 ms |
| Rerank latency (3 passages) | 1.04 ms | 150 ms | N/A | 25 ms |
| Model load (cached) | 0.36 s | 2.71 s | Network | 3.2 s |
| Cost / 1K embed | $0 | $0 | $0.02 | $0 |

*Remote APIs include network overhead.
---
> Numbers are indicative; adjust expectations for different hardware / batch sizes.
## 📄 License

MIT License - build amazing things with this code!
## 🚀 Quick Start
## 🧠 AI Operations Made Simple

# 1. Clone & setup

```python
# Drop-in replacement for OpenAI API - 10x faster with full control!
import openai
# 2. Install & run
client = openai.OpenAI(
    api_key="dummy-key",  # Not used, but required
    base_url="http://localhost:9000/v1"
# 3. Health check

# 1️⃣ Basic compatibility (existing code works unchanged!)
response = client.embeddings.create(
Visit: Swagger UI → http://localhost:9000/docs • ReDoc → /redoc
    model="text-embedding-ada-002"
)

## 🌐 API Modes (Choose Your Interface)
response = client.embeddings.create(
| Mode | Embeddings | Rerank | Typical Use |
|------|------------|--------|-------------|
| Native | `/api/v1/embed/` | `/api/v1/rerank/` | New projects / full control |
| OpenAI Compatible | `/v1/embeddings` | (use native/TEI rerank) | Existing OpenAI SDK code |
| TEI Compatible | `/embed` | `/rerank` | Replace Hugging Face TEI |
    model="text-embedding-ada-002",
### OpenAI (Drop‑in)
        "return_timing": True          # ⏱️ Get performance metrics
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9000/v1", api_key="dummy")
resp = client.embeddings.create(
    input=["Hello world", "Apple Silicon is fast!"],
    model="text-embedding-ada-002"
)
print(len(resp.data), 'embeddings')
```
)
### TEI Compatible
if hasattr(response.usage, 'mlx_processing_time'):
```bash
curl -X POST http://localhost:9000/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Hello world"], "truncate": true}'

### Rerank Documents  
### Native (Full Control)
curl -X POST "http://localhost:9000/api/v1/rerank/" \
  -H "Content-Type: application/json" \
```bash
# List models (OpenAI format)
curl http://localhost:9000/v1/models

curl -X POST "http://localhost:9000/v1/embeddings" \
  -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "passages": ["AI is cool", "Dogs are pets", "MLX is fast"]}'
  -H "Content-Type: application/json" \
  -d '{
    "input": ["High performance text", "Apple MLX acceleration"],
    "model": "text-embedding-ada-002",
## 🔧 Enhanced OpenAI Parameters

Add optional MLX tuning fields directly in request body (OpenAI embeddings call):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| batch_size | int | 32 | 1–128 batching for throughput |
| normalize | bool | false | L2 normalize embeddings |
| backend_preference | str | auto | mlx | torch | auto |
| device_preference | str | auto | mps | cpu | auto |
| max_tokens_per_text | int | 512 | Truncate safeguard |
| return_timing | bool | false | Include performance metrics |

Example:
```python
resp = client.embeddings.create(
  input=texts,
  model="text-embedding-ada-002",
  extra_body={
    "batch_size": 64,
    "normalize": True,
    "backend_preference": "mlx",
    "return_timing": True
  }
)
```

Sample enhanced response (abridged):
```json
{
  "data": [...],
  "usage": {
    "prompt_tokens": 42,
    "total_tokens": 42,
    "mlx_processing_time": 0.0045,
    "total_processing_time": 0.0123,
    "backend_used": "MLXBackend",
    "device_used": "mps",
    "batch_size_used": 64
  }
}
```

---

## 🔗 Endpoint Reference

| Endpoint | Method | Mode | Purpose |
|----------|--------|------|---------|
| `/api/v1/embed/` | POST | Native | Embeddings |
| `/api/v1/rerank/` | POST | Native | Reranking |
| `/v1/embeddings` | POST | OpenAI | OpenAI-compatible embeddings |
| `/v1/models` | GET | OpenAI | List models |
| `/embed` | POST | TEI | TEI embeddings |
| `/rerank` | POST | TEI | TEI reranking |
| `/info` | GET | TEI | Service info |
| `/health/` | GET | System | Health & metrics |
| `/docs` | GET | System | Swagger UI |
| `/redoc` | GET | System | ReDoc docs |

---

## 🛠 Configuration (.env)
    "normalize": true,
```env
BACKEND=auto                                   # auto | mlx | torch
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ
HOST=0.0.0.0
PORT=9000
BATCH_SIZE=32
MAX_TEXTS_PER_REQUEST=100
LOG_LEVEL=INFO
LOG_FORMAT=json
```
    "return_timing": true
Runtime precedence: request params > headers > env defaults.
  -H "Content-Type: application/json" \
---
```
## ✨ Key Features

| Category | Highlights |
|----------|------------|
| Performance | Sub‑ms embedding, ms-level reranking |
| Efficiency | 4‑bit quantized model, unified memory |
| Flexibility | Native + OpenAI + TEI in one binary |
| Resilience | Automatic backend fallback (MLX → MPS → CPU) |
| Observability | Structured logging + health metrics |
| Simplicity | Single model for both tasks |
| Privacy | 100% local – no data egress |
  -H "Content-Type: application/json" \
---
# TEI reranking (10x faster than standard TEI)
## 📊 Benchmarks (Illustrative)

| Backend | Load (cached) | Throughput (texts/s) | Single Batch Latency | Memory (usage %) |
|---------|---------------|----------------------|----------------------|------------------|
| MLX | 0.36 s | 30,411 | 0.78 ms | 13.4% |
| PyTorch (MPS) | 2.71 s | 569 | 35 ms | 15.2% |

> First run downloads model (one‑time). Subsequent starts use cached weights.
    "query": "What is Apple MLX?",
    "texts": [
      "Apple MLX is a machine learning framework for Apple Silicon",
## 🧪 Testing & Benchmark
      "MLX delivers incredible performance on Apple hardware"
```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app

# Quick backend smoke
python - <<'PY'
import asyncio
from app.backends.factory import BackendFactory
async def main():
  b = BackendFactory.create_backend('auto')
  await b.initialize()
  r = await b.embed_texts(['Hello Apple MLX!'])
  print('✅', len(r.vectors), 'embeddings in', f"{r.processing_time:.4f}s")
asyncio.run(main())
PY

# Performance micro benchmark
python -m app.utils.benchmark
```
curl http://localhost:9000/info
```

## � Error Handling

| Code | Meaning | Typical Cause |
|------|---------|---------------|
| 200 | Success | – |
| 400 | Bad request | Invalid parameter range (e.g. batch_size) |
| 422 | Validation error | Malformed JSON / schema mismatch |
| 503 | Backend not ready | Model still loading |
| 500 | Internal error | Unexpected failure |

Example:
```json
{ "error": "api_error", "detail": "batch_size must be between 1 and 128", "status_code": 400 }
```
- **� TEI Compatible**: Drop-in replacement for Hugging Face Text Embeddings Inference
- **�🔧 Configurable Performance**: Control batch sizes, backends, devices, and normalization
- **📊 Detailed Metrics**: Optional performance monitoring and timing information
## 🏗 Architecture
- **🚀 Apple MLX Native**: Built specifically for Apple Silicon's unified memory architecture
```
app/
├── main.py              # FastAPI bootstrap
├── backends/            # MLX + PyTorch abstraction
├── services/            # Business logic (embedding, rerank)
├── routers/             # API route definitions
└── utils/               # Benchmark, device detection, logging
```
- **📊 Production Ready**: Health checks, monitoring, structured logging
## 🤝 Contributing

1. Fork
2. Branch: `feat/your-feature`
3. Tests: `pytest -v`
4. Ensure style & lint (black/ruff if configured)
5. PR with concise description & performance notes (if applicable)

---

## 🌟 Community & References
- MLX Community: https://huggingface.co/mlx-community
- Example models:
  - Qwen3-Embedding-0.6B / 4B / 8B (various quantizations)

> Star ⭐ the repo if this helped you build faster on Apple Silicon.

---

## � License
MIT — see [LICENSE](LICENSE).

---

<div align="center"><strong>🚀 Built for Apple Silicon • Powered by MLX • Production Ready</strong><br/>Native • OpenAI • TEI — One Service.</div>

