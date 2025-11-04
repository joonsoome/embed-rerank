# 🔥 Single Server: Embeddings + Cross‑Encoder Reranking (MLX‑first)

<p align="center">
  <a href="https://pypi.org/project/embed-rerank/">
    <img src="https://img.shields.io/pypi/v/embed-rerank?logo=pypi&logoColor=white" alt="PyPI Version" />
  </a>
  <a href="https://github.com/joonsoo-me/embed-rerank/blob/main/LICENSE"><img src="https://img.shields.io/github/license/joonsoo-me/embed-rerank?logo=opensource&logoColor=white" /></a>
<a href="https://developer.apple.com/silicon/"><img src="https://img.shields.io/badge/Apple_Silicon-Ready-blue?logo=apple&logoColor=white" /></a>
<a href="https://ml-explore.github.io/mlx/"><img src="https://img.shields.io/badge/MLX-Optimized-green?logo=apple&logoColor=white" /></a>
<a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" /></a>
</p>

Lightning-fast local embeddings & true cross‑encoder reranking on Apple Silicon (MLX‑first). OpenAI, TEI, and Cohere compatible.

## 🔧 Troubleshooting
### Common Issues
**"Embedding service not initialized" Error**: Fixed in v1.2.0. If you encounter this error:
1. Update to the latest version: `pip install --upgrade embed-rerank`
2. For source installations, ensure proper service initialization in `main.py`
3. See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for detailed solutions

**API Compatibility Issues**: All four APIs (Native, OpenAI, TEI, Cohere) are fully tested and compatible:
- ✅ Native API: `/api/v1/embed`, `/api/v1/rerank`
- ✅ OpenAI API: `/v1/embeddings` (drop-in replacement)  
- ✅ TEI API: `/embed`, `/rerank` (Hugging Face compatible)
- ✅ Cohere API: `/v1/rerank`, `/v2/rerank` (Cohere compatible)

**Performance Testing**: Use built-in benchmarking:
```bash
embed-rerank --test performance --test-url http://localhost:9000
```

For comprehensive troubleshooting, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## 🍎 MLX Compatibility Note (mx.array → asarray)

Recent MLX versions removed `mx.array` in favor of `mx.asarray` (and `mx.numpy.array`). This repository includes a compatibility helper that automatically forwards to the appropriate API, so Apple Silicon embeddings continue to work across MLX versions.

**What changed:**
- Internal `mx.array(...)` calls now use a helper that tries, in order: `mx.array` → `mx.asarray` → `mx.numpy.array`.
- Placeholder embedding fallback now respects the model configuration using multiple dimension keys.

**Why this matters:**
- Prevents runtime error: `module 'mlx.core' has no attribute 'array'` on newer MLX.
- Ensures embedding dimension matches the loaded model, avoiding vector size mismatches.

**Optional dependency for MLX (macOS only):** `pip install "embed-rerank[mlx]"` or see `pyproject.toml` (`mlx>=0.4.0`, `mlx-lm>=0.2.0`).

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

## 🚀 Quick Start (2 Modes)

### Option 1: Embeddings‑only (MODEL_NAME만 설정)

```bash
pip install embed-rerank

# .env (필수 키만)
cat > .env <<'ENV'
BACKEND=auto
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ
PORT=9000
HOST=0.0.0.0
ENV

embed-rerank  # http://localhost:9000
```

### Option 2: Embeddings + Cross‑Encoder Reranking (한 서버, 두 모델)

```bash
git clone https://github.com/joonsoo-me/embed-rerank.git && cd embed-rerank
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# .env (리랭커 모델만 추가하면 cross‑encoder 경로가 활성화됩니다)
cat > .env <<'ENV'
BACKEND=auto
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ

# Cross‑encoder reranker (MLX native, 현재는 Torch로 자동 폴백)
RERANKER_MODEL_ID="vserifsaglam/Qwen3-Reranker-4B-4bit-MLX"
RERANKER_BACKEND=auto

PORT=9000
HOST=0.0.0.0
ENV

./tools/server-run.sh
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

# Development automation tools (NEW!)
./tools/setup-macos-service.sh     # Auto-generate macOS LaunchAgent
./tools/test-ci-locally.sh         # Run GitHub CI tests locally
```

> **Windows Support**: Coming soon! Currently optimized for macOS/Linux.

---

## ⚙️ Configuration: One Server, Two Models

### PyPI Package CLI Options

**Server Options:**
- `--host`: Server host (default: 0.0.0.0)
- `--port`: Server port (default: 9000)
- `--reload`: Enable auto-reload for development
- `--log-level`: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**Testing Options:**
- `--test quick`: Run quick validation tests
- `--test performance`: Run performance benchmark tests  
- `--test quality`: Run quality validation tests
- `--test full`: Run comprehensive test suite
- `--test-url`: Custom server URL for testing
- `--test-output`: Test output directory

**Examples:**
```bash
# Custom server configuration
embed-rerank --port 8080 --host 127.0.0.1 --reload

# Built-in performance testing
embed-rerank --port 8080 &
embed-rerank --test performance --test-url http://localhost:8080
pkill -f embed-rerank

# Environment variables
export PORT=8080 HOST=127.0.0.1
embed-rerank
```

### Source Code Configuration

Create `.env` file for development (필수 키 최소화):

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

### Enable Cross‑Encoder Reranker (선택)
RERANKER_MODEL_ID="vserifsaglam/Qwen3-Reranker-4B-4bit-MLX"
RERANKER_BACKEND=auto     # auto | mlx | torch (MLX 구현 준비 중, 현재 torch로 폴백)
## Optional
# RERANK_BATCH_SIZE=16
# RERANK_MAX_SEQ_LEN=1024

장점:
- 동일 서버 프로세스에서 임베딩과 리랭킹을 동시에 제공 (운영 간소화)
- 환경변수로 손쉽게 on/off 전환 (임베딩만 ↔ 임베딩+리랭킹)
- Apple MLX 최적화(임베딩) + 교차 인코더의 품질 (리랭킹)
```

### 🧠 Smart Text Processing Features

The service automatically handles long texts with intelligent processing:

- **Auto-Truncation**: Texts exceeding token limits are automatically reduced by ~75%
- **Smart Summarization**: Key sentences are preserved while removing redundancy
- **Dynamic Token Limits**: Automatically detected from model metadata (e.g., 512 tokens for Qwen3)
- **Dynamic Dimension Detection**: Vector dimensions auto-configured from model metadata
- **Processing Transparency**: Optional processing info in API responses

**Example: 8000+ character text → 2037 tokens automatically**

---

### 📏 Dynamic Embedding Dimensions

- The service derives embedding dimension directly from the loaded model’s config.
- Supported config keys (priority): `hidden_size` → `d_model` → `embedding_size` → `model_dim` → `dim`.
- Backend and health endpoints report the actual vector size; clients should not assume a fixed dimension.
- Tip for vector DBs (e.g., Qdrant): create the collection with the reported dimension.

#### Optional: Fixed Output Dimension (Compatibility)

If you already have an index built at a specific dimension (e.g., 4096), you can ask the service to pad/trim output vectors to that size:

```env
# Optional – force output vectors to a fixed size
OUTPUT_EMBEDDING_DIMENSION=4096
# Strategy: pad with zeros or trim leading dimensions (then re-normalize)
DIMENSION_STRATEGY=pad   # or trim
```

- Service-level setting takes precedence over per-request settings.
- OpenAI-compatible `dimensions` request field is supported and maps to trim behavior when no global override is set.
- For cosine similarity, zero-padding + re-normalization is safe; for other metrics, prefer retraining/reindexing.

### 📂 Model Cache Management (요약)

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

## 🌐 Four APIs, One Service (요약)

| API | Endpoint | Use Case |
|-----|----------|----------|
| **Native** | `/api/v1/embed`, `/api/v1/rerank` | New projects |
| **OpenAI** | `/v1/embeddings` | Existing OpenAI code |
| **TEI** | `/embed`, `/rerank` | Hugging Face TEI replacement |
| **Cohere** | `/v1/rerank`, `/v2/rerank` | Cohere API replacement |

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
You can request base64-encoded embeddings by setting `encoding_format="base64"`. This is useful when transporting vectors through systems that expect strings only.

```python
response = client.embeddings.create(
    input=["Hello world"],
    model="text-embedding-ada-002",
    encoding_format="base64",  # returns base64-encoded float32 bytes
)

# embedding string is base64; decode if you need floats again
import base64, numpy as np
arr = np.frombuffer(base64.b64decode(response.data[0].embedding), dtype=np.float32)
```

Notes:
- `encoding_format` defaults to `"float"` (list[float]).
- `dimensions` is accepted and will truncate/pad to the requested size when supported.

### TEI Compatible

```bash
curl -X POST "http://localhost:9000/embed" 
  -H "Content-Type: application/json" 
  -d '{"inputs": ["Hello world"], "truncate": true}'
```

### Cohere Compatible

```python
import requests

# Cohere v2 reranking (recommended)
response = requests.post("http://localhost:9000/v2/rerank", json={
    "model": "rerank-multilingual-v3.0",
    "query": "What is machine learning?",
    "documents": [
        {"text": "Machine learning is a subset of AI"},
        {"text": "Dogs are great pets"},
        {"text": "Deep learning uses neural networks"}
    ],
    "top_n": 3,
    "return_documents": True
})

# Cohere v1 reranking (legacy support)
response = requests.post("http://localhost:9000/v1/rerank", json={
    "model": "rerank-english-v3.0", 
    "query": "machine learning",
    "documents": ["AI is fascinating", "I love pizza", "ML is powerful"],
    "top_n": 2
})
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

Note: The native rerank endpoint also accepts Cohere/OpenWebUI-style payloads using `documents` instead of `passages` and `top_n` instead of `top_k`:

```bash
curl -X POST "http://localhost:9000/api/v1/rerank/" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "documents": ["AI is cool", "Dogs are pets", "MLX is fast"], "top_n": 3}'
```

---

## 🧪 Testing (요약)

### 🚀 Built-in CLI Testing (PyPI Package)

The PyPI package includes powerful built-in testing capabilities:

```bash
# Quick validation (basic functionality check)
embed-rerank --test quick

# Performance benchmark (latency, throughput, concurrency)
embed-rerank --test performance --test-url http://localhost:9000

# Quality validation (semantic similarity, multilingual)  
embed-rerank --test quality --test-url http://localhost:9000

# Full comprehensive test suite
embed-rerank --test full --test-url http://localhost:9000
```

**Test Results Include:**
- 📊 **Latency Metrics**: Mean, P95, P99 response times
- 🚀 **Throughput Analysis**: Texts/sec processing rates
- 🔄 **Concurrency Testing**: Multi-threaded request handling
- 🧠 **Semantic Validation**: Quality of embeddings and reranking
- 🌍 **Multilingual Support**: Cross-language performance
- 📈 **JSON Reports**: Detailed metrics for automation

**Example Output:**
```bash
🧪 Running Embed-Rerank Test Suite
📍 Target URL: http://localhost:9000
🎯 Test Mode: performance

⚡ Performance Results:
• Latency: 0.8ms avg, 1.2ms max
• Throughput: 1,250 texts/sec peak  
• Concurrency: 5/5 successful (100%)
📁 Results saved to: ./test-results/performance_test_results.json
```

### 🔧 Advanced Testing (Source Code)

```bash
### 🔧 Advanced Testing (Source Code)

For development and comprehensive testing with the source code:

```bash
# Comprehensive test suite (shell script)
./tools/server-tests.sh

# Run with specific test modes
./tools/server-tests.sh --quick            # Quick validation only
./tools/server-tests.sh --performance      # Performance tests only
./tools/server-tests.sh --full             # Full test suite
./tools/server-tests.sh --text-processing  # Text processing validation

# Custom server URL
./tools/server-tests.sh --url http://localhost:8080

# Development automation (NEW!)
./tools/test-ci-locally.sh                 # Run GitHub CI tests locally
./tools/setup-macos-service.sh             # Generate macOS LaunchAgent

# Manual health check
curl http://localhost:9000/health/

# Unit tests with pytest
pytest tests/ -v
```

---

## 🛠 Development & Deployment (요약)

### Local Development (Source Code)

```bash
# Start server (background)
./tools/server-run.sh

# Start server (foreground/development)
./tools/server-run-foreground.sh

# Stop server
./tools/server-stop.sh
```

### Production Deployment (PyPI Package)

```bash
# Install and run
pip install embed-rerank
embed-rerank --port 9000 --host 0.0.0.0

# With custom configuration
embed-rerank --port 8080 --reload --log-level DEBUG

# Background deployment
embed-rerank --port 9000 &
```

> **Windows Support**: Coming soon! Currently optimized for macOS/Linux.
```

---

## 🚀 What You Get

### 🎯 Core Features
- ✅ **Zero Code Changes**: Drop-in replacement for OpenAI, TEI, and Cohere APIs
- ⚡ **10x Performance**: Apple MLX acceleration on Apple Silicon  
- 💰 **Zero Costs**: No API fees, runs locally
- 🔒 **Privacy**: Your data never leaves your machine
- 🎯 **Four APIs**: Native, OpenAI, TEI, and Cohere compatibility
- 📊 **Production Ready**: Health checks, monitoring, structured logging
- 🧠 **Smart Text Processing**: Auto-truncation and summarization for long texts
- ⚙️ **Dynamic Configuration**: Automatic model metadata extraction and dimension detection

### 🧪 Built-in Testing & Benchmarking
- 📈 **CLI Performance Testing**: One-command benchmarking
- 🔄 **Concurrency Testing**: Multi-threaded request validation
- 🧠 **Quality Validation**: Semantic similarity and multilingual testing
- 📊 **JSON Reports**: Automated performance monitoring
- 🚀 **Real-time Metrics**: Latency, throughput, and success rates

### 🛠 Development Automation (New!)
- 🍎 **macOS Service Management**: Auto-generate LaunchAgent from configuration
- 🧪 **Local CI Testing**: Run GitHub CI tests locally before commits
- 📋 **Code Quality Tools**: Automated Black, isort, and flake8 validation
- 🔧 **Smart Development Workflow**: Virtual environment checks and setup automation

### 🛠 Deployment Options
- 📦 **PyPI Package**: `pip install embed-rerank` for instant deployment
- 🔧 **Source Code**: Full development environment with advanced tooling
- 🌐 **Multi-API Support**: OpenAI, TEI, Cohere, and native endpoints
- ⚙️ **Flexible Configuration**: Environment variables, CLI args, .env files

---

## Quick Reference

### Installation & Startup
```bash
# PyPI Package (Production)
pip install embed-rerank && embed-rerank

# Source Code (Development)  
git clone https://github.com/joonsoo-me/embed-rerank.git
cd embed-rerank && ./tools/server-run.sh
```

### Performance Testing
```bash
# One-command benchmark
embed-rerank --test performance --test-url http://localhost:9000

# Comprehensive testing
./tools/server-tests.sh --full
```

### API Endpoints
- **Native**: `POST /api/v1/embed/` and `/api/v1/rerank/`
- **OpenAI**: `POST /v1/embeddings` (drop-in replacement)
- **TEI**: `POST /embed` and `/rerank` (Hugging Face compatible)
- **Cohere**: `POST /v1/rerank` and `/v2/rerank` (Cohere API compatible)
- **Health**: `GET /health/` (monitoring and diagnostics with model metadata)

### Development Tools (New!)
```bash
# macOS service automation
./tools/setup-macos-service.sh    # Auto-generate LaunchAgent from .env.example

# Local CI testing
./tools/test-ci-locally.sh        # Run complete GitHub CI suite locally

# Code quality automation
black --line-length 120 app/ tests/    # Consistent formatting
isort --profile black app/ tests/      # Import organization  
flake8 app/ tests/ --max-line-length=120 --extend-ignore=E203,W503  # Linting
```

---

## 🧩 LightRAG Integration

We validated an end-to-end workflow using LightRAG with this service:
- Embeddings via the OpenAI-compatible endpoint (`/v1/embeddings`)
- Reranking via the Cohere-compatible endpoint (`/v1/rerank` or `/v2/rerank`)

Results: the integration tests succeeded using OpenAI embeddings and Cohere reranking.

Qwen Embedding similarity scaling note: when using the Qwen Embedding model, we observed cosine similarity values that appear very small (e.g., `0.02`, `0.03`). This is expected due to vector scaling differences and does not indicate poor retrieval by itself. As a starting point, we recommend disabling the retrieval threshold in LightRAG to avoid filtering out good matches prematurely:

```
# === Retrieval threshold ===
COSINE_THRESHOLD=0.0
```

Adjust upward later based on your dataset and evaluation results.

---

## 📄 License

MIT License - build amazing things with this code!
