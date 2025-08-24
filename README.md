# 🔥 Single Model Embedding & Reranker API

**Production-ready FastAPI service for text embeddings and document reranking, optimized for Apple Silicon.**

A high-performance, MLX-accelerated service achieving **sub-millisecond inference** on Apple Silicon with PyTorch fallback support.

---

## ✨ Key Features

- **⚡ 0.78ms** embedding generation (2 texts)
- **⚡ 1.29ms** document reranking (3 passages) 
- **🧠 320-dimensional** high-quality embeddings
- **🚀 Apple Silicon optimized** with MLX acceleration
- **🔄 Auto backend selection** (MLX → PyTorch MPS → CPU)
- **📊 Production monitoring** with health checks and metrics
- **📖 OpenAPI documentation** at `/docs`

---

## 📊 Performance Benchmarks

| Backend | Loading | Inference | Latency | Memory |
|---------|---------|-----------|---------|--------|
| **MLX** | 0.36s | 30,411 texts/sec | 0.78ms | 13.4% |
| **PyTorch** | 2.71s | 569 texts/sec | 35ms | 15.2% |

*Tested on Apple M4 Max with 128GB unified memory*

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/joonsoo-me/embed-rerank.git
cd embed-rerank

# Install dependencies  
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run server
python -m uvicorn app.main:app --host 0.0.0.0 --port 9000

# Test
curl http://localhost:9000/health/
```

**The server will automatically:**
1. Detect Apple Silicon and select MLX backend
2. Download the MLX model (first run: ~22s, cached: ~0.36s)
3. Start serving on `http://localhost:9000`

---

## 🔗 API Usage

### Generate Embeddings

```bash
curl -X POST "http://localhost:9000/api/v1/embed/" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "AI is amazing"]}'
```

### Rerank Documents

```bash
curl -X POST "http://localhost:9000/api/v1/rerank/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "passages": [
      "AI and machine learning are transforming industries",
      "Dogs are pets", 
      "Deep learning is a subset of machine learning"
    ]
  }'
```

### Health Check

```bash
curl http://localhost:9000/health/
```

**Interactive Documentation:** Visit `http://localhost:9000/docs`

---

## ⚙️ Configuration

Configure via `.env` file:

```env
# Backend Selection
BACKEND=auto                    # auto, mlx, torch
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ

# Server Settings
HOST=0.0.0.0
PORT=9000

# Performance
BATCH_SIZE=32
MAX_TEXTS_PER_REQUEST=100

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## 🏗️ Architecture

```
app/
├── main.py              # FastAPI application
├── backends/            # MLX + PyTorch backends
├── services/            # Business logic
├── routers/             # API endpoints
└── utils/               # Device detection, logging
```

**Key Features:**
- **Backend Abstraction**: Automatic MLX/PyTorch selection
- **Single Model**: One model serves both embedding and reranking
- **Production Ready**: Health checks, monitoring, error handling
- **Apple Silicon Optimized**: MLX for maximum performance

---

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app

# Performance benchmarks
python -m app.utils.benchmark
```

---

## 📚 Documentation

- **API Docs**: `http://localhost:9000/docs` (auto-generated)
- **Implementation Guide**: `README.instructions.md`
- **Development Status**: `DEVELOPMENT_PLAN.md`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests and formatting: `pytest tests/ && black app/`
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**🚀 Built for the AI community. Optimized for Apple Silicon, designed for production.**
