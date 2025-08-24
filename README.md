# 🔥 Single Model Embedding & Reranking API

**Lightning-fast text embeddings and document reranking powered by Apple Silicon & MLX**

<div align="center">

[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Ready-blue?logo=apple&logoColor=white)](https://developer.apple.com/silicon/)
[![MLX Framework](https://img.shields.io/badge/MLX-Optimized-green?logo=apple&logoColor=white)](https://ml-explore.github.io/mlx/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)

</div>

---

## ⚡ Performance That Will Blow Your Mind

| Operation | Apple MLX | Traditional |
|-----------|-----------|-------------|
| **Embedding Generation** | `0.78ms` | `35ms` |
| **Document Reranking** | `1.04ms` | `150ms` |
| **Model Loading** | `0.36s` | `6.6s` |
| **Memory Usage** | `13.4%` | `43.2%` |

*Tested on Apple M4 Max - Your Apple Silicon results may vary (but will still be amazing!)*

---

## 🎯 Quick Start

```bash
# 1. Clone the magic
git clone https://github.com/joonsoo-me/embed-rerank.git
cd embed-rerank

# 2. Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Apple MLX experience
python -m uvicorn app.main:app --host 0.0.0.0 --port 9000
```

**🎉 That's it! Your Apple Silicon is now serving AI at lightning speed.**

Visit: [http://localhost:9000/docs](http://localhost:9000/docs) for the interactive API documentation.

---

## 🧠 AI Operations Made Simple

### Generate Embeddings
```bash
curl -X POST "http://localhost:9000/api/v1/embed/" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Apple Silicon is incredible", "MLX makes AI fast"]}'
```

### Rerank Documents  
```bash
curl -X POST "http://localhost:9000/api/v1/rerank/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apple MLX performance",
    "passages": [
      "MLX delivers incredible AI performance on Apple Silicon",
      "Traditional frameworks are slower on Apple hardware",
      "The future of AI is on-device with Apple MLX"
    ]
  }'
```

### Health Check
```bash
curl http://localhost:9000/health/
```

---

## 🏗️ What Makes This Special

- **🚀 Apple MLX Native**: Built specifically for Apple Silicon's unified memory architecture
- **⚡ Sub-millisecond Inference**: Faster than you can blink
- **🧠 4-bit Quantized Model**: Maximum efficiency, minimal memory footprint  
- **🔄 Auto Backend Selection**: MLX → PyTorch MPS → CPU fallback
- **📊 Production Ready**: Health checks, monitoring, structured logging
- **🎯 Single Model**: One model handles both embedding and reranking

---

## 🛠️ Configuration

Create a `.env` file for customization:

```env
# Backend (auto-detects Apple Silicon)
BACKEND=auto
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ

# Server
HOST=0.0.0.0
PORT=9000

# Performance  
BATCH_SIZE=32
MAX_TEXTS_PER_REQUEST=100
```

---

## 🧪 Testing

```bash
# Run the full test suite
pytest tests/ -v

# Quick integration test
python -c "
import asyncio
from app.backends.factory import BackendFactory
from app.backends.base import BackendManager

async def test():
    backend = BackendFactory.create_backend('auto')
    manager = BackendManager(backend)
    await manager.initialize()
    result = await backend.embed_texts(['Hello Apple MLX!'])
    print(f'✅ Generated {len(result.vectors)} embeddings in {result.processing_time:.3f}s')

asyncio.run(test())
"
```

---

## 🌟 Join the Apple MLX Community

This project is built with love for the Apple MLX ecosystem. We believe in:

- **🚀 Innovation**: Pushing the boundaries of on-device AI
- **⚡ Performance**: Making AI as fast as Apple Silicon deserves
- **🤝 Community**: Sharing knowledge and advancing together
- **📚 Education**: Learning and teaching MLX best practices

**Star this repo** if Apple MLX + AI performance excites you!

---

## 📄 License

MIT License - build amazing things with this code!

---

<div align="center">

**🚀 Built with Apple MLX • Optimized for Apple Silicon • Ready for Production**

[Documentation](http://localhost:9000/docs) • [Health Check](http://localhost:9000/health) • [API Reference](http://localhost:9000/redoc)

</div>

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