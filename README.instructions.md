# Embed-Rerank API Implementation Guide

## Overview

A production-ready FastAPI service for text embeddings and document reranking, optimized for Apple Silicon with MLX acceleration and PyTorch fallback support.

## Architecture

### Core Design
- **Single Model Loading**: One model serves both embedding and reranking endpoints
- **Backend Abstraction**: Automatic selection between MLX (Apple Silicon) and PyTorch
- **Production Ready**: Health checks, monitoring, structured logging, error handling

### Technology Stack
- **Framework**: FastAPI + Uvicorn
- **ML Backends**: MLX (Apple Silicon), PyTorch (MPS/CPU fallback)
- **Model**: mlx-community/Qwen3-Embedding-4B-4bit-DWQ
- **Validation**: Pydantic v2

## Project Structure

```
app/
├── main.py              # FastAPI application
├── config.py            # Configuration
├── models/              # Pydantic schemas
│   ├── requests.py      # Request models
│   └── responses.py     # Response models
├── backends/            # ML backends
│   ├── base.py          # Abstract interface
│   ├── factory.py       # Backend selection
│   ├── mlx_backend.py   # MLX implementation
│   └── torch_backend.py # PyTorch implementation
├── services/            # Business logic
│   ├── embedding_service.py
│   └── reranking_service.py
├── routers/             # API routes
│   ├── embedding_router.py
│   ├── reranking_router.py
│   └── health_router.py
└── utils/               # Utilities
    ├── device.py        # Device detection
    ├── logger.py        # Logging setup
    └── benchmark.py     # Performance testing
tests/                   # Test suite
```

## Dependencies

**Core:**
- fastapi, uvicorn, pydantic, torch, sentence-transformers
- structlog, httpx, python-dotenv

**MLX (Apple Silicon only):**
- mlx>=0.4.0, mlx-lm>=0.2.0

## Configuration

Environment variables in `.env`:

```env
# Backend
BACKEND=auto                    # auto, mlx, torch
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ

# Server
HOST=0.0.0.0
PORT=9000

# Performance
BATCH_SIZE=32
MAX_TEXTS_PER_REQUEST=100

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## API Endpoints

- `GET /health/` - System health and performance metrics
- `POST /api/v1/embed/` - Generate text embeddings
- `POST /api/v1/rerank/` - Rerank documents by relevance
- `GET /docs/` - Interactive API documentation

## Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn app.main:app --host 0.0.0.0 --port 9000

# Test
curl http://localhost:9000/health/
```

## Key Classes

- **BaseBackend**: Abstract interface for ML backends
- **BackendFactory**: Automatic backend selection logic  
- **EmbeddingService**: Text embedding business logic
- **RerankingService**: Document reranking logic
- **BackendManager**: Backend lifecycle management

## Implementation Guidelines

1. Follow FastAPI best practices with dependency injection
2. Use Pydantic v2 for all request/response validation
3. Implement comprehensive error handling and logging
4. Ensure Apple Silicon optimization with MLX backend
5. Maintain backward compatibility with PyTorch fallback
6. Include health checks and monitoring capabilities
