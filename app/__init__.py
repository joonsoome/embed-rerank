"""
Embed-Rerank API: Text embedding and document reranking service.

This package provides FastAPI-based REST endpoints for:
- Text embedding generation using Qwen3-Embedding-4B
- Document reranking for information retrieval
- Apple Silicon MLX optimization with PyTorch fallback
- Multi-API compatibility: Native, OpenAI, TEI, and Cohere formats

🚀 NEW in v1.2.0: Cohere API compatibility!
Now supports Cohere /v1/rerank and /v2/rerank endpoints alongside existing APIs.

Author: joonsoo-me
"""

__version__ = "1.2.0"
__author__ = "joonsoo-me"
