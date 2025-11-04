"""
Embed-Rerank API: Text embedding and document reranking service.

This package provides FastAPI-based REST endpoints for:
- Text embedding generation using Qwen3-Embedding-4B
- Document reranking for information retrieval
- Apple Silicon MLX optimization with PyTorch fallback
- Multi-API compatibility: Native, OpenAI, TEI, and Cohere formats

🚀 NEW in v1.3.0: Dynamic embedding dimensions and enhanced model configuration!
- Automatic embedding dimension detection from model config (hidden_size, d_model, etc.)
- Optional fixed output dimension controls (OUTPUT_EMBEDDING_DIMENSION, DIMENSION_STRATEGY)
- OpenAI-compatible 'dimensions' request field support for per-request dimension control
- OpenAI base64 encoding support (encoding_format="base64")
- MLX backend now properly reads config with multiple dimension key fallbacks
- Enhanced README documentation with dimension configuration best practices
- LightRAG integration guidance and Qwen similarity scaling notes

Author: joonsoo-me
"""

__version__ = "1.3.0"
__author__ = "joonsoo-me"
