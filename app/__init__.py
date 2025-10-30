"""
Embed-Rerank API: Text embedding and document reranking service.

This package provides FastAPI-based REST endpoints for:
- Text embedding generation using Qwen3-Embedding-4B
- Document reranking for information retrieval
- Apple Silicon MLX optimization with PyTorch fallback
- Multi-API compatibility: Native, OpenAI, TEI, and Cohere formats

🚀 NEW in v1.2.3: OpenAI base64 encoding support + docs update
- Fixed Cohere API tests with proper environment variable handling
- Resolved pytest environment variable propagation issues
- Eliminated false warnings while maintaining 100% API compatibility
- Enhanced test suite reliability and consistency
- All API formats (Native, OpenAI, TEI, Cohere) now show clean success status

Author: joonsoo-me
"""

__version__ = "1.2.3"
__author__ = "joonsoo-me"
