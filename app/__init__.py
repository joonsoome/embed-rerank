"""
Embed-Rerank API: Text embedding and document reranking service.

This package provides FastAPI-based REST endpoints for:
- Text embedding generation using Qwen3-Embedding-4B
- Document reranking for information retrieval
- Apple Silicon MLX optimization with PyTorch fallback
- Multi-API compatibility: Native, OpenAI, TEI, and Cohere formats

🚀 NEW in v1.2.1: Enhanced test suite reliability and Cohere API support!
- Improved virtual environment detection in test scripts
- Fixed text processing test result parsing (100% success rate display)
- Enhanced server URL display (localhost instead of 0.0.0.0)
- Added comprehensive Cohere API compatibility
- Updated documentation and troubleshooting guides

Author: joonsoo-me
"""

__version__ = "1.2.1"
__author__ = "joonsoo-me"
