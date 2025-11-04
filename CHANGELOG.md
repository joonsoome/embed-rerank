# Changelog

All notable changes to the Apple MLX Embed-Rerank API project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-09-10
 
## [1.2.3] - 2025-10-30

### Added
- OpenAI compatibility: base64 encoding support via `encoding_format="base64"` for `/v1/embeddings`.
- OpenAI compatibility: optional `dimensions` handling (truncate/pad to requested size).

### Documentation
- README: Added LightRAG integration note (OpenAI embeddings + Cohere reranking tested successfully).
- README: Added Qwen Embedding similarity scaling note and recommended starting threshold `COSINE_THRESHOLD=0.0` for LightRAG.
- README: Example for requesting base64-encoded embeddings and decoding back to float32.

### Notes
- These updates maintain full compatibility with existing OpenAI SDK usage; default remains `encoding_format="float"`.


## [1.3.0] - 2025-11-04

### Added
- 📏 Dynamic embedding dimension documentation in README, including supported config keys and best practices for vector DBs
- 🧩 Optional fixed output-dimension controls (disabled by default)
  - Env: `OUTPUT_EMBEDDING_DIMENSION`, `DIMENSION_STRATEGY` (pad|trim)
  - OpenAI-compatible `dimensions` request field mapped to per-request trim when no global override is set

### Fixed
- 🛠 MLX backend embedding dimension alignment
  - Properly reads model config dimension using multiple keys (`hidden_size`, `d_model`, `embedding_size`, `model_dim`, `dim`)
  - Placeholder/fallback path now respects config dict values (no unintended 4096 defaulting)

### Changed
- 🧹 Removed hardcoded dimension assumptions from docs; router docstrings updated to reflect dynamic dimensions
- 📚 README updated with “Dynamic Embedding Dimensions” section and optional compatibility knobs

### Added
- 🆕 **Cohere API v1/v2 Compatibility**: Full support for Cohere reranking API
  - `/v1/rerank` endpoint (legacy format support)
  - `/v2/rerank` endpoint (modern format with document objects)
  - Complete request/response format compatibility
  - Drop-in replacement for Cohere API calls
- 🌐 **Four-API Multi-Compatibility**: Now supports Native, OpenAI, TEI, and Cohere APIs simultaneously
- 📋 **Enhanced API Documentation**: Updated README with all four API usage examples
- 🔧 **Troubleshooting Guide**: Comprehensive problem resolution documentation

### Fixed
- ✅ **Critical Fix**: "Embedding service not initialized" error in OpenAI and TEI routers
  - Root cause: Missing `set_embedding_service()` calls in main.py lifespan function
  - Impact: OpenAI API tests (10/18 failing) and TEI API tests (4/23 failing) 
  - Resolution: Automatic embedding service initialization during server startup
  - All API compatibility tests now pass: Native (18/18), OpenAI (18/18), TEI (23/23), Cohere (4/4)

### Changed
- 🔄 **Improved Service Initialization**: Enhanced startup sequence ensures all API routers have proper service access
- 📊 **Updated Test Suite**: Comprehensive API compatibility validation with `--api-compatibility` option
- 🎯 **Enhanced Health Checks**: All API endpoints now properly report service availability

### Technical Details
**Problem Resolution:**
```python
# Added to app/main.py lifespan function:
from .services.embedding_service import EmbeddingService
embedding_service = EmbeddingService(backend_manager)

# Critical fix - these lines were missing:
openai_router.set_embedding_service(embedding_service)
tei_router.set_embedding_service(embedding_service)
```

**Test Results After Fix:**
- ✅ Native API: 18/18 tests passed
- ✅ OpenAI API: 18/18 tests passed (was 8/18)
- ✅ TEI API: 23/23 tests passed (was 19/23)  
- ✅ Cohere API: 4/4 tests passed (new)
- ✅ Full compatibility suite: All tests passing

## [1.1.0] - 2025-09-08

### Added
- 🌐 **TEI (Text Embedding Inference) Compatibility**: Hugging Face TEI drop-in replacement
  - `/embed` endpoint with TEI-compatible request/response format
  - `/rerank` endpoint for document reranking
  - `/info` endpoint for model information
  - Complete compatibility with existing TEI clients

### Fixed
- 🔧 **Performance Improvements**: Optimized MLX backend initialization
- 📊 **Enhanced Logging**: Better structured logging with performance metrics

### Changed
- 🎯 **Multi-API Architecture**: Refactored to support multiple API standards simultaneously
- 📋 **Documentation Updates**: Added TEI usage examples and compatibility notes

## [1.0.0] - 2025-09-05

### Added
- 🚀 **Initial Release**: Apple MLX-powered embedding and reranking service
- ⚡ **Apple Silicon Optimization**: Native MLX backend with sub-millisecond inference
- 🔄 **PyTorch Fallback**: Automatic backend selection (MLX → PyTorch MPS → CPU)
- 🌐 **OpenAI API Compatibility**: Drop-in replacement for OpenAI embeddings API
  - `/v1/embeddings` endpoint with full OpenAI SDK compatibility
  - `/v1/models` endpoint for model listing
  - Enhanced MLX-specific configuration options
- 🎯 **Native API**: High-performance native endpoints
  - `/api/v1/embed` for text embedding generation
  - `/api/v1/rerank` for document reranking
- 📊 **Production Features**:
  - Health checks and monitoring endpoints
  - Structured logging with performance metrics
  - CORS and security middleware
  - Error handling and graceful degradation
- 🧠 **Smart Text Processing**:
  - Auto-truncation for long texts
  - Dynamic token limit detection from model metadata
  - Intelligent text summarization
- 📦 **Easy Deployment**:
  - PyPI package installation (`pip install embed-rerank`)
  - CLI interface with configuration options
  - Built-in performance testing and benchmarking

### Performance Benchmarks
- **Embeddings**: 0.78ms avg (10x faster than OpenAI API)
- **Reranking**: 1.04ms avg (25x faster than typical solutions)
- **Model Loading**: 0.36s (9x faster than alternatives)
- **Cost**: $0 (vs $0.02/1K tokens for OpenAI)

### Supported Models
- Primary: `mlx-community/Qwen3-Embedding-4B-4bit-DWQ`
- Architecture: 4-bit quantized for optimal Apple Silicon performance
- Dimensions: 1024 (auto-detected)
- Context Length: 512 tokens (auto-detected)

---

## Migration Guide

### From 1.1.0 to 1.2.0
- **No breaking changes** - all existing API calls continue to work
- **New feature**: Cohere API compatibility available immediately
- **Improved reliability**: Previous API errors automatically resolved

### From 1.0.0 to 1.1.0  
- **No breaking changes** - all OpenAI compatibility maintained
- **New feature**: TEI compatibility available as additional endpoints

## Upgrading

### PyPI Package Users
```bash
pip install --upgrade embed-rerank
embed-rerank
```

### Source Code Users
```bash
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
./tools/server-run.sh
```

## API Compatibility Matrix

| Version | Native API | OpenAI API | TEI API | Cohere API |
|---------|------------|------------|---------|------------|
| 1.2.0   | ✅ Full    | ✅ Full    | ✅ Full | ✅ Full    |
| 1.1.0   | ✅ Full    | ✅ Full    | ✅ Full | ❌ None    |
| 1.0.0   | ✅ Full    | ✅ Full    | ❌ None | ❌ None    |

## Links
- **GitHub Repository**: https://github.com/joonsoo-me/embed-rerank
- **PyPI Package**: https://pypi.org/project/embed-rerank/
- **Documentation**: https://github.com/joonsoo-me/embed-rerank#readme
- **Issues & Support**: https://github.com/joonsoo-me/embed-rerank/issues
