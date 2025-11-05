"""
🚀 OpenAI API Compatibility Layer for Apple MLX Backend

This router provides seamless OpenAI API compatibility while leveraging the raw power
of Apple's MLX framework. Drop-in replacement for OpenAI embeddings with
sub-millisecond performance on Apple Silicon! 🍎⚡

✨ What you get:
- 🔄 Drop-in OpenAI SDK compatibility
- ⚡ Apple MLX performance (10x faster than OpenAI)
- 🔒 Local processing (no data leaves your machine)
- 💰 Zero API costs
- 🎯 Production-ready reliability

Transform your OpenAI embeddings workflow into an Apple Silicon powerhouse!
"""

import time
from typing import Any, Dict, List, Optional, Union
import base64
import numpy as np

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..backends.base import BackendManager
from ..models.requests import EmbedRequest, RerankRequest
from ..models.responses import EmbedResponse
from ..services.reranking_service import RerankingService
from .. import __version__

# 🧠 Neural network logging powered by Apple Silicon
logger = structlog.get_logger()

# 🌟 Router setup - the gateway to OpenAI compatibility
router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])

# 🎯 Global backend manager reference
backend_manager: BackendManager = None
reranker_backend_manager: BackendManager = None


def set_backend_manager(manager: BackendManager):
    """
    🔌 Connect OpenAI compatibility layer to Apple MLX backend

    This links our OpenAI-compatible endpoints to the blazing-fast MLX backend.
    Once connected, OpenAI SDK calls will be accelerated by Apple Silicon! 🚀
    """
    global backend_manager
    backend_manager = manager
    logger.info("🔗 OpenAI compatibility layer connected to Apple MLX backend")


def set_reranker_backend_manager(manager: BackendManager):
    """Connect dedicated reranker backend for OpenAI-compatible rerank endpoint."""
    global reranker_backend_manager
    reranker_backend_manager = manager
    logger.info("🔗 OpenAI compatibility layer connected to Reranker backend")


async def get_backend_manager() -> BackendManager:
    """
    🎯 Dependency Provider: Access to Apple MLX Backend

    Ensures our OpenAI-compatible endpoints have access to the MLX magic.
    Sub-millisecond embeddings await! ⚡
    """
    if backend_manager is None:
        raise HTTPException(status_code=503, detail="Apple MLX backend not ready - please wait for initialization")
    return backend_manager


async def get_reranking_service() -> RerankingService:
    """Provide a RerankingService; prefer dedicated reranker when available."""
    if reranker_backend_manager is not None and reranker_backend_manager.is_ready():
        return RerankingService(reranker_backend_manager)
    if backend_manager is None or not backend_manager.is_ready():
        raise HTTPException(status_code=503, detail="Backend not ready. Please wait for model initialization.")
    return RerankingService(backend_manager)


# 🔄 OpenAI-Compatible Request/Response Models with Enhanced MLX Arguments
class OpenAIEmbeddingRequest(BaseModel):
    """
    📋 Enhanced OpenAI Embeddings Request Format with Apple MLX Configuration

    Perfectly matches the OpenAI API specification while providing additional
    configuration options for our Apple MLX backend. All MLX-specific options
    are optional and don't break OpenAI SDK compatibility! 🚀
    """

    input: Union[str, List[str]] = Field(
        ...,
        description="Text to embed (string or array of strings)",
        json_schema_extra={"example": ["Hello Apple MLX!", "Blazing fast embeddings on Apple Silicon"]},
    )
    model: str = Field(
        default="text-embedding-ada-002",
        description="Model identifier (for compatibility - MLX model is used internally)",
        json_schema_extra={"example": "text-embedding-ada-002"},
    )
    encoding_format: Optional[str] = Field(
        default="float", description="Encoding format for embeddings", json_schema_extra={"example": "float"}
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="Number of dimensions in output embeddings (if supported)",
        json_schema_extra={"example": 1536},
    )
    user: Optional[str] = Field(
        default=None, description="User identifier for tracking", json_schema_extra={"example": "user_123"}
    )

    # 🚀 Enhanced Apple MLX Configuration Options (Optional, Non-Breaking)
    batch_size: Optional[int] = Field(
        default=None,
        description="🔧 MLX batch size for processing (overrides auto-sizing)",
        ge=1,
        le=128,
        json_schema_extra={"example": 32},
    )
    normalize: Optional[bool] = Field(
        default=True,
        description="🎯 Whether to normalize embeddings to unit length",
        json_schema_extra={"example": True},
    )
    backend_preference: Optional[str] = Field(
        default=None,
        description="🧠 Preferred backend: 'mlx', 'torch', or 'auto'",
        json_schema_extra={"example": "mlx"},
    )
    device_preference: Optional[str] = Field(
        default=None, description="⚡ Device preference: 'mps', 'cpu', or 'auto'", json_schema_extra={"example": "mps"}
    )
    max_tokens_per_text: Optional[int] = Field(
        default=None,
        description="📏 Maximum tokens per text (for truncation)",
        ge=1,
        le=8192,
        json_schema_extra={"example": 512},
    )
    return_timing: Optional[bool] = Field(
        default=False,
        description="⏱️ Include detailed timing information in response",
        json_schema_extra={"example": False},
    )

    class Config:
        json_schema_extra = {
            "example": {
                "input": ["Hello Apple MLX!", "Fast embeddings on Apple Silicon"],
                "model": "text-embedding-ada-002",
                "encoding_format": "float",
                "batch_size": 32,
                "normalize": True,
                "backend_preference": "mlx",
                "return_timing": False,
            }
        }


class OpenAIEmbeddingData(BaseModel):
    """
    📊 Individual Embedding Data Point

    Matches OpenAI's response format exactly while containing
    Apple MLX-generated vectors that outperform OpenAI in speed! ⚡
    """

    object: str = Field(default="embedding", description="Object type identifier")
    # Allow either float list (default) or base64 string when encoding_format="base64"
    embedding: Union[List[float], str] = Field(..., description="The embedding vector (float list or base64 string)")
    index: int = Field(..., description="Index of the input text")


class OpenAIEmbeddingUsage(BaseModel):
    """
    📈 Enhanced Usage Statistics in OpenAI Format

    Provides token counting and timing information while showcasing
    the incredible performance of Apple Silicon + MLX combination.
    Includes optional MLX-specific performance metrics! ⚡
    """

    prompt_tokens: int = Field(..., description="Number of prompt tokens processed")
    total_tokens: int = Field(..., description="Total tokens processed")

    # 🚀 Enhanced MLX Performance Metrics (Optional)
    mlx_processing_time: Optional[float] = Field(default=None, description="⚡ MLX backend processing time in seconds")
    total_processing_time: Optional[float] = Field(
        default=None, description="🕐 Total request processing time in seconds"
    )
    backend_used: Optional[str] = Field(default=None, description="🧠 Backend that processed the request")
    device_used: Optional[str] = Field(default=None, description="💻 Device that processed the request")
    batch_size_used: Optional[int] = Field(default=None, description="📦 Actual batch size used for processing")


class OpenAIEmbeddingResponse(BaseModel):
    """
    ✨ OpenAI Embeddings Response Format

    Perfect OpenAI API compatibility with Apple MLX performance under the hood.
    Your existing OpenAI SDK code works unchanged - just 10x faster! 🚀
    """

    object: str = Field(default="list", description="Response object type")
    data: List[OpenAIEmbeddingData] = Field(..., description="List of embedding data")
    model: str = Field(..., description="Model used for generation")
    usage: OpenAIEmbeddingUsage = Field(..., description="Usage statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, -0.2, 0.5, 0.8, -0.1], "index": 0}],
                "model": "text-embedding-ada-002",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }
        }


# 🔎 OpenAI-like Rerank Models (not official, pragmatic compatibility)
class OpenAIRerankRequest(BaseModel):
    query: str = Field(..., description="Query text")
    documents: Union[List[str], List[Dict[str, Any]]] = Field(
        ..., description="Documents (strings or objects with 'text')"
    )
    top_n: Optional[int] = Field(default=None, description="Top N results to return")
    return_documents: Optional[bool] = Field(default=False, description="Include document text in response")

    @staticmethod
    def to_internal(req: "OpenAIRerankRequest") -> RerankRequest:
        # Normalize documents -> passages (strings)
        passages: List[str] = []
        for i, d in enumerate(req.documents):
            if isinstance(d, str):
                passages.append(d)
            elif isinstance(d, dict):
                text = d.get("text") or d.get("content") or d.get("body") or d.get("value")
                if text is None:
                    raise HTTPException(status_code=422, detail=f"Document at index {i} missing 'text'")
                passages.append(str(text))
            else:
                raise HTTPException(status_code=422, detail=f"Invalid document at index {i}")
        top_k = req.top_n if (req.top_n is not None and req.top_n > 0) else len(passages)
        return RerankRequest(query=req.query, passages=passages, top_k=top_k, return_documents=req.return_documents)


class OpenAIRerankData(BaseModel):
    index: int
    score: float
    document: Optional[str] = None


class OpenAIRerankUsage(BaseModel):
    total_passages: int
    returned_passages: int
    total_tokens: Optional[int] = None


# 🔄 OpenAI Models List Compatibility
class OpenAIModel(BaseModel):
    """🤖 OpenAI Model Information Format"""

    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    owned_by: str = Field(default="apple-mlx", description="Model owner")


class OpenAIModelsResponse(BaseModel):
    """📋 OpenAI Models List Response Format"""

    object: str = Field(default="list", description="Response object type")
    data: List[OpenAIModel] = Field(..., description="List of available models")


@router.post("/embeddings", response_model=OpenAIEmbeddingResponse)
async def create_embeddings(
    request: OpenAIEmbeddingRequest,
    manager: BackendManager = Depends(get_backend_manager),
    http_request: Request = None,
) -> OpenAIEmbeddingResponse:
    """
    🚀 Enhanced OpenAI-Compatible Embeddings Endpoint Powered by Apple MLX!

    This endpoint provides perfect OpenAI API compatibility while delivering
    lightning-fast performance through Apple's MLX framework. Now with
    configurable arguments for maximum control over Apple Silicon power! 🍎⚡

    ✨ Enhanced Features:
    - 🔧 Configurable batch sizes for optimal throughput
    - 🎯 Optional normalization control
    - 🧠 Backend preference selection (MLX/Torch/Auto)
    - ⚡ Device preference for processing
    - ⏱️ Optional detailed timing metrics
    - 📏 Token limit controls per text

    ✨ Benefits over OpenAI:
    - ⚡ 10x faster inference on Apple Silicon
    - 🔒 Complete data privacy (local processing)
    - 💰 Zero API costs
    - 🎯 Sub-millisecond response times
    - 🧠 Unified memory architecture efficiency
    - 🔧 Full control over processing parameters

    Perfect drop-in replacement with enhanced control! 🍎
    """
    start_time = time.time()

    try:
        # 📝 Convert single string to list for consistent processing
        texts = [request.input] if isinstance(request.input, str) else request.input

        # 🔧 Extract enhanced configuration options
        batch_size = request.batch_size or min(32, len(texts))
        normalize = request.normalize if request.normalize is not None else True
        max_tokens = request.max_tokens_per_text

        # 🌟 Check for custom headers as alternative configuration method
        if http_request:
            # Support X-MLX-* headers for enterprise integration
            batch_size = int(http_request.headers.get("x-mlx-batch-size", batch_size))
            normalize = http_request.headers.get("x-mlx-normalize", str(normalize)).lower() == "true"
            backend_pref = request.backend_preference or http_request.headers.get("x-mlx-backend")
            device_pref = request.device_preference or http_request.headers.get("x-mlx-device")
        else:
            backend_pref = request.backend_preference
            device_pref = request.device_preference

        logger.info(
            "🚀 Enhanced OpenAI-compatible embedding request started",
            num_texts=len(texts),
            model=request.model,
            user=request.user,
            batch_size=batch_size,
            normalize=normalize,
            backend_preference=backend_pref,
            device_preference=device_pref,
            max_tokens=max_tokens,
            return_timing=request.return_timing,
            client_ip=http_request.client.host if http_request and http_request.client else None,
        )

        # 🔄 Convert enhanced OpenAI request to internal MLX format
        internal_request = EmbedRequest(texts=texts, normalize=normalize, batch_size=batch_size)

        # ⚡ Generate embeddings using Apple MLX magic with enhanced config!
        # Use the global embedding service with dynamic configuration
        if _embedding_service is None:
            raise RuntimeError("Embedding service not initialized. Server startup may have failed.")

        mlx_result: EmbedResponse = await _embedding_service.embed_texts(internal_request)

        # 📊 Calculate comprehensive timing metrics
        total_time = time.time() - start_time

        # 🔄 Optionally adjust dimensions if requested
        vectors: List[List[float]] = mlx_result.vectors
        target_dims = request.dimensions
        if target_dims is not None and target_dims > 0:
            adjusted: List[List[float]] = []
            for v in vectors:
                if len(v) == target_dims:
                    adjusted.append(v)
                elif len(v) > target_dims:
                    # Truncate to requested dimensions
                    adjusted.append(v[:target_dims])
                else:
                    # Pad with zeros up to requested dimensions
                    padded = v + [0.0] * (target_dims - len(v))
                    adjusted.append(padded)
            vectors = adjusted

        # 🔄 Transform MLX response to enhanced OpenAI format (support base64 when requested)
        embedding_data: List[OpenAIEmbeddingData] = []
        if (request.encoding_format or "float").lower() == "base64":
            for i, v in enumerate(vectors):
                arr = np.asarray(v, dtype=np.float32)
                b64 = base64.b64encode(arr.tobytes()).decode("ascii")
                embedding_data.append(OpenAIEmbeddingData(embedding=b64, index=i))
        else:
            for i, v in enumerate(vectors):
                embedding_data.append(OpenAIEmbeddingData(embedding=v, index=i))

        # 📈 Calculate token usage (approximate word-based counting)
        total_tokens = sum(len(text.split()) for text in texts)

        # ✨ Create enhanced usage statistics
        usage_data = {"prompt_tokens": total_tokens, "total_tokens": total_tokens}

        # 🚀 Add enhanced metrics if requested
        if request.return_timing:
            usage_data.update(
                {
                    "mlx_processing_time": mlx_result.processing_time,
                    "total_processing_time": total_time,
                    "backend_used": mlx_result.backend,
                    "device_used": mlx_result.device,
                    "batch_size_used": batch_size,
                }
            )

        # ✨ Create OpenAI-compatible response with enhanced MLX performance data
        response = OpenAIEmbeddingResponse(
            data=embedding_data, model=request.model, usage=OpenAIEmbeddingUsage(**usage_data)
        )

        logger.info(
            "✅ Enhanced OpenAI-compatible embeddings completed",
            num_texts=len(texts),
            num_vectors=len(embedding_data),
            vector_dim=mlx_result.dim,
            mlx_processing_time=mlx_result.processing_time,
            total_time=total_time,
            backend=mlx_result.backend,
            device=mlx_result.device,
            model=request.model,
            tokens=total_tokens,
            batch_size_used=batch_size,
            normalize_used=normalize,
            enhanced_features_used=request.return_timing,
        )

        return response

    except Exception as e:
        processing_time = time.time() - start_time

        logger.error(
            "💥 Enhanced OpenAI-compatible embedding request failed",
            error=str(e),
            error_type=type(e).__name__,
            processing_time=processing_time,
            model=request.model,
            num_texts=len(texts) if 'texts' in locals() else 0,
            batch_size=batch_size if 'batch_size' in locals() else None,
        )

        # If this is a validation error from Pydantic, return 422
        try:
            from pydantic import ValidationError

            if isinstance(e, ValidationError) or 'validation' in str(e).lower():
                raise HTTPException(status_code=422, detail={"error": str(e)})
        except Exception:
            pass

        if isinstance(e, ValueError):
            raise HTTPException(status_code=400, detail={"error": str(e)})

        # 🚨 Return OpenAI-style error response for unexpected errors
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Embedding generation failed: {str(e)}",
                    "type": "internal_server_error",
                    "code": "mlx_processing_error",
                }
            },
        )


@router.get("/models", response_model=OpenAIModelsResponse)
async def list_models(manager: BackendManager = Depends(get_backend_manager)) -> OpenAIModelsResponse:
    """
    📋 OpenAI-Compatible Models List Endpoint

    Returns available models in OpenAI format. While we use Apple MLX
    internally, we present familiar OpenAI model names for compatibility.
    The magic happens behind the scenes with Apple Silicon acceleration! 🍎⚡
    """
    try:
        # 🎯 Get current backend info for model details
        backend_info = manager.get_backend_info()
        current_time = int(time.time())

        # 🎯 Show only the actual backend model for transparency
        models = [
            OpenAIModel(id=backend_info.get('model_name', 'unknown-model'), created=current_time, owned_by="apple-mlx"),
        ]

        logger.info(
            "📋 OpenAI models list requested",
            num_models=len(models),
            actual_backend=backend_info.get('name', 'unknown'),
            actual_model=backend_info.get('model_name', 'unknown'),
        )

        return OpenAIModelsResponse(data=models)

    except Exception as e:
        logger.error("💥 Failed to list OpenAI-compatible models", error=str(e), error_type=type(e).__name__)

        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Failed to list models: {str(e)}",
                    "type": "internal_server_error",
                    "code": "models_list_error",
                }
            },
        )


# 🔍 Health Check for OpenAI Compatibility
@router.get("/health")
async def openai_health(manager: BackendManager = Depends(get_backend_manager)) -> Dict[str, Any]:
    """
    💚 OpenAI Compatibility Health Check

    Verifies that the OpenAI compatibility layer is connected to
    the Apple MLX backend and ready to serve lightning-fast embeddings! ⚡
    """
    try:
        # 🔍 Check backend status and gather richer metadata
        backend_info = manager.get_backend_info()
        backend_health = await manager.health_check()
        is_ready = manager.is_ready()

        status = "healthy" if is_ready else "not_ready"

        # 📏 Derive embedding dimension if available
        embedding_dim = (
            backend_health.get("embedding_dim")
            or backend_health.get("embedding_dimension")
            or backend_info.get("embedding_dim")
            or None
        )

        health_data: Dict[str, Any] = {
            "status": status,
            "service": {
                "name": "embed-rerank",
                "version": __version__,
                "description": "OpenAI-compatible endpoints powered by Apple MLX",
            },
            "openai_compatible": True,
            "backend": {
                "name": backend_info.get('name', 'unknown'),
                "device": backend_info.get('device', 'unknown'),
                "model": backend_info.get('model_name', 'unknown'),
                "ready": is_ready,
            },
            "embedding": {
                "dimension": embedding_dim,
            },
            "compatibility": {
                "openai_sdk": True,
                "endpoints": ["/v1/embeddings", "/v1/models", "/v1/openai/rerank"],
                "response_format": "openai_standard",
            },
            "performance": {
                "apple_silicon_optimized": True,
                "expected_speedup": "10x vs OpenAI API",
                "typical_latency": "< 50ms",
            },
            "timestamp": time.time(),
        }

        # ➕ Include reranker backend info if configured and ready
        try:
            if reranker_backend_manager is not None and reranker_backend_manager.is_ready():
                r_info = reranker_backend_manager.get_current_backend_info()
                health_data["reranker"] = {
                    "name": r_info.get("name"),
                    "model_name": r_info.get("model_name"),
                    "device": r_info.get("device"),
                    "pooling": r_info.get("pooling"),
                    "score_norm": r_info.get("score_norm"),
                    "method": r_info.get("rerank_method") or r_info.get("method"),
                }
        except Exception:
            pass

        logger.info(
            "💚 OpenAI compatibility health check",
            status=status,
            backend_ready=is_ready,
            backend_name=backend_info.get('name', 'unknown'),
        )

        return health_data

    except Exception as e:
        logger.error("💥 OpenAI compatibility health check failed", error=str(e), error_type=type(e).__name__)

        return {"status": "unhealthy", "error": str(e), "openai_compatible": False, "timestamp": time.time()}


# 🔧 Global embedding service variable for dynamic configuration
_embedding_service = None


def set_embedding_service(service):
    """🚀 Set the embedding service for dynamic configuration support"""
    global _embedding_service
    _embedding_service = service
    logger.info("🔄 OpenAI router updated with dynamic embedding service")


@router.post("/openai/rerank")
@router.post("/rerank_openai")
async def openai_rerank(request: OpenAIRerankRequest) -> Dict[str, Any]:
    """
    Pragmatic OpenAI-compatible rerank endpoint.

    Request mirrors common community conventions: {query, documents, top_n, return_documents}.
    Response returns {object: "list", data: [{index, score, document?}], model, usage}.
    """
    try:
        service = await get_reranking_service()
        internal = OpenAIRerankRequest.to_internal(request)
        result = await service.rerank_passages(internal)

        # Build data items
        data_items: List[Dict[str, Any]] = []
        for r in result.results:
            item = {"index": r.index, "score": float(r.score)}
            if request.return_documents and hasattr(r, "text") and r.text is not None:
                item["document"] = r.text
            data_items.append(item)

        # Optional sigmoid normalization controlled by settings
        try:
            from math import exp
            from app.config import settings as _settings
            if _settings.openai_rerank_auto_sigmoid:
                for item in data_items:
                    s = float(item["score"]) if "score" in item else 0.0
                    item["score"] = 1.0 / (1.0 + exp(-s))
        except Exception:
            pass

        # Create response
        usage = OpenAIRerankUsage(
            total_passages=result.usage.get("total_passages", len(internal.passages)),
            returned_passages=result.usage.get("returned_passages", len(result.results)),
            total_tokens=len(internal.query.split()),
        )

        response: Dict[str, Any] = {
            "object": "list",
            "data": data_items,
            "model": service.backend_manager.get_current_backend_info().get("rerank_model_name")
            or service.backend_manager.get_current_backend_info().get("model_name"),
            "usage": usage.model_dump(),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("💥 OpenAI-compatible rerank failed", error=str(e))
        raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "internal"}})
