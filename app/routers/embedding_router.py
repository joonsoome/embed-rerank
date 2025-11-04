"""
🚀 Apple MLX Embedding Router: Text to Vectors at Light Speed

This router transforms your text into high-dimensional embeddings using the power
of Apple Silicon and MLX. Every request is a demonstration of what's possible
when cutting-edge AI meets Apple's unified memory architecture.

⚡ Performance Highlights:
- Sub-millisecond text embedding generation
- Dynamic vector dimension (auto-detected from model)
- Batch processing with MLX acceleration
- Zero-copy operations on Apple Silicon

Join the Apple MLX community in revolutionizing on-device AI!
"""

from fastapi import APIRouter, Depends, HTTPException

from ..backends.base import BackendManager
from ..models.requests import EmbedRequest
from ..models.responses import EmbedResponse, ErrorResponse
from ..services.embedding_service import EmbeddingService

# 🎯 Apple MLX Embedding Router Configuration
router = APIRouter(
    prefix="/api/v1/embed",
    tags=["🧠 Apple MLX Embeddings"],
    responses={
        503: {"model": ErrorResponse, "description": "Apple MLX Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Invalid Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
    },
)

# 🌟 Global services - our connection to Apple Silicon magic
_backend_manager: BackendManager = None
_embedding_service: EmbeddingService = None


def set_backend_manager(manager: BackendManager):
    """🔌 Connect to the Apple MLX Backend Manager"""
    global _backend_manager
    _backend_manager = manager


def set_embedding_service(service: EmbeddingService):
    """🚀 Set the configured embedding service with dynamic metadata"""
    global _embedding_service
    _embedding_service = service


async def get_backend_manager() -> BackendManager:
    """🎯 Dependency: Access to Apple MLX Backend Power"""
    if _backend_manager is None:
        raise HTTPException(status_code=503, detail="Apple MLX backend not initialized - please wait")
    return _backend_manager


async def get_embedding_service(manager: BackendManager = Depends(get_backend_manager)) -> EmbeddingService:
    """
    🧠 Embedding Service Dependency: Your Gateway to Apple Silicon AI

    This dependency ensures our MLX backend is ready and provides access
    to the embedding service that orchestrates the text-to-vector magic.
    Now with dynamic model metadata configuration! 🚀
    """
    if not manager.is_ready():
        raise HTTPException(
            status_code=503, detail="Apple MLX backend warming up - please wait for model initialization"
        )

    # 🎯 Use globally configured service if available (with dynamic config)
    if _embedding_service is not None:
        return _embedding_service

    # 🔄 Fallback to basic service (legacy mode)
    return EmbeddingService(manager)


@router.post("/", response_model=EmbedResponse)
async def generate_embeddings(request: EmbedRequest, service: EmbeddingService = Depends(get_embedding_service)):
    """
    🚀 Generate Text Embeddings: Apple Silicon AI in Action

    Transform your text into high-dimensional semantic vectors using Apple MLX!
    This endpoint showcases the incredible speed of Apple Silicon unified memory
    architecture with sub-millisecond embedding generation.

    ✨ What happens here:
    - Text tokenization optimized for Apple Silicon
    - MLX-accelerated model inference through unified memory
    - Dynamic vector dimension auto-detected from model
    - Automatic normalization for cosine similarity

    🎯 Perfect for:
    - Semantic search applications
    - Document similarity analysis
    - RAG (Retrieval Augmented Generation) systems
    - Real-time content recommendations

    try:
        #  generate embeddings using Apple MLX magic
        response = await service.embed_texts(request)

        # Convert to dict and add backward-compatible fields expected by tests
        resp = response.model_dump()
        resp["num_texts"] = len(response.embeddings)
        return resp

    except ValueError as e:
    Example:
        ```json
        {
            "texts": ["Apple Silicon is incredible", "MLX makes AI fast"],
            "batch_size": 32,
            "normalize": true
        }
        ```
    """
    try:
        # 🧠 Generate embeddings using Apple MLX magic
        response = await service.embed_texts(request)

        # Convert to dict and add backward-compatible fields expected by tests
        response_dict = response.model_dump()
        response_dict["num_texts"] = len(response.embeddings)

        return response_dict

    except ValueError as e:
        # 📝 Input validation errors - help users get it right
        raise HTTPException(status_code=400, detail=f"Invalid input for Apple MLX embedding: {str(e)}")

    except RuntimeError as e:
        # ⚠️ Backend/model errors - MLX or system issues
        raise HTTPException(status_code=503, detail=f"Apple MLX service error: {str(e)}")

    except Exception as e:
        # 💥 Unexpected errors - something went really wrong
        raise HTTPException(status_code=500, detail=f"Unexpected Apple MLX error: {str(e)}")


@router.get("/info")
async def get_embedding_info(service: EmbeddingService = Depends(get_embedding_service)):
    """
    📊 Apple MLX Embedding Service Information

    Get detailed information about our Apple Silicon-powered embedding service.
    Perfect for monitoring performance, checking model details, and understanding
    the incredible capabilities of MLX on Apple Silicon.

    Returns comprehensive metrics including:
    - Model information and architecture details
    - Apple Silicon performance characteristics
    - MLX framework version and capabilities
    - Current service status and health

    Returns:
        Dictionary with Apple MLX service information and performance metrics
    """
    try:
        info = service.get_service_info()

        # 🚀 Add Apple MLX branding to the response
        info["powered_by"] = "Apple MLX Framework"
        info["optimized_for"] = "Apple Silicon"
        info["community"] = "Apple MLX Community"

        return info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Apple MLX service info: {str(e)}")
