"""
Embedding router for text embedding operations.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List

from ..models.requests import EmbedRequest
from ..models.responses import EmbedResponse, ErrorResponse
from ..services.embedding_service import EmbeddingService
from ..backends.base import BackendManager

router = APIRouter(
    prefix="/api/v1/embed",
    tags=["embedding"],
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"}
    }
)

# This will be set by the main app
_backend_manager: BackendManager = None


def set_backend_manager(manager: BackendManager):
    """Set the backend manager instance."""
    global _backend_manager
    _backend_manager = manager


async def get_backend_manager() -> BackendManager:
    """Dependency to get the backend manager."""
    if _backend_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Backend manager not initialized"
        )
    return _backend_manager


async def get_embedding_service(
    manager: BackendManager = Depends(get_backend_manager)
) -> EmbeddingService:
    """Dependency to get the embedding service."""
    if not manager.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Backend not ready. Please wait for model initialization."
        )
    return EmbeddingService(manager)


@router.post("/", response_model=EmbedResponse)
async def generate_embeddings(
    request: EmbedRequest,
    service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Generate embeddings for the provided texts.
    
    Args:
        request: EmbedRequest containing texts and optional parameters
        service: EmbeddingService dependency
    
    Returns:
        EmbedResponse with generated embeddings and metadata
    
    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Generate embeddings using the service
        response = await service.embed_texts(request)
        
        return response
    
    except ValueError as e:
        # Input validation errors
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    
    except RuntimeError as e:
        # Backend/model errors
        raise HTTPException(
            status_code=503,
            detail=f"Service error: {str(e)}"
        )
    
    except Exception as e:
        # Unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/info")
async def get_embedding_info(
    service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Get information about the embedding service and model.
    
    Returns:
        Dictionary with model information, capabilities, and status
    """
    try:
        info = await service.get_service_info()
        return info
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get service info: {str(e)}"
        )
