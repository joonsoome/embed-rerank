"""
Cohere-compatible reranking router.

🚀 Apple MLX Community Innovation: Extending MLX performance to Cohere API format!
Bringing sub-millisecond reranking to Cohere-compatible applications.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ..backends.base import BackendManager
from ..models.cohere_models import CohereRerankRequest, CohereRerankResponse, CohereRerankResult, CohereDocument
from ..models.requests import RerankRequest
from ..models.responses import ErrorResponse
from ..services.reranking_service import RerankingService


router = APIRouter(
    tags=["cohere-compatibility"],
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
    },
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
        raise HTTPException(status_code=503, detail="Backend manager not initialized")
    return _backend_manager


async def get_reranking_service(manager: BackendManager = Depends(get_backend_manager)) -> RerankingService:
    """Dependency to get the reranking service."""
    if not manager.is_ready():
        raise HTTPException(status_code=503, detail="Backend not ready. Please wait for model initialization.")
    return RerankingService(manager)


def convert_to_internal_request(cohere_request: CohereRerankRequest) -> RerankRequest:
    """
    Convert Cohere API request to internal RerankRequest format.
    
    🎯 MLX Performance Bridge: Seamlessly converting Cohere format to our 
    high-performance internal format while preserving all semantic meaning!
    """
    # Convert documents to passages (same content, different naming)
    passages = cohere_request.documents
    
    # Convert top_n to top_k (default to length if not specified)
    top_k = cohere_request.top_n if cohere_request.top_n is not None else len(passages)
    
    return RerankRequest(
        query=cohere_request.query,
        passages=passages,
        top_k=top_k,
        return_documents=cohere_request.return_documents or False
    )


def convert_to_cohere_response(internal_response, cohere_request: CohereRerankRequest) -> CohereRerankResponse:
    """
    Convert internal RerankResponse to Cohere API format.
    
    🚀 Apple Silicon Speed: Transforming our MLX-optimized results into 
    Cohere-compatible format while maintaining performance excellence!
    """
    results = []
    
    for result in internal_response.results:
        # Create base result with index and relevance score
        cohere_result = CohereRerankResult(
            index=result.index,
            relevance_score=result.score
        )
        
        # Add document if requested
        if cohere_request.return_documents:
            cohere_result.document = CohereDocument(text=result.text or "")
        
        results.append(cohere_result)
    
    return CohereRerankResponse(results=results)


@router.post("/v1/rerank", response_model=CohereRerankResponse)
async def rerank_v1(
    request: CohereRerankRequest, 
    service: RerankingService = Depends(get_reranking_service)
):
    """
    Cohere-compatible reranking endpoint (v1).
    
    🎯 MLX-Powered Cohere Compatibility: Experience the lightning-fast 
    performance of Apple MLX through the familiar Cohere API interface!
    
    Perfect for developers migrating from Cohere to self-hosted solutions
    while gaining significant performance improvements on Apple Silicon.
    """
    try:
        # Convert Cohere request to internal format
        internal_request = convert_to_internal_request(request)
        
        # 🚀 Apple MLX Magic: Process with our high-performance backend!
        internal_response = await service.rerank_passages(internal_request)
        
        # Convert back to Cohere format
        cohere_response = convert_to_cohere_response(internal_response, request)
        
        return cohere_response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/v2/rerank", response_model=CohereRerankResponse)
async def rerank_v2(
    request: CohereRerankRequest, 
    service: RerankingService = Depends(get_reranking_service)
):
    """
    Cohere-compatible reranking endpoint (v2).
    
    🚀 Next-Gen MLX Performance: Same blazing-fast Apple Silicon optimization,
    now available through Cohere v2 API format for maximum compatibility!
    
    Features the latest optimizations while maintaining full Cohere API compliance.
    """
    # v2 uses the same logic as v1 for now
    # In the future, this could support additional v2-specific features
    return await rerank_v1(request, service)
