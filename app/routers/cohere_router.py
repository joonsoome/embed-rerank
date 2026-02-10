"""
Cohere-compatible reranking router.

🚀 Apple MLX Community Innovation: Extending MLX performance to Cohere API format!
Bringing sub-millisecond reranking to Cohere-compatible applications.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..backends.base import BackendManager
from ..config import settings
from ..models.cohere_models import CohereRerankRequest, CohereRerankResponse, CohereRerankResult, CohereDocument
from ..models.requests import RerankRequest
from ..models.responses import ErrorResponse
from ..services.reranking_service import RerankingService
from ..utils.rerank_chunking import aggregate_scores, chunk_documents


router = APIRouter(
    tags=["cohere-compatibility"],
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
    },
)


def _filter_none_values(data):
    """
    Recursively filter out keys with None values.

    This ensures fields like `document: null` are omitted entirely
    when `return_documents` is False, matching Cohere API behavior.
    """
    if isinstance(data, dict):
        return {k: _filter_none_values(v) for k, v in data.items() if v is not None}
    if isinstance(data, list):
        return [_filter_none_values(item) for item in data]
    return data


# This will be set by the main app
_backend_manager: BackendManager = None
_reranker_backend_manager: BackendManager = None


def set_backend_manager(manager: BackendManager):
    """Set the backend manager instance."""
    global _backend_manager
    _backend_manager = manager


def set_reranker_backend_manager(manager: BackendManager):
    """Set a dedicated reranker backend manager (cross-encoder) for Cohere-compatible endpoints."""
    global _reranker_backend_manager
    _reranker_backend_manager = manager


async def get_backend_manager() -> BackendManager:
    """Dependency to get the backend manager."""
    if _backend_manager is None:
        raise HTTPException(status_code=503, detail="Backend manager not initialized")
    return _backend_manager


async def get_reranking_service(manager: BackendManager = Depends(get_backend_manager)) -> RerankingService:
    """Dependency to get the reranking service."""
    # Prefer dedicated reranker backend if configured and ready
    if _reranker_backend_manager is not None and _reranker_backend_manager.is_ready():
        return RerankingService(_reranker_backend_manager)
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
        return_documents=cohere_request.return_documents or False,
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
        cohere_result = CohereRerankResult(index=result.index, relevance_score=result.score)

        # Add document if requested
        if cohere_request.return_documents:
            cohere_result.document = CohereDocument(text=result.text or "")

        results.append(cohere_result)

    return CohereRerankResponse(results=results)


@router.post("/v1/rerank", response_model=CohereRerankResponse)
async def rerank_v1(request: CohereRerankRequest, service: RerankingService = Depends(get_reranking_service)):
    """
    Cohere-compatible reranking endpoint (v1).

    🎯 MLX-Powered Cohere Compatibility: Experience the lightning-fast
    performance of Apple MLX through the familiar Cohere API interface!

    Perfect for developers migrating from Cohere to self-hosted solutions
    while gaining significant performance improvements on Apple Silicon.
    """
    try:
        # Optional document chunking (LightRAG parity) for models with strict token limits.
        if settings.rerank_enable_chunking:
            original_docs = request.documents
            chunked_docs, doc_indices = chunk_documents(
                original_docs,
                max_tokens=int(settings.rerank_max_tokens_per_doc),
            )

            # Guardrail: never exceed the server's configured passage limit.
            if len(chunked_docs) > settings.max_passages_per_rerank:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Chunking produced {len(chunked_docs)} chunks which exceeds "
                        f"MAX_PASSAGES_PER_RERANK={settings.max_passages_per_rerank}. "
                        f"Reduce RERANK_MAX_TOKENS_PER_DOC or increase MAX_PASSAGES_PER_RERANK."
                    ),
                )

            scores = await service.compute_relevance_scores(query=request.query, passages=chunked_docs)
            aggregated = aggregate_scores(
                scores=scores,
                doc_indices=doc_indices,
                num_docs=len(original_docs),
                aggregation="max",
            )

            # Apply top_n at document level (post-aggregation).
            top_n = request.top_n if request.top_n is not None else len(original_docs)
            top_n = min(int(top_n), len(aggregated))

            results: List[CohereRerankResult] = []
            for doc_idx, score in aggregated[:top_n]:
                item = CohereRerankResult(index=int(doc_idx), relevance_score=float(score))
                if request.return_documents:
                    item.document = CohereDocument(text=original_docs[int(doc_idx)])
                results.append(item)

            cohere_response = CohereRerankResponse(results=results)
            payload = _filter_none_values(cohere_response.model_dump())
            return JSONResponse(content=payload)

        # Convert Cohere request to internal format
        internal_request = convert_to_internal_request(request)

        # 🚀 Apple MLX Magic: Process with our high-performance backend!
        internal_response = await service.rerank_passages(internal_request)

        # Convert back to Cohere format
        cohere_response = convert_to_cohere_response(internal_response, request)

        # Ensure None fields (e.g., document when return_documents=False) are omitted
        payload = _filter_none_values(cohere_response.model_dump())

        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/v2/rerank", response_model=CohereRerankResponse)
async def rerank_v2(request: CohereRerankRequest, service: RerankingService = Depends(get_reranking_service)):
    """
    Cohere-compatible reranking endpoint (v2).

    🚀 Next-Gen MLX Performance: Same blazing-fast Apple Silicon optimization,
    now available through Cohere v2 API format for maximum compatibility!

    Features the latest optimizations while maintaining full Cohere API compliance.
    """
    # v2 uses the same logic as v1 for now
    # In the future, this could support additional v2-specific features
    return await rerank_v1(request, service)
