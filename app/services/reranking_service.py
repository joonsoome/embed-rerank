"""
Reranking service for query-document relevance scoring.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.backends.base import BackendManager
from app.models.requests import RerankRequest
from app.models.responses import RerankResponse, RerankResult

logger = logging.getLogger(__name__)


class RerankingService:
    """Service for handling reranking operations with backend abstraction."""

    def __init__(self, backend_manager: BackendManager):
        """
        Initialize the reranking service.

        Args:
            backend_manager: Backend management instance
        """
        self.backend_manager = backend_manager
        self._request_counter = 0

    async def rerank_passages(self, request: RerankRequest, request_id: Optional[str] = None) -> RerankResponse:
        """
        Rerank passages based on relevance to the query.

        Args:
            request: Reranking request with query, passages, and parameters
            request_id: Optional request identifier for tracking

        Returns:
            RerankResponse with ranked passages and metadata

        Raises:
            ValueError: If backend is not available or inputs are invalid
            RuntimeError: If reranking fails
        """
        start_time = time.time()
        self._request_counter += 1

        if request_id is None:
            request_id = f"rerank_{self._request_counter}_{int(time.time())}"

        logger.info(f"Processing rerank request {request_id} with query and {len(request.passages)} passages")

        try:
            # Validate backend availability
            if not self.backend_manager.is_available():
                raise ValueError("No backend available for reranking")

            # Get reranking scores from backend
            scores = await self._compute_relevance_scores(query=request.query, passages=request.passages)

            # Create ranked results
            ranked_results = self._create_ranked_results(
                query=request.query,
                passages=request.passages,
                scores=scores,
                top_k=request.top_k,
                return_documents=request.return_documents,
            )

            # Process results
            processing_time = time.time() - start_time
            backend_info = self.backend_manager.get_current_backend_info()

            # Create response
            response = RerankResponse(
                results=ranked_results,
                query=request.query,
                backend=backend_info.get("name", "unknown"),
                device=backend_info.get("device", "unknown"),
                method=backend_info.get("rerank_method", "cross-encoder"),
                processing_time=processing_time,
                model_info=backend_info.get("rerank_model_name", "unknown"),
                usage={
                    "total_passages": len(request.passages),
                    "returned_passages": len(ranked_results),
                    "processing_time_ms": processing_time * 1000,
                    "backend": backend_info.get("name", "unknown"),
                    "top_k": request.top_k,
                    "query_length": len(request.query),
                },
                timestamp=datetime.now(),
                num_passages=len(request.passages),  # Add this field for test compatibility
            )

            logger.info(f"Successfully processed rerank request {request_id} in {processing_time:.3f}s")
            return response

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to rerank passages: {str(e)}"
            logger.error(f"Rerank request {request_id} failed after {processing_time:.3f}s: {error_msg}")

            raise RuntimeError(error_msg) from e

    async def _compute_relevance_scores(self, query: str, passages: List[str]) -> List[float]:
        """
        Compute relevance scores for query-passage pairs.

        Args:
            query: Query text
            passages: List of passage texts

        Returns:
            List of relevance scores
        """
        # Get current backend
        backend = self.backend_manager.get_current_backend()
        if backend is None:
            raise ValueError("No backend available")

        # Check if backend supports reranking
        if not hasattr(backend, 'rerank_passages'):
            # Fallback to embedding-based similarity
            logger.warning("Backend doesn't support reranking, falling back to embedding similarity")
            return await self._compute_embedding_similarity(query, passages)

        # Use backend's reranking method
        scores = await backend.rerank_passages(query=query, passages=passages)
        return scores

    async def _compute_embedding_similarity(self, query: str, passages: List[str]) -> List[float]:
        """
        Compute similarity scores using embeddings.

        Args:
            query: Query text
            passages: List of passage texts

        Returns:
            List of similarity scores
        """
        backend = self.backend_manager.get_current_backend()

        # Generate embeddings
        query_result = await backend.embed_texts([query])
        passage_result = await backend.embed_texts(passages)

        # Get vectors and normalize them
        import numpy as np

        query_vector = np.array(query_result.vectors[0])
        query_vector = query_vector / np.linalg.norm(query_vector)

        passage_vectors = np.array(passage_result.vectors)
        passage_norms = np.linalg.norm(passage_vectors, axis=1, keepdims=True)
        passage_norms[passage_norms == 0] = 1  # Avoid division by zero
        passage_vectors = passage_vectors / passage_norms

        # Compute cosine similarities
        scores = np.dot(passage_vectors, query_vector).tolist()

        return scores

    def _create_ranked_results(
        self, query: str, passages: List[str], scores: List[float], top_k: int, return_documents: bool = True
    ) -> List[RerankResult]:
        """
        Create ranked results from scores.

        Args:
            query: Original query
            passages: Original passages
            scores: Relevance scores
            top_k: Number of top results to return
            return_documents: Whether to include document text

        Returns:
            List of ranked results
        """
        # Create (score, index, passage) tuples
        scored_passages = [(score, idx, passage) for idx, (score, passage) in enumerate(zip(scores, passages))]

        # Sort by score (descending)
        scored_passages.sort(key=lambda x: x[0], reverse=True)

        # Take top_k results
        top_results = scored_passages[:top_k]

        # Create RerankResult objects
        results = []
        for score, original_idx, passage in top_results:
            if return_documents:
                result = RerankResult(text=passage, score=float(score), index=original_idx)
            else:
                result = RerankResult(score=float(score), index=original_idx)
            results.append(result)

        return results

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information and status.

        Returns:
            Dictionary with service metadata
        """
        backend_info = self.backend_manager.get_current_backend_info()

        return {
            "service": "RerankingService",
            "version": "1.0.0",
            "backend": backend_info,
            "requests_processed": self._request_counter,
            "available": self.backend_manager.is_available(),
            "supported_operations": ["rerank_passages"],
            "max_passages": 1000,
            "max_query_length": 2048,
            "max_passage_length": 4096,
            "fallback_method": "embedding_similarity",
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the service.

        Returns:
            Health status information
        """
        try:
            # Test with simple reranking
            test_query = "Health check test query"
            test_passages = [
                "This is a test passage for health checking",
                "Another test passage with different content",
            ]
            test_request = RerankRequest(query=test_query, passages=test_passages, top_k=2)

            start_time = time.time()
            result = await self.rerank_passages(test_request, request_id="health_check")
            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "backend_available": True,
                "test_results_count": len(result.results),
                "service_info": self.get_service_info(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend_available": self.backend_manager.is_available(),
                "service_info": self.get_service_info(),
            }

    async def batch_rerank(self, queries: List[str], passages: List[str], top_k: int = 10) -> List[RerankResponse]:
        """
        Batch reranking for multiple queries against the same set of passages.

        Args:
            queries: List of query texts
            passages: List of passage texts (shared across all queries)
            top_k: Number of top results per query

        Returns:
            List of rerank responses, one per query
        """
        results = []

        for i, query in enumerate(queries):
            request = RerankRequest(query=query, passages=passages, top_k=top_k)

            response = await self.rerank_passages(request, request_id=f"batch_rerank_{i}_{int(time.time())}")
            results.append(response)

        return results
