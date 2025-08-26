"""
Embedding service for text vectorization operations.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.backends.base import BackendManager
from app.models.requests import EmbedRequest
from app.models.responses import EmbeddingVector, EmbedResponse

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for handling embedding operations with backend abstraction."""

    def __init__(self, backend_manager: BackendManager):
        """
        Initialize the embedding service.

        Args:
            backend_manager: Backend management instance
        """
        self.backend_manager = backend_manager
        self._request_counter = 0

    async def embed_texts(self, request: EmbedRequest, request_id: Optional[str] = None) -> EmbedResponse:
        """
        Generate embeddings for the provided texts.

        Args:
            request: Embedding request with texts and parameters
            request_id: Optional request identifier for tracking

        Returns:
            EmbedResponse with embeddings and metadata

        Raises:
            ValueError: If backend is not available or texts are invalid
            RuntimeError: If embedding generation fails
        """
        start_time = time.time()
        self._request_counter += 1

        if request_id is None:
            request_id = f"embed_{self._request_counter}_{int(time.time())}"

        logger.info(f"Processing embedding request {request_id} with {len(request.texts)} texts")

        try:
            # Validate backend availability
            if not self.backend_manager.is_available():
                raise ValueError("No backend available for embedding generation")

            # Get embeddings from backend
            embeddings_array = await self._generate_embeddings(
                texts=request.texts, batch_size=request.batch_size, normalize=request.normalize
            )

            # Process results
            processing_time = time.time() - start_time

            # Create embedding objects with metadata
            embeddings = []
            for i, (text, embedding) in enumerate(zip(request.texts, embeddings_array)):
                embeddings.append(
                    EmbeddingVector(
                        embedding=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                        index=i,
                        text=text if len(text) <= 100 else f"{text[:97]}...",  # Truncate for response
                    )
                )

            # Get backend info
            backend_info = self.backend_manager.get_current_backend_info()

            # Create response
            response = EmbedResponse(
                embeddings=embeddings,
                vectors=[emb.embedding for emb in embeddings],  # Legacy format
                dim=len(embeddings[0].embedding) if embeddings else 0,
                backend=backend_info.get("name", "unknown"),
                device=backend_info.get("device", "unknown"),
                processing_time=processing_time,
                model_info=backend_info.get("model_name", "unknown"),
                usage={
                    "total_texts": len(request.texts),
                    "total_tokens": sum(len(text.split()) for text in request.texts),
                    "processing_time_ms": processing_time * 1000,
                    "backend": backend_info.get("name", "unknown"),
                    "batch_size": request.batch_size,
                    "normalize": request.normalize,
                },
                timestamp=datetime.now(),
                num_texts=len(request.texts),  # Add this field for test compatibility
            )

            logger.info(f"Successfully processed embedding request {request_id} in {processing_time:.3f}s")
            return response

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to generate embeddings: {str(e)}"
            logger.error(f"Embedding request {request_id} failed after {processing_time:.3f}s: {error_msg}")

            raise RuntimeError(error_msg) from e

    async def _generate_embeddings(
        self, texts: List[str], batch_size: int, normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings using the current backend.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings

        Returns:
            List of embedding vectors
        """
        # Get current backend
        backend = self.backend_manager.get_current_backend()
        if backend is None:
            raise ValueError("No backend available")

        # Generate embeddings using backend
        result = await backend.embed_texts(texts=texts, batch_size=batch_size)

        # Extract vectors from result
        embeddings = result.vectors.tolist() if hasattr(result.vectors, 'tolist') else result.vectors

        # Apply normalization if requested
        if normalize:
            import numpy as np

            embeddings_array = np.array(embeddings)
            # L2 normalization
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_array = embeddings_array / norms
            embeddings = embeddings_array.tolist()

        return embeddings

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information and status.

        Returns:
            Dictionary with service metadata
        """
        backend_info = self.backend_manager.get_current_backend_info()

        return {
            "service": "EmbeddingService",
            "version": "1.0.0",
            "backend": backend_info,
            "requests_processed": self._request_counter,
            "available": self.backend_manager.is_available(),
            "supported_operations": ["embed_texts"],
            "max_batch_size": 128,
            "max_text_length": 8192,
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the service.

        Returns:
            Health status information
        """
        try:
            # Test with a simple embedding
            test_texts = ["Health check test"]
            test_request = EmbedRequest(texts=test_texts, batch_size=1)

            start_time = time.time()
            result = await self.embed_texts(test_request, request_id="health_check")
            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "backend_available": True,
                "test_embedding_dim": result.dim,
                "service_info": self.get_service_info(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend_available": self.backend_manager.is_available(),
                "service_info": self.get_service_info(),
            }
