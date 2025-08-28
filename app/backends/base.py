"""
Abstract base class for embedding backends.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    vectors: np.ndarray
    processing_time: float
    device: str
    model_info: str


@dataclass
class RerankResult:
    """Result from reranking operation."""

    scores: List[float]
    indices: List[int]
    processing_time: float
    method: str
    results: Optional[List] = None  # For backwards compatibility with tests
    
    def __post_init__(self):
        """Set results based on scores and indices for test compatibility."""
        if self.results is None:
            self.results = [
                {"score": score, "index": idx} 
                for idx, score in enumerate(self.scores)
            ]


class BaseBackend(ABC):
    """Abstract base class for embedding backends."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the backend.

        Args:
            model_name: Name/path of the model to load
            device: Device to use (optional, will auto-detect if None)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._load_time = None

    @abstractmethod
    async def load_model(self) -> None:
        """Load the embedding model into memory."""
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            EmbeddingResult with vectors and metadata
        """
        pass

    @abstractmethod
    async def compute_similarity(self, query_embedding: np.ndarray, passage_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity scores between query and passage embeddings.

        Args:
            query_embedding: Query embedding vector
            passage_embeddings: Passage embedding matrix

        Returns:
            Array of similarity scores
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and configuration."""
        pass

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Return device information and capabilities."""
        pass

    @abstractmethod
    async def rerank_passages(self, query: str, passages: List[str]) -> List[float]:
        """
        Rerank passages based on relevance to the query.

        Args:
            query: Query text
            passages: List of passage texts

        Returns:
            List of relevance scores (higher is more relevant)
        """
        pass

    async def rerank_documents(self, query: str, docs: List[str]) -> RerankResult:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: Query text
            docs: List of document texts

        Returns:
            RerankResult with scores and metadata
        """
        start_time = time.time()
        scores = await self.rerank_passages(query, docs)
        processing_time = time.time() - start_time
        indices = list(range(len(docs)))
        
        return RerankResult(
            scores=scores,
            indices=indices,
            processing_time=processing_time,
            method="embedding_similarity"
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def load_time(self) -> Optional[float]:
        """Get model loading time in seconds."""
        return self._load_time

    def validate_inputs(self, texts: List[str], max_length: int = 512) -> None:
        """
        Validate input texts for processing.

        Args:
            texts: List of texts to validate
            max_length: Maximum length per text

        Raises:
            ValueError: If inputs are invalid
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")

        if len(texts) > 1000:  # Reasonable limit
            raise ValueError(f"Too many texts: {len(texts)} > 1000")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string, got {type(text)}")

            if len(text.strip()) == 0:
                raise ValueError(f"Text at index {i} is empty or whitespace only")

            if len(text) > max_length * 4:  # Rough character limit
                raise ValueError(f"Text at index {i} is too long: {len(text)} characters")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the backend.

        Returns:
            Dict with health status and metrics
        """
        start_time = time.time()

        health_status = {
            "backend": self.__class__.__name__,
            "model_loaded": self.is_loaded,
            "model_name": self.model_name,
            "device": self.device,
            "load_time": self.load_time,
        }

        if self.is_loaded:
            try:
                # Test with a simple embedding
                test_result = await self.embed_texts(["health check test"], batch_size=1)
                health_status.update(
                    {
                        "status": "healthy",
                        "test_embedding_time": test_result.processing_time,
                        "embedding_dim": (
                            test_result.vectors.shape[1] if test_result.vectors.ndim > 1 else len(test_result.vectors)
                        ),
                    }
                )
            except Exception as e:
                health_status.update({"status": "unhealthy", "error": str(e)})
        else:
            health_status["status"] = "not_loaded"

        health_status["check_time"] = time.time() - start_time
        return health_status


class BackendManager:
    """Manager for backend instances and operations."""

    def __init__(self, backend: BaseBackend):
        """
        Initialize the backend manager.

        Args:
            backend: Backend instance to manage
        """
        self.backend = backend
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the backend by loading the model."""
        if not self._initialized:
            await self.backend.load_model()
            self._initialized = True

    def get_backend(self) -> BaseBackend:
        """Get the managed backend instance."""
        return self.backend

    def get_current_backend(self) -> BaseBackend:
        """Alias for get_backend() for backward compatibility."""
        return self.get_backend()

    def is_ready(self) -> bool:
        """Check if the backend is ready for use."""
        return self._initialized and self.backend.is_loaded

    def is_available(self) -> bool:
        """Alias for is_ready() for backward compatibility."""
        return self.is_ready()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the managed backend."""
        if not self._initialized:
            return {"status": "not_initialized", "backend": self.backend.__class__.__name__}

        return await self.backend.health_check()

    def get_current_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        if not self._initialized:
            return {"status": "not_initialized"}

        info = self.backend.get_model_info()
        device_info = self.backend.get_device_info()

        return {
            "name": self.backend.__class__.__name__,
            "model_name": self.backend.model_name,
            "device": device_info.get("device", "unknown"),
            "status": "ready" if self.is_ready() else "initializing",
            **info,
        }

    async def cleanup(self) -> None:
        """Cleanup resources associated with the backend and manager.

        This attempts to gracefully shut down executors and unload models where
        possible. It's safe to call multiple times.
        """
        try:
            # Attempt to call backend-specific cleanup/unload if available
            if hasattr(self.backend, 'unload_model'):
                maybe = self.backend.unload_model
                if asyncio.iscoroutinefunction(maybe):
                    await maybe()
                else:
                    maybe()

            # Shutdown backend thread pool if present
            if hasattr(self.backend, '_executor') and self.backend._executor is not None:
                try:
                    self.backend._executor.shutdown(wait=True)
                except Exception:
                    pass

            # Clear loaded flags
            if hasattr(self.backend, '_is_loaded'):
                try:
                    self.backend._is_loaded = False
                except Exception:
                    pass

        finally:
            self._initialized = False

    # Backwards-compatible alias
    def get_backend_info(self) -> Dict[str, Any]:
        return self.get_current_backend_info()
