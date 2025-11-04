"""
MLX Cross-Encoder Reranker Backend (experimental scaffold)

This backend is a placeholder for a true MLX-native cross-encoder reranker.
It defines the class interface and basic wiring so we can iterate toward
supporting MLX-formatted reranker models (e.g., vserifsaglam/Qwen3-Reranker-4B-4bit-MLX).

For now, the factory will continue to route reranking to the Torch CrossEncoder
implementation. When this backend is completed, the factory can instantiate this
class when RERANKER_BACKEND=mlx on Apple Silicon.
"""

from typing import Any, Dict, List, Optional

try:
    import mlx  # type: ignore

    MLX_AVAILABLE = True
except Exception:
    MLX_AVAILABLE = False

from .base import BaseBackend, EmbeddingResult


class MLXCrossEncoderBackend(BaseBackend):
    """Experimental MLX-native cross-encoder reranker (scaffold).

    Notes:
        - This class currently does not implement actual scoring.
        - It's intended to be filled in with MLX-LM/transformer loading,
          pair tokenization, and a classification head forward pass that
          outputs a relevance score per (query, passage) pair.
        - Until implemented, the rerank_passages method raises NotImplementedError.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 16):
        super().__init__(model_name, device=device)
        self._batch_size = batch_size

    async def load_model(self) -> None:
        """Load the MLX model and tokenizer.

        Currently sets loaded flag to False if MLX is unavailable; otherwise marks
        as loaded to support health checks while implementation proceeds.
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX backend requested but MLX is not available on this system.\n"
                "Install mlx and ensure you're on Apple Silicon (arm64)."
            )

        # TODO: Load MLX model and tokenizer suitable for cross-encoder scoring
        # Placeholder: mark as loaded without actual model for now
        self.model = None
        self.tokenizer = None
        self._is_loaded = True
        self._load_time = 0.0

    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        """Embedding is not supported for the reranker backend.

        Returns a minimal zero vector to satisfy interface expectations in health checks.
        """
        import numpy as np

        self.validate_inputs(texts)
        vectors = np.zeros((len(texts), 1), dtype=np.float32)
        return EmbeddingResult(vectors=vectors, processing_time=0.0, device=self.device or "mlx", model_info=self.model_name)

    async def compute_similarity(self, query_embedding, passage_embeddings):
        """Not applicable for cross-encoder; provided for interface completeness."""
        import numpy as np

        return np.zeros((passage_embeddings.shape[0],), dtype=np.float32)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "rerank_method": "cross-encoder",
            "rerank_model_name": self.model_name,
            "backend": "mlx",
            "batch_size": self._batch_size,
            "implemented": False,
        }

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device": self.device or ("mlx" if MLX_AVAILABLE else "cpu"),
            "mlx_available": MLX_AVAILABLE,
        }

    async def rerank_passages(self, query: str, passages: List[str]) -> List[float]:
        """Score (query, passage) pairs with a cross-encoder.

        Raises:
            NotImplementedError: Until MLX cross-encoder scoring is implemented.
        """
        raise NotImplementedError(
            "MLXCrossEncoderBackend.rerank_passages is not implemented yet.\n"
            "Set RERANKER_BACKEND=torch to use the Torch CrossEncoder reranker for now."
        )
