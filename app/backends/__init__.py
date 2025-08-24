"""Backend implementations for embedding models."""

from .base import BaseBackend, EmbeddingResult, RerankResult
from .torch_backend import TorchBackend
from .mlx_backend import MLXBackend, MLX_AVAILABLE
from .factory import BackendFactory

__all__ = [
    "BaseBackend",
    "EmbeddingResult",
    "RerankResult",
    "TorchBackend",
    "MLXBackend",
    "MLX_AVAILABLE",
    "BackendFactory",
]
