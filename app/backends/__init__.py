"""Backend implementations for embedding models."""

from .base import BaseBackend, EmbeddingResult, RerankResult
from .factory import BackendFactory
from .mlx_backend import MLX_AVAILABLE, MLXBackend
from .torch_backend import TorchBackend

__all__ = [
    "BaseBackend",
    "EmbeddingResult",
    "RerankResult",
    "TorchBackend",
    "MLXBackend",
    "MLX_AVAILABLE",
    "BackendFactory",
]
