"""
Pydantic models for the embedding and reranking API.
"""

from .requests import EmbedRequest, RerankRequest
from .responses import (
    EmbeddingVector,
    EmbedResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    RerankResponse,
    RerankResult,
)

__all__ = [
    # Request models
    "EmbedRequest",
    "RerankRequest",
    # Response models
    "EmbeddingVector",
    "EmbedResponse",
    "RerankResult",
    "RerankResponse",
    "HealthResponse",
    "ErrorResponse",
    "ModelInfo",
    "ModelsResponse",
]
