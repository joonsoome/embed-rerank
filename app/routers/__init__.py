"""Router modules for FastAPI application."""

from . import (
    embedding_router,
    health_router,
    openai_router,
    reranking_router,
    tei_router,
)

__all__ = ["embedding_router", "reranking_router", "health_router", "openai_router", "tei_router"]
