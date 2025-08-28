"""
Pydantic models for API responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingVector(BaseModel):
    """Single embedding vector with metadata."""

    embedding: List[float] = Field(
        ..., description="The embedding vector", json_schema_extra={"example": [0.1, -0.2, 0.5, 0.8, -0.1]}
    )
    index: int = Field(
        ..., description="Index of the original text in the input list", ge=0, json_schema_extra={"example": 0}
    )
    text: Optional[str] = Field(
        None, description="Original text that was embedded (optional)", json_schema_extra={"example": "Hello world"}
    )


class EmbedResponse(BaseModel):
    """Response model for embedding generation."""
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        protected_namespaces=()  # Allow model_ fields
    )

    # Enhanced structure with both new format and backward compatibility
    embeddings: List[EmbeddingVector] = Field(
        ..., description="List of embedding vectors with metadata", min_length=1
    )
    # Keep original fields for backward compatibility
    vectors: List[List[float]] = Field(..., description="Generated embedding vectors (legacy format)")
    dim: int = Field(..., description="Dimension of embedding vectors", json_schema_extra={"example": 384})
    backend: str = Field(..., description="Backend used for generation", json_schema_extra={"example": "mlx"})
    device: str = Field(..., description="Device used for computation", json_schema_extra={"example": "mps"})
    processing_time: float = Field(
        ..., description="Processing time in seconds", json_schema_extra={"example": 0.045}
    )
    model_info: str = Field(
        ..., description="Model identifier", json_schema_extra={"example": "all-MiniLM-L6-v2"}
    )
    # Enhanced metadata
    usage: Dict[str, Any] = Field(
        ...,
        description="Usage statistics",
        json_schema_extra={"example": {"total_texts": 3, "total_tokens": 15, "processing_time_ms": 45.2}},
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    # Additional fields for test compatibility
    num_texts: Optional[int] = Field(
        None, description="Number of input texts processed", json_schema_extra={"example": 2}
    )


class RerankResult(BaseModel):
    """Individual rerank result."""

    text: Optional[str] = Field(
        None,
        description="Original passage text",
        json_schema_extra={"example": "Machine learning is a subset of artificial intelligence."},
    )
    score: float = Field(
        ...,
        description="Relevance score (higher is more relevant, can be negative for similarity methods)",
        ge=-1.0,
        le=1.0,
        json_schema_extra={"example": 0.8542},
    )
    index: int = Field(
        ..., description="Original index in input list", ge=0, json_schema_extra={"example": 0}
    )


class RerankResponse(BaseModel):
    """Response model for document reranking."""
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        protected_namespaces=()  # Allow model_ fields
    )

    results: List[RerankResult] = Field(
        ..., description="Ranked results ordered by relevance score (descending)", min_length=1
    )
    query: str = Field(
        ...,
        description="The query that was used for ranking",
        json_schema_extra={"example": "What is machine learning?"},
    )
    backend: str = Field(..., description="Backend used for reranking", json_schema_extra={"example": "torch"})
    device: str = Field(..., description="Device used for computation", json_schema_extra={"example": "mps"})
    method: str = Field(
        ..., description="Reranking method used", json_schema_extra={"example": "cross-encoder"}
    )
    processing_time: float = Field(
        ..., description="Processing time in seconds", json_schema_extra={"example": 0.123}
    )
    model_info: str = Field(
        ...,
        description="Model identifier",
        json_schema_extra={"example": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
    )
    # Enhanced metadata
    usage: Dict[str, Any] = Field(
        ...,
        description="Usage statistics",
        json_schema_extra={"example": {"total_passages": 10, "returned_passages": 5, "processing_time_ms": 123.7}},
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    # Additional fields for test compatibility
    num_passages: Optional[int] = Field(
        None, description="Number of input passages processed", json_schema_extra={"example": 5}
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    status: str = Field(
        ...,
        description="Service status (healthy/unhealthy/not_ready/warning)",
        json_schema_extra={"example": "healthy"},
    )
    uptime: float = Field(..., description="Service uptime in seconds", json_schema_extra={"example": 3600.5})
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    service: Optional[Dict[str, Any]] = Field(
        None,
        description="Service information",
        json_schema_extra={
            "example": {"name": "embed-rerank", "version": "1.0.0", "description": "Embedding & reranking service"}
        },
    )
    backend: Optional[Dict[str, Any]] = Field(
        None,
        description="Backend information",
        json_schema_extra={
            "example": {
                "name": "MLXBackend",
                "status": "healthy",
                "model_loaded": True,
                "model_name": "Qwen/Qwen3-Embedding-4B",
                "device": "mlx",
                "load_time": 2.5,
            }
        },
    )
    system: Optional[Dict[str, Any]] = Field(
        None,
        description="System resource information",
        json_schema_extra={
            "example": {"cpu_percent": 15.2, "memory_percent": 45.8, "memory_available_gb": 8.2, "memory_total_gb": 16.0}
        },
    )
    performance: Optional[Dict[str, Any]] = Field(
        None,
        description="Performance metrics",
        json_schema_extra={"example": {"test_embedding_time": 0.045, "embedding_dimension": 384}},
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    error: str = Field(..., description="Error type", json_schema_extra={"example": "ValidationError"})
    message: str = Field(
        ...,
        description="Human-readable error message",
        json_schema_extra={"example": "Invalid input: texts cannot be empty"},
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details",
        json_schema_extra={"example": {"field": "texts", "input": [], "constraint": "min_length=1"}},
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(
        None, description="Request identifier for tracking", json_schema_extra={"example": "req_123456789"}
    )


class ModelInfo(BaseModel):
    """Model information response."""

    name: str = Field(..., description="Model name", json_schema_extra={"example": "all-MiniLM-L6-v2"})
    type: str = Field(
        ..., description="Model type (embedding or reranking)", json_schema_extra={"example": "embedding"}
    )
    backend: str = Field(
        ..., description="Backend being used (torch or mlx)", json_schema_extra={"example": "mlx"}
    )
    dimension: Optional[int] = Field(
        None, description="Embedding dimension (for embedding models)", json_schema_extra={"example": 384}
    )
    max_length: int = Field(..., description="Maximum sequence length", json_schema_extra={"example": 512})
    loaded: bool = Field(
        ..., description="Whether the model is currently loaded", json_schema_extra={"example": True}
    )


class ModelsResponse(BaseModel):
    """Response model for listing available models."""
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    embedding_models: List[ModelInfo] = Field(..., description="Available embedding models")
    reranking_models: List[ModelInfo] = Field(..., description="Available reranking models")
    default_embedding_model: str = Field(
        ..., description="Default embedding model name", json_schema_extra={"example": "all-MiniLM-L6-v2"}
    )
    default_reranking_model: str = Field(
        ...,
        description="Default reranking model name",
        json_schema_extra={"example": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
