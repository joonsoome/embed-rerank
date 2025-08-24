"""
Pydantic models for API responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


"""
Pydantic models for API responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class EmbeddingVector(BaseModel):
    """Single embedding vector with metadata."""
    
    embedding: List[float] = Field(
        ...,
        description="The embedding vector",
        example=[0.1, -0.2, 0.5, 0.8, -0.1]
    )
    index: int = Field(
        ...,
        description="Index of the original text in the input list",
        ge=0,
        example=0
    )
    text: Optional[str] = Field(
        None,
        description="Original text that was embedded (optional)",
        example="Hello world"
    )


class EmbedResponse(BaseModel):
    """Response model for embedding generation."""
    
    # Enhanced structure with both new format and backward compatibility
    embeddings: List[EmbeddingVector] = Field(
        ...,
        description="List of embedding vectors with metadata",
        min_items=1
    )
    # Keep original fields for backward compatibility
    vectors: List[List[float]] = Field(
        ..., 
        description="Generated embedding vectors (legacy format)"
    )
    dim: int = Field(
        ..., 
        description="Dimension of embedding vectors",
        example=384
    )
    backend: str = Field(
        ..., 
        description="Backend used for generation",
        example="mlx"
    )
    device: str = Field(
        ..., 
        description="Device used for computation",
        example="mps"
    )
    processing_time: float = Field(
        ..., 
        description="Processing time in seconds",
        example=0.045
    )
    model_info: str = Field(
        ..., 
        description="Model identifier",
        example="all-MiniLM-L6-v2"
    )
    # Enhanced metadata
    usage: Dict[str, Any] = Field(
        ...,
        description="Usage statistics",
        example={
            "total_texts": 3,
            "total_tokens": 15,
            "processing_time_ms": 45.2
        }
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RerankResult(BaseModel):
    """Individual rerank result."""
    
    text: str = Field(
        ..., 
        description="Original passage text",
        example="Machine learning is a subset of artificial intelligence."
    )
    score: float = Field(
        ..., 
        description="Relevance score (higher is more relevant)",
        ge=0.0,
        le=1.0,
        example=0.8542
    )
    index: int = Field(
        ..., 
        description="Original index in input list",
        ge=0,
        example=0
    )


class RerankResponse(BaseModel):
    """Response model for document reranking."""
    
    results: List[RerankResult] = Field(
        ..., 
        description="Ranked results ordered by relevance score (descending)",
        min_items=1
    )
    query: str = Field(
        ...,
        description="The query that was used for ranking",
        example="What is machine learning?"
    )
    backend: str = Field(
        ..., 
        description="Backend used for reranking",
        example="torch"
    )
    device: str = Field(
        ..., 
        description="Device used for computation",
        example="mps"
    )
    method: str = Field(
        ..., 
        description="Reranking method used",
        example="cross-encoder"
    )
    processing_time: float = Field(
        ..., 
        description="Processing time in seconds",
        example=0.123
    )
    model_info: str = Field(
        ...,
        description="Model identifier",
        example="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    # Enhanced metadata
    usage: Dict[str, Any] = Field(
        ...,
        description="Usage statistics",
        example={
            "total_passages": 10,
            "returned_passages": 5,
            "processing_time_ms": 123.7
        }
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(
        ..., 
        description="Service status (healthy/unhealthy/not_loaded)",
        example="healthy"
    )
    backend: str = Field(
        ..., 
        description="Active backend",
        example="mlx"
    )
    model_loaded: bool = Field(
        ..., 
        description="Whether model is loaded",
        example=True
    )
    model_info: Dict[str, Any] = Field(
        ..., 
        description="Model metadata",
        example={
            "embedding_model": "all-MiniLM-L6-v2",
            "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "embedding_dim": 384
        }
    )
    device_info: Dict[str, Any] = Field(
        ..., 
        description="Device information",
        example={
            "device": "mps",
            "backend": "mlx",
            "memory_usage": "2.1GB"
        }
    )
    uptime: float = Field(
        ..., 
        description="Service uptime in seconds",
        example=3600.5
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Health check timestamp"
    )
    version: str = Field(
        ...,
        description="API version",
        example="1.0.0"
    )
    backends: Dict[str, bool] = Field(
        ...,
        description="Backend availability status",
        example={
            "torch": True,
            "mlx": True
        }
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(
        ..., 
        description="Error type",
        example="ValidationError"
    )
    message: str = Field(
        ..., 
        description="Human-readable error message",
        example="Invalid input: texts cannot be empty"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional error details",
        example={
            "field": "texts",
            "input": [],
            "constraint": "min_items=1"
        }
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request identifier for tracking",
        example="req_123456789"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelInfo(BaseModel):
    """Model information response."""
    
    name: str = Field(
        ...,
        description="Model name",
        example="all-MiniLM-L6-v2"
    )
    type: str = Field(
        ...,
        description="Model type (embedding or reranking)",
        example="embedding"
    )
    backend: str = Field(
        ...,
        description="Backend being used (torch or mlx)",
        example="mlx"
    )
    dimension: Optional[int] = Field(
        None,
        description="Embedding dimension (for embedding models)",
        example=384
    )
    max_length: int = Field(
        ...,
        description="Maximum sequence length",
        example=512
    )
    loaded: bool = Field(
        ...,
        description="Whether the model is currently loaded",
        example=True
    )


class ModelsResponse(BaseModel):
    """Response model for listing available models."""
    
    embedding_models: List[ModelInfo] = Field(
        ...,
        description="Available embedding models"
    )
    reranking_models: List[ModelInfo] = Field(
        ...,
        description="Available reranking models"
    )
    default_embedding_model: str = Field(
        ...,
        description="Default embedding model name",
        example="all-MiniLM-L6-v2"
    )
    default_reranking_model: str = Field(
        ...,
        description="Default reranking model name",
        example="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
