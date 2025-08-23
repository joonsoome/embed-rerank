"""
Pydantic models for API responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class EmbedResponse(BaseModel):
    """Response model for embedding generation."""
    
    vectors: List[List[float]] = Field(
        ..., 
        description="Generated embedding vectors"
    )
    dim: int = Field(
        ..., 
        description="Dimension of embedding vectors"
    )
    backend: str = Field(
        ..., 
        description="Backend used for generation"
    )
    device: str = Field(
        ..., 
        description="Device used for computation"
    )
    processing_time: float = Field(
        ..., 
        description="Processing time in seconds"
    )
    model_info: str = Field(
        ..., 
        description="Model identifier"
    )


class RerankResult(BaseModel):
    """Individual rerank result."""
    
    text: str = Field(
        ..., 
        description="Original passage text"
    )
    score: float = Field(
        ..., 
        description="Relevance score"
    )
    index: int = Field(
        ..., 
        description="Original index in input list"
    )


class RerankResponse(BaseModel):
    """Response model for document reranking."""
    
    results: List[RerankResult] = Field(
        ..., 
        description="Ranked results"
    )
    backend: str = Field(
        ..., 
        description="Backend used for reranking"
    )
    device: str = Field(
        ..., 
        description="Device used for computation"
    )
    method: str = Field(
        ..., 
        description="Reranking method used"
    )
    processing_time: float = Field(
        ..., 
        description="Processing time in seconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(
        ..., 
        description="Service status (healthy/unhealthy/not_loaded)"
    )
    backend: str = Field(
        ..., 
        description="Active backend"
    )
    model_loaded: bool = Field(
        ..., 
        description="Whether model is loaded"
    )
    model_info: Dict[str, Any] = Field(
        ..., 
        description="Model metadata"
    )
    device_info: Dict[str, Any] = Field(
        ..., 
        description="Device information"
    )
    uptime: float = Field(
        ..., 
        description="Service uptime in seconds"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(
        ..., 
        description="Error type"
    )
    message: str = Field(
        ..., 
        description="Error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional error details"
    )
