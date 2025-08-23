"""
Pydantic models for API requests.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class EmbedRequest(BaseModel):
    """Request model for embedding generation."""
    
    texts: List[str] = Field(
        ..., 
        description="List of texts to embed",
        min_items=1,
        max_items=100
    )
    batch_size: Optional[int] = Field(
        32,
        description="Batch size for processing",
        ge=1,
        le=128
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate text inputs."""
        if not v:
            raise ValueError("texts cannot be empty")
        
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")
            
            if not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or whitespace only")
            
            if len(text) > 8192:  # Reasonable character limit
                raise ValueError(f"Text at index {i} too long: {len(text)} > 8192 characters")
        
        return v


class RerankRequest(BaseModel):
    """Request model for document reranking."""
    
    query: str = Field(
        ...,
        description="Query text for reranking",
        min_length=1,
        max_length=8192
    )
    passages: List[str] = Field(
        ...,
        description="List of passages to rerank",
        min_items=1,
        max_items=1000
    )
    top_k: Optional[int] = Field(
        None,
        description="Number of top results to return",
        ge=1
    )
    use_cross_encoder: Optional[bool] = Field(
        False,
        description="Use cross-encoder for reranking"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query text."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()
    
    @validator('passages')
    def validate_passages(cls, v):
        """Validate passage texts."""
        if not v:
            raise ValueError("passages cannot be empty")
        
        for i, passage in enumerate(v):
            if not isinstance(passage, str):
                raise ValueError(f"Passage at index {i} must be a string")
            
            if not passage.strip():
                raise ValueError(f"Passage at index {i} cannot be empty or whitespace only")
            
            if len(passage) > 8192:  # Reasonable character limit
                raise ValueError(f"Passage at index {i} too long: {len(passage)} > 8192 characters")
        
        return v
    
    @validator('top_k')
    def validate_top_k(cls, v, values):
        """Validate top_k parameter."""
        if v is not None and 'passages' in values:
            if v > len(values['passages']):
                raise ValueError(f"top_k ({v}) cannot be greater than number of passages ({len(values['passages'])})")
        return v
