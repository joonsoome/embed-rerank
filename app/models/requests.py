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
        max_items=100,
        example=["Hello world", "How are you?", "FastAPI is awesome"]
    )
    batch_size: Optional[int] = Field(
        32,
        description="Batch size for processing",
        ge=1,
        le=128,
        example=32
    )
    normalize: Optional[bool] = Field(
        True,
        description="Whether to normalize embeddings to unit length",
        example=True
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
    
    @validator('batch_size')
    def validate_batch_size(cls, v, values):
        """Validate batch size relative to number of texts."""
        if 'texts' in values and values['texts']:
            num_texts = len(values['texts'])
            if v > num_texts:
                # Adjust batch size to not exceed number of texts
                return num_texts
        return v


class RerankRequest(BaseModel):
    """Request model for reranking query-passage pairs."""
    
    query: str = Field(
        ...,
        description="Query text to rank passages against",
        min_length=1,
        max_length=2048,
        example="What is machine learning?"
    )
    passages: List[str] = Field(
        ...,
        description="List of passages to rerank",
        min_items=1,
        max_items=1000,
        example=[
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing helps computers understand text."
        ]
    )
    top_k: Optional[int] = Field(
        10,
        description="Number of top-ranked passages to return",
        ge=1,
        le=100,
        example=5
    )
    return_documents: Optional[bool] = Field(
        True,
        description="Whether to return the original passage texts",
        example=True
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query text."""
        if not isinstance(v, str):
            raise ValueError("Query must be a string")
        
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        
        return v.strip()
    
    @validator('passages')
    def validate_passages(cls, v):
        """Validate passage inputs."""
        if not v:
            raise ValueError("passages cannot be empty")
        
        for i, passage in enumerate(v):
            if not isinstance(passage, str):
                raise ValueError(f"Passage at index {i} must be a string")
            
            if not passage.strip():
                raise ValueError(f"Passage at index {i} cannot be empty or whitespace only")
            
            if len(passage) > 4096:  # Reasonable character limit for passages
                raise ValueError(f"Passage at index {i} too long: {len(passage)} > 4096 characters")
        
        return v
    
    @validator('top_k')
    def validate_top_k(cls, v, values):
        """Validate top_k relative to number of passages."""
        if 'passages' in values and values['passages']:
            num_passages = len(values['passages'])
            if v > num_passages:
                # Adjust top_k to not exceed number of passages
                return num_passages
        return v