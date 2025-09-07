"""
Pydantic models for API requests.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class EmbedRequest(BaseModel):
    """Request model for embedding generation."""

    texts: List[str] = Field(
        ...,
        description="List of texts to embed",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": ["Hello world", "How are you?", "FastAPI is awesome"]},
    )
    batch_size: Optional[int] = Field(
        32, description="Batch size for processing", ge=1, le=128, json_schema_extra={"example": 32}
    )
    normalize: Optional[bool] = Field(
        True, description="Whether to normalize embeddings to unit length", json_schema_extra={"example": True}
    )

    # 🚀 텍스트 처리 옵션 추가!
    auto_truncate: Optional[bool] = Field(
        True,
        description="Whether to automatically truncate texts exceeding token limits",
        json_schema_extra={"example": True},
    )
    truncation_strategy: Optional[
        Literal["smart_truncate", "truncate", "extract", "error"]
    ] = Field(
        "smart_truncate",
        description=(
            "Strategy for handling long texts: smart_truncate (preserve sentences), "
            "truncate (hard cut), extract (key sentences), error (raise exception)"
        ),
        json_schema_extra={"example": "smart_truncate"},
    )
    max_tokens_override: Optional[int] = Field(
        None,
        description="Override the model's recommended max tokens (cannot exceed absolute maximum)",
        ge=1,
        le=8192,
        json_schema_extra={"example": 1024},
    )
    return_processing_info: Optional[bool] = Field(
        False,
        description="Whether to include text processing information in the response",
        json_schema_extra={"example": False},
    )

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v, info):
        """Validate texts array with enhanced error handling."""
        if not isinstance(v, list):
            raise ValueError("texts must be a list")

        if not v:
            raise ValueError("texts cannot be empty")

        # Get auto_truncate setting from the same data
        auto_truncate = info.data.get('auto_truncate', True) if info.data else True

        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")

            if not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or whitespace only")

            # Only enforce length limit if auto_truncate is False
            if not auto_truncate and len(text) > 8192:
                raise ValueError(f"Text at index {i} too long: {len(text)} > 8192 characters")

        return v

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v, info):
        """Validate batch size relative to number of texts."""
        if info.data and 'texts' in info.data and info.data['texts']:
            num_texts = len(info.data['texts'])
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
        json_schema_extra={"example": "What is machine learning?"},
    )
    passages: List[str] = Field(
        ...,
        description="List of passages to rerank",
        min_length=1,
        max_length=1000,
        json_schema_extra={
            "example": [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with many layers.",
                "Natural language processing helps computers understand text.",
            ]
        },
    )
    top_k: Optional[int] = Field(
        10, description="Number of top-ranked passages to return", ge=1, le=100, json_schema_extra={"example": 5}
    )
    return_documents: Optional[bool] = Field(
        True, description="Whether to return the original passage texts", json_schema_extra={"example": True}
    )

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate query text."""
        if not isinstance(v, str):
            raise ValueError("Query must be a string")

        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")

        return v.strip()

    @field_validator('passages')
    @classmethod
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

    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v, info):
        """Validate top_k relative to number of passages."""
        if info.data and 'passages' in info.data and info.data['passages']:
            num_passages = len(info.data['passages'])
            if v > num_passages:
                # Adjust top_k to not exceed number of passages
                return num_passages
        return v
