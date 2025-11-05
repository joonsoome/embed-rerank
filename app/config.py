"""
Configuration management for the embed-rerank API.
"""

import platform
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import AliasChoices, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=9000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload for development")

    # Backend Selection
    backend: Literal["auto", "mlx", "torch"] = Field(default="auto", description="Backend to use for embeddings")
    reranker_backend: Literal["auto", "mlx", "torch"] = Field(
        default="auto", description="Backend to use for reranker (cross-encoder)",
        validation_alias=AliasChoices("RERANKER_BACKEND", "reranker_backend"),
    )

    # Model Configuration
    model_name: str = Field(default="Qwen/Qwen3-Embedding-4B", description="HuggingFace model identifier")
    model_path: Optional[Path] = Field(default=None, description="Path to MLX converted model (optional)")
    cross_encoder_model: Optional[str] = Field(
        default=None,
        description="Cross-encoder model for reranking",
        validation_alias=AliasChoices(
            "RERANKER_MODEL_ID",
            "RERANKER_MODEL_NAME",  # prefer explicit alias first
            "CROSS_ENCODER_MODEL",
            "cross_encoder_model",
        ),
    )
    max_sequence_length: int = Field(default=512, description="Maximum input sequence length")

    # Reranker-specific optional settings
    rerank_max_seq_len: Optional[int] = Field(
        default=None,
        description="Optional override for reranker max sequence length (pair)",
        validation_alias=AliasChoices("RERANK_MAX_SEQ_LEN", "rerank_max_seq_len"),
    )
    rerank_batch_size: Optional[int] = Field(
        default=None,
        description="Optional override for reranker batch size",
        validation_alias=AliasChoices("RERANK_BATCH_SIZE", "rerank_batch_size"),
    )
    # Reranker pooling and score normalization (experimental MLX)
    rerank_pooling: Literal["mean", "cls"] = Field(
        default="mean",
        description="Pooling strategy for MLX reranker (mean or cls)",
        validation_alias=AliasChoices("RERANK_POOLING", "rerank_pooling"),
    )
    rerank_score_norm: Literal["none", "sigmoid", "minmax"] = Field(
        default="none",
        description="Score normalization strategy for reranker outputs",
        validation_alias=AliasChoices("RERANK_SCORE_NORM", "rerank_score_norm"),
    )
    # OpenAI compatibility toggles
    openai_rerank_auto_sigmoid: bool = Field(
        default=True,
        description="Automatically apply sigmoid normalization for OpenAI-compatible rerank requests",
        validation_alias=AliasChoices("OPENAI_RERANK_AUTO_SIGMOID", "openai_rerank_auto_sigmoid"),
    )

    # Embedding output dimension controls (env-configurable)
    output_embedding_dimension: Optional[int] = Field(
        default=None,
        description="If set, enforce this embedding dimension for outputs (padding/truncation)",
        validation_alias=AliasChoices("OUTPUT_EMBEDDING_DIMENSION", "output_embedding_dimension"),
    )
    dimension_strategy: Literal["as_is", "pad_or_truncate", "hidden_size"] = Field(
        default="as_is",
        description=(
            "How to determine final embedding dimension: "
            "'as_is' = use backend output; "
            "'pad_or_truncate' = enforce OUTPUT_EMBEDDING_DIMENSION; "
            "'hidden_size' = enforce model hidden_size from config"
        ),
        validation_alias=AliasChoices("DIMENSION_STRATEGY", "dimension_strategy"),
    )

    # Performance Settings
    batch_size: int = Field(default=32, description="Default batch size")
    max_batch_size: int = Field(default=128, description="Maximum batch size")
    device_memory_fraction: float = Field(default=0.8, description="Fraction of device memory to use")

    # API Limits
    max_texts_per_request: int = Field(default=100, description="Maximum texts per embedding request")
    max_passages_per_rerank: int = Field(default=1000, description="Maximum passages per rerank request")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")

    # 🚀 Text Processing Configuration (NEW!)
    # Default settings for text processing when not specified in requests
    default_auto_truncate: bool = Field(default=True, description="Default auto-truncation setting")
    default_truncation_strategy: Literal["smart_truncate", "truncate", "extract", "error"] = Field(
        default="smart_truncate", description="Default truncation strategy"
    )
    default_max_tokens_override: Optional[int] = Field(
        default=None, description="Default max tokens override (None = use model default)"
    )
    default_return_processing_info: bool = Field(
        default=False, description="Default setting for returning processing information"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")

    # FastAPI Configuration
    allowed_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    allowed_hosts: List[str] = Field(default=["*"], description="Trusted host middleware allowed hosts")
    cors_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_methods: List[str] = Field(default=["GET", "POST", "OPTIONS"], description="Allowed CORS methods")

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Auto-detect best backend if set to 'auto'."""
        if v == "auto":
            # Check if we're on Apple Silicon
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                try:
                    import mlx.core  # noqa: F401

                    return "mlx"
                except ImportError:
                    return "torch"
            else:
                return "torch"
        return v

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: Optional[Path], info: ValidationInfo) -> Optional[Path]:  # type: ignore[override]
        """Validate model path if specified."""
        if v is not None and not v.exists():
            raise ValueError(f"Model path does not exist: {v}")
        return v

    @field_validator("cross_encoder_model")
    @classmethod
    def empty_string_to_none(cls, v: Optional[str]) -> Optional[str]:
        """Normalize empty strings from env into None so other aliases can take effect."""
        if v is not None and str(v).strip() == "":
            return None
        return v

    @field_validator(
        "output_embedding_dimension",
        "default_max_tokens_override",
        "rerank_max_seq_len",
        "rerank_batch_size",
        mode="before",
    )
    @classmethod
    def empty_string_ints_to_none(cls, v):
        """Treat empty-string values for optional int fields as None to avoid validation errors."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("batch_size", "max_batch_size")
    @classmethod
    def validate_batch_sizes(cls, v: int) -> int:
        """Ensure batch sizes are positive."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v

    @field_validator("device_memory_fraction")
    @classmethod
    def validate_memory_fraction(cls, v: float) -> float:
        """Ensure memory fraction is between 0 and 1."""
        if not 0.0 < v <= 1.0:
            raise ValueError("Device memory fraction must be between 0 and 1")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # ignore unknown keys (backward compatibility with older .env entries)
    )


# Global settings instance
settings = Settings()
