"""
Configuration management for the embed-rerank API.
"""

import platform
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=9000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload for development")

    # Backend Selection
    backend: Literal["auto", "mlx", "torch"] = Field(default="auto", description="Backend to use for embeddings")

    # Model Configuration
    model_name: str = Field(default="Qwen/Qwen3-Embedding-4B", description="HuggingFace model identifier")
    model_path: Optional[Path] = Field(default=None, description="Path to MLX converted model (optional)")
    cross_encoder_model: Optional[str] = Field(default=None, description="Cross-encoder model for reranking")
    max_sequence_length: int = Field(default=512, description="Maximum input sequence length")

    # Performance Settings
    batch_size: int = Field(default=32, description="Default batch size")
    max_batch_size: int = Field(default=128, description="Maximum batch size")
    device_memory_fraction: float = Field(default=0.8, description="Fraction of device memory to use")

    # API Limits
    max_texts_per_request: int = Field(default=100, description="Maximum texts per embedding request")
    max_passages_per_rerank: int = Field(default=1000, description="Maximum passages per rerank request")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")

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
