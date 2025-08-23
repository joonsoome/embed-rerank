"""
Configuration management for the embed-rerank API.
"""

import os
import platform
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=9000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    
    # Backend Selection
    backend: Literal["auto", "mlx", "torch"] = Field(
        default="auto",
        description="Backend to use for embeddings"
    )
    
    # Model Configuration
    model_name: str = Field(
        default="Qwen/Qwen3-Embedding-4B",
        description="HuggingFace model identifier"
    )
    mlx_model_path: Optional[Path] = Field(
        default=None,
        description="Path to MLX converted model (optional)"
    )
    cross_encoder_model: Optional[str] = Field(
        default=None,
        description="Cross-encoder model for reranking"
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum input sequence length"
    )
    
    # Performance Settings
    batch_size: int = Field(default=32, description="Default batch size")
    max_batch_size: int = Field(default=128, description="Maximum batch size")
    device_memory_fraction: float = Field(
        default=0.8,
        description="Fraction of device memory to use"
    )
    
    # API Limits
    max_texts_per_request: int = Field(
        default=100,
        description="Maximum texts per embedding request"
    )
    max_passages_per_rerank: int = Field(
        default=1000,
        description="Maximum passages per rerank request"
    )
    request_timeout: int = Field(
        default=300,
        description="Request timeout in seconds"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    
    @validator("backend")
    def validate_backend(cls, v):
        """Auto-detect best backend if set to 'auto'."""
        if v == "auto":
            # Check if we're on Apple Silicon
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                try:
                    import mlx.core
                    return "mlx"
                except ImportError:
                    return "torch"
            else:
                return "torch"
        return v
    
    @validator("mlx_model_path")
    def validate_mlx_path(cls, v, values):
        """Validate MLX model path if specified."""
        if v is not None:
            if not isinstance(v, Path):
                v = Path(v)
            if not v.exists():
                raise ValueError(f"MLX model path does not exist: {v}")
        return v
    
    @validator("batch_size", "max_batch_size")
    def validate_batch_sizes(cls, v):
        """Ensure batch sizes are positive."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v
    
    @validator("device_memory_fraction")
    def validate_memory_fraction(cls, v):
        """Ensure memory fraction is between 0 and 1."""
        if not 0.0 < v <= 1.0:
            raise ValueError("Device memory fraction must be between 0 and 1")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False


# Global settings instance
settings = Settings()
