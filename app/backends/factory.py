"""
Backend factory for creating appropriate embedding backends.
"""

import platform
from typing import Optional

from app.backends.base import BaseBackend
from app.backends.mlx_backend import MLX_AVAILABLE, MLXBackend
from app.backends.torch_backend import TorchBackend
from app.config import settings
from app.utils.device import detect_optimal_device
from app.utils.logger import setup_logging

logger = setup_logging()


class BackendFactory:
    """Factory for creating embedding backends."""

    @staticmethod
    def create_backend(backend_type: str = "auto", model_name: Optional[str] = None, **kwargs) -> BaseBackend:
        """
        Create and configure the appropriate backend for the current platform.

        Args:
            backend_type: Backend to use ("auto", "mlx", "torch")
            model_name: Model name to use (overrides settings)
            **kwargs: Additional backend configuration

        Returns:
            Configured backend instance

        Raises:
            ValueError: If backend type is invalid or unavailable for platform
        """
        # Use provided model_name or fallback to settings
        if model_name is None:
            model_name = settings.model_name

        if backend_type == "auto":
            backend_type = BackendFactory._detect_optimal_backend(model_name)

        logger.info("Creating backend", backend_type=backend_type, model_name=model_name)

        if backend_type == "mlx":
            return BackendFactory._create_mlx_backend(model_name, **kwargs)
        elif backend_type == "torch":
            return BackendFactory._create_torch_backend(model_name, **kwargs)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    @staticmethod
    def _detect_optimal_backend(model_name: str = None) -> str:
        """Detect the optimal backend for the current system and model."""
        device_info = detect_optimal_device()

        # Check if the model name suggests MLX compatibility
        if model_name and ("mlx-community" in model_name.lower() or "mlx" in model_name.lower()):
            if device_info.get("apple_silicon", False) and device_info.get("mlx_available", False):
                logger.info("Auto-selected MLX backend", reason="mlx_model_detected")
                return "mlx"
            else:
                logger.warning(
                    "MLX model detected but MLX not available - falling back to torch",
                    model_name=model_name
                )

        # Default to PyTorch for better compatibility
        if device_info.get("torch_available", False):
            logger.info("Auto-selected Torch backend", reason="torch_available_default")
            return "torch"

        # Only use MLX if specifically optimized models are available
        if device_info.get("apple_silicon", False) and device_info.get("mlx_available", False):
            logger.info("Auto-selected MLX backend", reason="apple_silicon_with_mlx")
            return "mlx"

        # Default fallback
        logger.info("Defaulting to Torch backend")
        return "torch"

    @staticmethod
    def _create_mlx_backend(model_name: str, **kwargs) -> MLXBackend:
        """Create MLX backend with validation."""
        if not MLX_AVAILABLE:
            raise ValueError(
                "MLX backend requested but MLX is not available. "
                "MLX requires macOS with Apple Silicon. "
                "Install with: pip install mlx>=0.4.0"
            )

        if platform.system() != "Darwin":
            raise ValueError("MLX backend requires macOS")

        if platform.machine() != "arm64":
            raise ValueError("MLX backend requires Apple Silicon (arm64)")

        model_path = kwargs.get("model_path", settings.model_path)

        # Use the provided model name instead of forcing MLX-specific model
        logger.info("Creating MLX backend", model_name=model_name, model_path=model_path)

        return MLXBackend(model_name, model_path=model_path)

    @staticmethod
    def _create_torch_backend(model_name: str, **kwargs) -> TorchBackend:
        """Create PyTorch backend."""
        device = kwargs.get("device")

        logger.info("Creating Torch backend", model_name=model_name, device=device)

        return TorchBackend(model_name, device=device)

    @staticmethod
    def get_available_backends() -> dict:
        """Get information about available backends."""
        device_info = detect_optimal_device()

        backends = {
            "torch": {"available": device_info.get("torch_available", False), "devices": []},
            "mlx": {
                "available": (device_info.get("apple_silicon", False) and device_info.get("mlx_available", False)),
                "devices": ["mlx"] if device_info.get("mlx_available", False) else [],
            },
        }

        # Add PyTorch device options
        if backends["torch"]["available"]:
            if device_info.get("mps_available", False):
                backends["torch"]["devices"].append("mps")
            if device_info.get("cuda_available", False):
                backends["torch"]["devices"].append("cuda")
            backends["torch"]["devices"].append("cpu")

        return backends

    @staticmethod
    def validate_backend_config(backend_type: str, model_name: str) -> bool:
        """
        Validate if a backend configuration is valid.

        Args:
            backend_type: Backend type to validate
            model_name: Model name to validate

        Returns:
            True if configuration is valid
        """
        try:
            # Try to create the backend (without loading the model)
            BackendFactory.create_backend(backend_type, model_name)
            return True
        except Exception as e:
            logger.warning(
                "Backend configuration validation failed",
                backend_type=backend_type,
                model_name=model_name,
                error=str(e),
            )
            return False
