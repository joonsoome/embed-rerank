"""
Backend Tests for Embed-Rerank Service

This module contains comprehensive tests for all backend implementations:
- Backend Factory functionality
- Apple MLX Backend
- PyTorch Backend
- Backend Manager functionality
"""

import platform

import numpy as np
import pytest

from app.backends.base import BackendManager
from app.backends.factory import BackendFactory
from app.backends.mlx_backend import MLX_AVAILABLE
from app.backends.torch_backend import TorchBackend
from app.utils.benchmark import BackendBenchmark


class TestBackendFactory:
    """Tests for backend factory functionality."""

    def test_get_available_backends(self):
        """Test available backends detection."""
        available = BackendFactory.get_available_backends()
        assert "torch" in available
        if platform.system() == "Darwin":  # macOS
            assert "mlx" in available

    def test_create_auto_backend(self):
        """Test auto-detection backend creation."""
        backend = BackendFactory.create_backend("auto", "sentence-transformers/all-MiniLM-L6-v2")
        assert backend is not None
        assert hasattr(backend, 'load_model')
        assert hasattr(backend, 'embed_texts')

    def test_create_torch_backend(self):
        """Test explicit torch backend creation."""
        backend = BackendFactory.create_backend("torch", "sentence-transformers/all-MiniLM-L6-v2")
        assert backend is not None
        assert backend.__class__.__name__ == "TorchBackend"

    @pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
    def test_create_mlx_backend(self):
        """Test explicit MLX backend creation."""
        backend = BackendFactory.create_backend("mlx", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
        assert backend is not None
        assert backend.__class__.__name__ == "MLXBackend"

    def test_invalid_backend_type(self):
        """Test handling of invalid backend type."""
        with pytest.raises(ValueError):
            BackendFactory.create_backend("invalid", "some-model")


class TestTorchBackend:
    """Tests for PyTorch backend functionality."""

    @pytest.mark.asyncio
    async def test_torch_backend_initialization(self):
        """Test torch backend initialization."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        assert not backend.is_loaded
        assert backend.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_torch_backend_model_loading(self):
        """Test torch backend model loading."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")

        # Test model loading
        await backend.load_model()
        assert backend.is_loaded
        assert backend.load_time is not None
        assert backend.load_time > 0

    @pytest.mark.asyncio
    async def test_torch_backend_embedding_generation(self):
        """Test torch backend embedding generation."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        await backend.load_model()

        # Test embedding generation
        texts = ["Hello world", "How are you?", "This is a test"]
        result = await backend.embed_texts(texts, batch_size=2)

        assert result.vectors.shape[0] == len(texts)
        assert result.vectors.shape[1] > 0  # Should have some embedding dimension
        assert result.processing_time > 0
        assert isinstance(result.vectors, np.ndarray)

    @pytest.mark.asyncio
    async def test_torch_backend_single_text(self):
        """Test torch backend with single text."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        await backend.load_model()

        # Test single text
        result = await backend.embed_texts(["Single text"])
        assert result.vectors.shape[0] == 1
        assert result.vectors.shape[1] > 0

    @pytest.mark.asyncio
    async def test_torch_backend_device_info(self):
        """Test torch backend device information."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        await backend.load_model()

        device_info = backend.get_device_info()
        assert "device" in device_info
        assert "available_memory" in device_info
        assert device_info["device"] in ["cpu", "mps", "cuda"]

    @pytest.mark.asyncio
    async def test_torch_backend_model_info(self):
        """Test torch backend model information."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        await backend.load_model()

        model_info = backend.get_model_info()
        assert "model_name" in model_info
        assert "embedding_dim" in model_info
        assert model_info["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_torch_backend_health_check(self):
        """Test torch backend health check."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        await backend.load_model()

        health = await backend.health_check()
        assert "status" in health
        assert "backend" in health
        assert health["status"] == "healthy"


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMLXBackend:
    """Tests for Apple MLX backend functionality."""

    @pytest.mark.asyncio
    async def test_mlx_backend_creation(self):
        """Test MLX backend creation."""
        backend = BackendFactory.create_backend("mlx", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
        assert backend is not None
        assert backend.__class__.__name__ == "MLXBackend"

    @pytest.mark.asyncio
    async def test_mlx_backend_loading(self):
        """Test MLX backend model loading."""
        backend = BackendFactory.create_backend("mlx", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ")

        # Test loading
        await backend.load_model()
        assert backend.is_loaded
        assert backend.load_time is not None

    @pytest.mark.asyncio
    async def test_mlx_backend_embeddings(self):
        """Test MLX backend embedding generation."""
        backend = BackendFactory.create_backend("mlx", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
        await backend.load_model()

        # Test embeddings
        texts = ["Apple MLX is fast", "Testing embedding generation"]
        result = await backend.embed_texts(texts)

        assert result.vectors.shape[0] == len(texts)
        assert result.vectors.shape[1] > 0
        assert result.processing_time > 0


class TestBackendManager:
    """Tests for backend manager functionality."""

    @pytest.mark.asyncio
    async def test_backend_manager_initialization(self):
        """Test backend manager initialization."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        manager = BackendManager(backend)

        assert not manager.is_ready()
        await manager.initialize()
        assert manager.is_ready()

    @pytest.mark.asyncio
    async def test_backend_manager_get_backend(self):
        """Test backend manager get_backend method."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        manager = BackendManager(backend)

        retrieved_backend = manager.get_backend()
        assert retrieved_backend is backend

    @pytest.mark.asyncio
    async def test_backend_manager_health_check(self):
        """Test backend manager health check."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        manager = BackendManager(backend)
        await manager.initialize()

        health = await manager.health_check()
        assert "status" in health
        assert "backend" in health

    @pytest.mark.asyncio
    async def test_backend_manager_info(self):
        """Test backend manager current backend info."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        manager = BackendManager(backend)
        await manager.initialize()

        info = manager.get_current_backend_info()
        assert "name" in info
        assert "model_name" in info
        assert "status" in info


class TestBackendBenchmarks:
    """Tests for backend benchmarking functionality."""

    @pytest.mark.asyncio
    async def test_backend_benchmark_creation(self):
        """Test backend benchmark creation."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        benchmark = BackendBenchmark(backend)
        assert benchmark.backend is backend

    @pytest.mark.asyncio
    async def test_backend_benchmark_single_run(self):
        """Test single benchmark run."""
        backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
        await backend.load_model()

        benchmark = BackendBenchmark(backend)
        result = await benchmark.run_single_benchmark(texts=["Test text for benchmarking"], batch_size=1)

        assert "processing_time" in result
        assert "throughput" in result
        assert result["processing_time"] > 0
        assert result["throughput"] > 0
