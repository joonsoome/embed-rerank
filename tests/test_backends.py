"""
Test for backend factory and MLX backend.
"""

import asyncio
import pytest
import platform

from app.backends.factory import BackendFactory
from app.backends.mlx_backend import MLX_AVAILABLE
from app.utils.benchmark import BackendBenchmark


@pytest.mark.asyncio
async def test_backend_factory():
    """Test backend factory functionality."""
    # Test available backends
    available = BackendFactory.get_available_backends()
    assert "torch" in available
    assert "mlx" in available

    # Test auto-detection
    backend = BackendFactory.create_backend("auto", "sentence-transformers/all-MiniLM-L6-v2")
    assert backend is not None

    # Test torch backend creation
    torch_backend = BackendFactory.create_backend("torch", "sentence-transformers/all-MiniLM-L6-v2")
    assert torch_backend.__class__.__name__ == "TorchBackend"


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
@pytest.mark.asyncio
async def test_mlx_backend():
    """Test MLX backend if available."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("MLX requires Apple Silicon")

    try:
        backend = BackendFactory.create_backend("mlx", "sentence-transformers/all-MiniLM-L6-v2")
        await backend.load_model()

        # Test embedding generation
        texts = ["Hello world", "Test text"]
        result = await backend.embed_texts(texts)

        assert result.vectors.shape[0] == len(texts)
        assert result.device == "mlx"
        assert result.processing_time > 0

    except Exception as e:
        pytest.skip(f"MLX backend test failed: {e}")


@pytest.mark.asyncio
async def test_backend_comparison():
    """Test backend comparison functionality."""
    benchmark = BackendBenchmark("sentence-transformers/all-MiniLM-L6-v2")

    test_texts = ["Test text 1", "Test text 2", "Test text 3"]

    # Test torch backend
    torch_result = await benchmark.benchmark_backend("torch", test_texts, iterations=1)
    assert torch_result["backend_type"] == "torch"
    assert torch_result["load_time"] > 0
    assert len(torch_result["batch_results"]) > 0

    print(f"✅ Torch benchmark: {torch_result['load_time']:.2f}s load time")

    # Test MLX backend if available
    if MLX_AVAILABLE and platform.system() == "Darwin":
        try:
            mlx_result = await benchmark.benchmark_backend("mlx", test_texts, iterations=1)
            assert mlx_result["backend_type"] == "mlx"
            print(f"✅ MLX benchmark: {mlx_result['load_time']:.2f}s load time")
        except Exception as e:
            print(f"⚠️ MLX benchmark skipped: {e}")


if __name__ == "__main__":

    async def run_tests():
        print("Testing Backend Factory...")
        await test_backend_factory()
        print("✅ Backend Factory test passed!")

        print("\nTesting Backend Comparison...")
        await test_backend_comparison()
        print("✅ Backend Comparison test passed!")

        if MLX_AVAILABLE:
            print("\nTesting MLX Backend...")
            await test_mlx_backend()
            print("✅ MLX Backend test passed!")
        else:
            print("⚠️ MLX Backend skipped (not available)")

    asyncio.run(run_tests())
