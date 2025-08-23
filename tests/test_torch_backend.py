"""
Basic test for TorchBackend functionality.
"""

import asyncio
import pytest
import numpy as np

from app.backends.torch_backend import TorchBackend


@pytest.mark.asyncio
async def test_torch_backend_basic():
    """Test basic TorchBackend functionality."""
    # Use a small, fast model for testing
    backend = TorchBackend("sentence-transformers/all-MiniLM-L6-v2")
    
    # Test model loading
    await backend.load_model()
    assert backend.is_loaded
    assert backend.load_time is not None
    assert backend.load_time > 0
    
    # Test embedding generation
    texts = ["Hello world", "How are you?", "This is a test"]
    result = await backend.embed_texts(texts, batch_size=2)
    
    assert result.vectors.shape[0] == len(texts)
    assert result.vectors.shape[1] > 0  # Should have some embedding dimension
    assert result.processing_time > 0
    assert result.device in ["cpu", "mps", "cuda"]
    assert result.model_info == "sentence-transformers/all-MiniLM-L6-v2"
    
    # Test similarity computation
    query_embedding = result.vectors[0]
    passage_embeddings = result.vectors[1:]
    
    similarities = await backend.compute_similarity(query_embedding, passage_embeddings)
    assert len(similarities) == len(passage_embeddings)
    assert all(isinstance(score, (float, np.floating)) for score in similarities)
    
    # Test model info
    model_info = backend.get_model_info()
    assert model_info["backend"] == "torch"
    assert model_info["is_loaded"] is True
    assert "embedding_dimension" in model_info
    
    # Test device info
    device_info = backend.get_device_info()
    assert device_info["backend"] == "torch"
    assert "torch_version" in device_info
    
    # Test health check
    health = await backend.health_check()
    assert health["status"] == "healthy"
    assert health["model_loaded"] is True


if __name__ == "__main__":
    # Simple test runner for development
    async def run_test():
        print("Testing TorchBackend...")
        await test_torch_backend_basic()
        print("✅ TorchBackend test passed!")
    
    asyncio.run(run_test())
