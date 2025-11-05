"""
Tests that the reranking endpoint uses a dedicated reranker backend when provided,
and returns expected metadata (method/backend/model_info) without requiring network downloads.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backends.base import BaseBackend, BackendManager, EmbeddingResult
from app.routers import reranking_router
from app.main import app


class FakeRerankerBackend(BaseBackend):
    """Minimal fake cross-encoder backend for tests (deterministic)."""

    def __init__(self, model_name: str = "fake-cross-encoder", device: Optional[str] = None):
        super().__init__(model_name, device or "cpu")

    async def load_model(self) -> None:
        self._is_loaded = True
        self._load_time = 0.0

    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        # Return 1-dim zero vectors; not used in this test path
        vectors = np.zeros((len(texts), 1), dtype=np.float32)
        return EmbeddingResult(vectors=vectors, processing_time=0.0, device=self.device, model_info=self.model_name)

    async def compute_similarity(self, query_embedding: np.ndarray, passage_embeddings: np.ndarray) -> np.ndarray:  # type: ignore[name-defined]
        return np.zeros((passage_embeddings.shape[0],), dtype=np.float32)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "rerank_method": "cross-encoder",
            "rerank_model_name": self.model_name,
        }

    def get_device_info(self) -> Dict[str, Any]:
        return {"device": self.device}

    async def rerank_passages(self, query: str, passages: List[str]) -> List[float]:
        # Deterministic descending scores in [0,1]
        return [max(0.0, 1.0 - 0.1 * i) for i in range(len(passages))]


@pytest.fixture(scope="module")
def client_with_fake_reranker():
    # Install a dedicated reranker backend manager so router prefers it
    backend = FakeRerankerBackend()
    manager = BackendManager(backend)

    # Initialize (loads backend)
    import asyncio

    loop = asyncio.new_event_loop()
    loop.run_until_complete(manager.initialize())
    loop.close()

    reranking_router.set_reranker_backend_manager(manager)

    with TestClient(app) as c:
        yield c


def test_rerank_uses_dedicated_backend(client_with_fake_reranker: TestClient):
    payload = {
        "query": "what is machine learning?",
        "passages": [
            "Machine learning is a subset of AI",
            "Dogs are not relevant",
            "Deep learning uses neural networks",
        ],
        "top_k": 3,
        "return_documents": True,
    }

    resp = client_with_fake_reranker.post("/api/v1/rerank/", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Check response metadata reflects dedicated backend info
    assert data.get("method") == "cross-encoder"
    assert data.get("backend") == "FakeRerankerBackend"
    assert data.get("num_passages") == 3

    # Result ordering should match our deterministic descending scores
    results = data.get("results", [])
    assert len(results) == 3
    # Highest score should be index 0 per our scoring
    assert results[0]["index"] == 0
    # Next should be index 1
    assert results[1]["index"] == 1
    # Then index 2
    assert results[2]["index"] == 2
