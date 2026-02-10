from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _FakeEmbeddingService:
    def __init__(self, vectors: Optional[List[List[float]]] = None):
        self.last_request = None
        self._vectors = vectors or [[0.0, 1.0, 2.0, 3.0]]

    async def embed_texts(self, request):  # noqa: ANN001
        from app.models.responses import EmbedResponse, EmbeddingVector

        self.last_request = request
        vectors = self._vectors
        embeddings = [EmbeddingVector(embedding=vectors[0], index=0, text="x")]
        return EmbedResponse(
            embeddings=embeddings,
            vectors=vectors,
            dim=len(vectors[0]),
            backend="fake",
            device="cpu",
            processing_time=0.0,
            model_info="fake",
            usage={"total_texts": 1, "total_tokens": 1},
            timestamp=datetime.now(),
            num_texts=1,
        )


@pytest.fixture
def client_and_service():
    from app.routers import openai_router

    app = FastAPI()
    app.include_router(openai_router.router)

    openai_router.set_backend_manager(object())
    svc = _FakeEmbeddingService()
    openai_router.set_embedding_service(svc)

    with TestClient(app) as client:
        yield client, svc, openai_router


def test_openai_embeddings_maps_max_tokens_per_text_to_internal_override(client_and_service):
    client, svc, _openai_router = client_and_service
    resp = client.post(
        "/v1/embeddings",
        json={"input": "hello", "model": "text-embedding-ada-002", "max_tokens_per_text": 123},
    )
    assert resp.status_code == 200
    assert svc.last_request is not None
    assert getattr(svc.last_request, "max_tokens_override", None) == 123


def test_openai_embeddings_dimensions_gate_respected(client_and_service, monkeypatch):
    client, svc, openai_router = client_and_service

    # Disable honoring `dimensions` and ensure vector size is unchanged.
    monkeypatch.setattr(openai_router.app_settings, "embedding_send_dim", False, raising=False)
    resp = client.post("/v1/embeddings", json={"input": "hello", "model": "text-embedding-ada-002", "dimensions": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"][0]["embedding"]) == 4

    # Enable honoring `dimensions` and ensure vector is truncated/padded.
    monkeypatch.setattr(openai_router.app_settings, "embedding_send_dim", True, raising=False)
    resp2 = client.post("/v1/embeddings", json={"input": "hello", "model": "text-embedding-ada-002", "dimensions": 2})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert len(data2["data"][0]["embedding"]) == 2
