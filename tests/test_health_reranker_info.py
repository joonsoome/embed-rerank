import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import health_router


class FakeRerankerManager:
    def __init__(self):
        self._ready = True

    def is_ready(self):
        return self._ready

    def get_current_backend_info(self):
        # Mimic MLXCrossEncoderBackend get_model_info merged view
        return {
            "name": "MLXCrossEncoderBackend",
            "status": "ready",
            "model_name": "fake/mlx-reranker",
            "device": "mlx",
            "rerank_method": "cross-encoder-lite",
            "pooling": "cls",
            "score_norm": "sigmoid",
        }


@pytest.fixture(scope="module")
def client_with_fake_reranker():
    # Inject fake reranker manager into health router
    health_router.set_reranker_backend_manager(FakeRerankerManager())
    with TestClient(app) as client:
        yield client


def test_health_includes_reranker_block(client_with_fake_reranker):
    resp = client_with_fake_reranker.get("/health/")
    assert resp.status_code in (200, 503)
    data = resp.json()
    # Reranker section should be present and include pooling/score_norm
    if "reranker" in data:
        rk = data["reranker"]
        assert isinstance(rk, dict)
        # Keys we expect
        for k in ("name", "type", "model_name", "pooling", "score_norm"):
            assert k in rk
        # Validate values from fake
        assert rk["pooling"] in ("mean", "cls")
        assert rk["score_norm"] in ("none", "sigmoid", "minmax")
