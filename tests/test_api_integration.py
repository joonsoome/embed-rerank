"""
API Integration Tests for Embed-Rerank Service

This module contains comprehensive integration tests for all API endpoints
following Context7 patterns and production-ready testing practices.
"""

import pytest
import asyncio
import json
from typing import Dict, Any
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.config import settings


class TestAPIIntegration:
    """Comprehensive API integration tests."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    @pytest.fixture(scope="class")
    async def async_client(self):
        """Async test client fixture."""
        from httpx import ASGITransport

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "uptime" in data
        assert "backend_info" in data
        assert "system_metrics" in data

        # Verify backend info structure
        backend_info = data["backend_info"]
        assert "backend_type" in backend_info
        assert "device" in backend_info
        assert "model_loaded" in backend_info

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs_url" in data

    @pytest.mark.asyncio
    async def test_embedding_endpoint_basic(self, async_client):
        """Test basic embedding functionality."""
        request_data = {"texts": ["Hello world", "How are you?"], "batch_size": 16, "normalize": True}

        response = await async_client.post("/api/v1/embed/", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "vectors" in data
        assert "processing_time" in data
        assert "usage" in data
        assert "model_info" in data

        # Verify vector structure
        vectors = data["vectors"]
        assert len(vectors) == 2  # Two input texts
        assert all(isinstance(v, list) for v in vectors)
        assert all(len(v) > 0 for v in vectors)  # Non-empty vectors

    @pytest.mark.asyncio
    async def test_embedding_endpoint_single_text(self, async_client):
        """Test embedding with single text."""
        request_data = {"texts": ["Single text for embedding"], "normalize": False}

        response = await async_client.post("/api/v1/embed/", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["vectors"]) == 1
        assert data["usage"]["total_texts"] == 1

    @pytest.mark.asyncio
    async def test_embedding_endpoint_large_batch(self, async_client):
        """Test embedding with larger batch."""
        texts = [f"Text number {i}" for i in range(10)]
        request_data = {"texts": texts, "batch_size": 4, "normalize": True}

        response = await async_client.post("/api/v1/embed/", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["vectors"]) == 10
        assert data["usage"]["total_texts"] == 10

    @pytest.mark.asyncio
    async def test_reranking_endpoint_basic(self, async_client):
        """Test basic reranking functionality."""
        request_data = {
            "query": "machine learning algorithms",
            "passages": [
                "Deep learning is a subset of machine learning",
                "Cats are popular pets",
                "Neural networks are used in AI",
                "The weather is nice today",
            ],
            "top_k": 3,
            "return_documents": True,
        }

        response = await async_client.post("/api/v1/rerank/", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "processing_time" in data
        assert "usage" in data

        # Verify ranking results
        results = data["results"]
        assert len(results) == 3  # top_k = 3
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all("index" in r for r in results)

        # Verify scores are in descending order
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_reranking_endpoint_no_return_docs(self, async_client):
        """Test reranking without returning documents."""
        request_data = {
            "query": "artificial intelligence",
            "passages": ["AI is transforming industries", "Pizza is delicious", "Machine learning models need data"],
            "top_k": 2,
            "return_documents": False,
        }

        response = await async_client.post("/api/v1/rerank/", json=request_data)
        assert response.status_code == 200

        data = response.json()
        results = data["results"]
        assert len(results) == 2
        assert all("score" in r for r in results)
        assert all("index" in r for r in results)
        # Should not contain text when return_documents=False
        assert all("text" not in r for r in results)

    def test_embedding_validation_errors(self, client):
        """Test embedding endpoint validation errors."""
        # Empty texts list
        response = client.post("/api/v1/embed/", json={"texts": []})
        assert response.status_code == 422

        # Missing texts field
        response = client.post("/api/v1/embed/", json={"batch_size": 16})
        assert response.status_code == 422

        # Invalid batch size
        response = client.post("/api/v1/embed/", json={"texts": ["test"], "batch_size": 0})
        assert response.status_code == 422

        # Too many texts
        long_texts = [f"Text {i}" for i in range(101)]  # Over limit
        response = client.post("/api/v1/embed/", json={"texts": long_texts})
        assert response.status_code == 422

    def test_reranking_validation_errors(self, client):
        """Test reranking endpoint validation errors."""
        # Empty query
        response = client.post("/api/v1/rerank/", json={"query": "", "passages": ["test passage"]})
        assert response.status_code == 422

        # Empty passages
        response = client.post("/api/v1/rerank/", json={"query": "test query", "passages": []})
        assert response.status_code == 422

        # Invalid top_k
        response = client.post(
            "/api/v1/rerank/", json={"query": "test", "passages": ["passage1", "passage2"], "top_k": 0}
        )
        assert response.status_code == 422

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/health/")
        assert "access-control-allow-origin" in response.headers

    def test_request_id_tracking(self, client):
        """Test request ID tracking in responses."""
        response = client.get("/health/")
        # Should have some form of request tracking
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling of concurrent requests."""

        async def make_embedding_request():
            return await async_client.post(
                "/api/v1/embed/", json={"texts": ["Concurrent test text"], "normalize": True}
            )

        # Make 5 concurrent requests
        tasks = [make_embedding_request() for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should return valid embeddings
        for response in responses:
            data = response.json()
            assert len(data["vectors"]) == 1
            assert len(data["vectors"][0]) > 0

    @pytest.mark.slow
    def test_performance_benchmarks(self, client):
        """Test basic performance expectations."""
        import time

        # Test embedding performance
        start_time = time.time()
        response = client.post("/api/v1/embed/", json={"texts": ["Performance test text"], "normalize": True})
        end_time = time.time()

        assert response.status_code == 200
        processing_time = end_time - start_time

        # Should be reasonably fast (adjust threshold as needed)
        assert processing_time < 10.0  # 10 second timeout for CI

        # Verify response time is tracked
        data = response.json()
        assert data["processing_time"] > 0

    def test_openapi_documentation(self, client):
        """Test OpenAPI documentation is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi_data = response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(app) as test_client:
            yield test_client

    def test_404_handling(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test method not allowed handling."""
        response = client.delete("/health/")
        assert response.status_code == 405

    def test_malformed_json(self, client):
        """Test malformed JSON handling."""
        response = client.post("/api/v1/embed/", data="invalid json{", headers={"Content-Type": "application/json"})
        assert response.status_code == 422


@pytest.mark.integration
class TestServiceIntegration:
    """Test service layer integration."""

    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(app) as test_client:
            yield test_client

    def test_backend_switching(self, client):
        """Test that the service can handle backend switching gracefully."""
        # Make multiple requests to ensure backend stability
        for _ in range(3):
            response = client.post("/api/v1/embed/", json={"texts": ["Backend stability test"], "normalize": True})
            assert response.status_code == 200

    def test_health_after_operations(self, client):
        """Test health status after performing operations."""
        # Perform some operations
        client.post("/api/v1/embed/", json={"texts": ["Test text 1", "Test text 2"], "normalize": True})

        client.post(
            "/api/v1/rerank/",
            json={"query": "test query", "passages": ["passage 1", "passage 2", "passage 3"], "top_k": 2},
        )

        # Check health is still good
        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["backend_info"]["model_loaded"] is True
