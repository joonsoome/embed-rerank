"""
Native API Tests for Embed-Rerank Service

This module contains tests for the native API endpoints:
- /api/v1/embed - Text embeddings
- /api/v1/rerank - Document reranking
- /health - Health checks and system status
- / - Root endpoint and documentation
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestNativeAPI:
    """Tests for native API endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_root_endpoint(self, client):
        """Test root endpoint accessibility."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "description" in data
        assert "version" in data

    def test_openapi_documentation(self, client):
        """Test OpenAPI documentation accessibility."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_documentation(self, client):
        """Test ReDoc documentation accessibility."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        """Test OpenAPI schema accessibility."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data


class TestHealthAPI:
    """Tests for health check endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_health_endpoint(self, client):
        """Test health endpoint functionality."""
        response = client.get("/health/")

        # Accept both ready and not-ready states
        assert response.status_code in [200, 503]
        data = response.json()

        # Basic structure validation
        assert "status" in data
        assert "timestamp" in data
        assert "service" in data

        if response.status_code == 200:
            # If ready, should have backend info
            assert data["status"] in ["healthy", "ready"]
            assert "backend" in data
        else:
            # If not ready, should indicate initialization
            assert data["status"] in ["initializing", "not_ready"]

    def test_health_endpoint_structure(self, client):
        """Test health endpoint response structure."""
        response = client.get("/health/")
        data = response.json()

        # Required fields
        required_fields = ["status", "timestamp", "service"]
        for field in required_fields:
            assert field in data
            assert data[field] is not None

        # Service info structure
        assert isinstance(data["service"], dict)
        service_info = data["service"]
        service_required = ["name", "version"]
        for field in service_required:
            assert field in service_info


class TestEmbeddingAPI:
    """Tests for native embedding API."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_embedding_endpoint_format(self, client):
        """Test embedding endpoint request/response format."""
        response = client.post(
            "/api/v1/embed/", json={"texts": ["Hello world", "Testing embeddings"], "normalize": True}
        )

        # Accept both success and not-ready responses
        if response.status_code == 200:
            data = response.json()

            # Check response structure
            assert "vectors" in data
            assert "processing_time" in data
            assert "num_texts" in data

            # Check vectors structure
            vectors = data["vectors"]
            assert isinstance(vectors, list)
            assert len(vectors) == 2  # Two input texts

            # Each vector should be a list of numbers
            for vector in vectors:
                assert isinstance(vector, list)
                assert len(vector) > 0
                assert all(isinstance(x, (int, float)) for x in vector)

        elif response.status_code == 503:
            # Backend not ready - acceptable for format testing
            data = response.json()
            assert "detail" in data
            assert "backend" in data["detail"].lower() or "ready" in data["detail"].lower()

    def test_embedding_single_text(self, client):
        """Test embedding endpoint with single text."""
        response = client.post("/api/v1/embed/", json={"texts": ["Single text input"]})

        if response.status_code == 200:
            data = response.json()
            assert len(data["vectors"]) == 1
            assert data["num_texts"] == 1

    def test_embedding_empty_input_handling(self, client):
        """Test embedding endpoint with empty input."""
        response = client.post("/api/v1/embed/", json={"texts": []})

        # Should handle gracefully with validation error or empty result
        assert response.status_code in [200, 400, 422, 503]

    def test_embedding_invalid_input(self, client):
        """Test embedding endpoint with invalid input."""
        response = client.post("/api/v1/embed/", json={"invalid_field": ["text"]})

        # Should return validation error
        assert response.status_code == 422


class TestRerankingAPI:
    """Tests for native reranking API."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_reranking_endpoint_format(self, client):
        """Test reranking endpoint request/response format."""
        response = client.post(
            "/api/v1/rerank/",
            json={
                "query": "What is machine learning?",
                "passages": [
                    "Machine learning is a subset of artificial intelligence",
                    "Dogs are pets that need care and attention",
                    "Neural networks are used in deep learning",
                ],
            },
        )

        # Accept both success and not-ready responses
        if response.status_code == 200:
            data = response.json()

            # Check response structure
            assert "results" in data
            assert "processing_time" in data
            assert "num_passages" in data

            # Check results structure
            results = data["results"]
            assert isinstance(results, list)
            assert len(results) == 3  # Three input passages

            # Each result should have required fields
            for result in results:
                assert "index" in result
                assert "score" in result
                assert "text" in result
                assert isinstance(result["index"], int)
                assert isinstance(result["score"], (int, float))
                assert isinstance(result["text"], str)

        elif response.status_code == 503:
            # Backend not ready - acceptable for format testing
            data = response.json()
            assert "detail" in data

    def test_reranking_single_passage(self, client):
        """Test reranking endpoint with single passage."""
        response = client.post("/api/v1/rerank/", json={"query": "test query", "passages": ["Single passage to rank"]})

        if response.status_code == 200:
            data = response.json()
            assert len(data["results"]) == 1
            assert data["num_passages"] == 1

    def test_reranking_empty_passages(self, client):
        """Test reranking endpoint with empty passages."""
        response = client.post("/api/v1/rerank/", json={"query": "test query", "passages": []})

        # Should handle gracefully
        assert response.status_code in [200, 400, 422, 503]

    def test_reranking_invalid_input(self, client):
        """Test reranking endpoint with invalid input."""
        response = client.post("/api/v1/rerank/", json={"invalid_field": "value"})

        # Should return validation error
        assert response.status_code == 422


class TestAPIErrorHandling:
    """Tests for API error handling and edge cases."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_nonexistent_endpoint(self, client):
        """Test request to nonexistent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test wrong HTTP method on endpoint."""
        response = client.get("/api/v1/embed/")  # Should be POST
        assert response.status_code == 405

    def test_malformed_json(self, client):
        """Test malformed JSON request."""
        response = client.post("/api/v1/embed/", data="invalid json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422

    def test_missing_content_type(self, client):
        """Test request without proper content type."""
        response = client.post("/api/v1/embed/", data='{"texts": ["test"]}')
        # Should either work or return appropriate error
        assert response.status_code in [200, 400, 422, 503]
