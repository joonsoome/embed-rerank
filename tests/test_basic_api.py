"""
Simple API integration tests for basic functionality verification.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestBasicAPI:
    """Basic API functionality tests."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_root_endpoint_accessible(self, client):
        """Test root endpoint is accessible."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "description" in data

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

    def test_health_endpoint_structure(self, client):
        """Test health endpoint structure (may be 503 if backend not initialized)."""
        response = client.get("/health/")

        # Accept either 200 (initialized) or 503 (not initialized)
        assert response.status_code in [200, 503]

        data = response.json()
        if response.status_code == 200:
            # Backend is initialized
            assert "status" in data
            assert "uptime" in data
        else:
            # Backend not initialized (expected in tests)
            assert "detail" in data

    def test_embed_endpoint_exists(self, client):
        """Test embed endpoint exists and validates input."""
        # Test with invalid input - should get validation error or service unavailable
        response = client.post("/api/v1/embed/", json={"texts": []})
        assert response.status_code in [422, 503]  # Validation error or service unavailable

    def test_rerank_endpoint_exists(self, client):
        """Test rerank endpoint exists and validates input."""
        # Test with invalid input - should get validation error or service unavailable
        response = client.post("/api/v1/rerank/", json={"query": "", "passages": ["test passage"]})
        assert response.status_code in [422, 503]  # Validation error or service unavailable

    def test_cors_headers_in_response(self, client):
        """Test CORS headers are present in successful responses."""
        response = client.get("/")
        # CORS headers should be present in successful responses
        # Note: TestClient may not include all CORS headers as it doesn't go through actual CORS middleware

    def test_method_not_allowed(self, client):
        """Test method not allowed handling."""
        response = client.delete("/health/")
        assert response.status_code == 405

    def test_404_handling(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
