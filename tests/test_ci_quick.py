"""
Quick CI tests - lightweight tests for GitHub Actions
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestQuickCI:
    """Lightweight tests for CI environment"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_app_creation(self):
        """Test that the FastAPI app can be created"""
        assert app is not None
        assert app.title == "🚀 Apple MLX Embed-Rerank API"

    def test_health_endpoint_basic(self, client):
        """Test basic health endpoint functionality"""
        response = client.get("/health/")
        # In CI without backend initialization, expect 503
        assert response.status_code in [200, 503]

        data = response.json()
        # Check for either proper health response or error response
        assert "status" in data or "error" in data

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "🚀 Apple MLX Embed-Rerank API"

    def test_openapi_docs(self, client):
        """Test OpenAPI documentation endpoint"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "info" in data

    def test_embed_endpoint_exists(self, client):
        """Test that embed endpoint exists"""
        response = client.post("/api/v1/embed", json={})
        # Should return 422 (validation error) or 503 (backend not ready)
        assert response.status_code in [422, 503]

    def test_openai_models_endpoint(self, client):
        """Test OpenAI models endpoint"""
        response = client.get("/v1/models")
        # Accept both 200 (backend ready) and 503 (backend not ready)
        assert response.status_code in [200, 503]
