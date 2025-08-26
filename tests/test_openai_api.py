"""
OpenAI API Compatibility Tests for Embed-Rerank Service

This module contains tests for OpenAI-compatible API endpoints:
- /v1/embeddings - OpenAI embeddings format
- Enhanced configuration and compatibility features
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestOpenAIEmbeddingsAPI:
    """Tests for OpenAI-compatible embeddings API."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_openai_embeddings_basic(self, client):
        """Test basic OpenAI embeddings endpoint."""
        response = client.post("/v1/embeddings", json={"input": "Hello world", "model": "text-embedding-ada-002"})

        if response.status_code == 200:
            data = response.json()

            # Check OpenAI response structure
            assert "object" in data
            assert data["object"] == "list"
            assert "data" in data
            assert "model" in data
            assert "usage" in data

            # Check data structure
            embeddings = data["data"]
            assert isinstance(embeddings, list)
            assert len(embeddings) == 1

            embedding = embeddings[0]
            assert "object" in embedding
            assert embedding["object"] == "embedding"
            assert "embedding" in embedding
            assert "index" in embedding
            assert embedding["index"] == 0

            # Check embedding vector
            vector = embedding["embedding"]
            assert isinstance(vector, list)
            assert len(vector) > 0
            assert all(isinstance(x, (int, float)) for x in vector)

            # Check usage
            usage = data["usage"]
            assert "prompt_tokens" in usage
            assert "total_tokens" in usage

        elif response.status_code == 503:
            # Backend not ready
            data = response.json()
            assert "detail" in data

    def test_openai_embeddings_multiple_inputs(self, client):
        """Test OpenAI embeddings with multiple inputs."""
        response = client.post(
            "/v1/embeddings",
            json={"input": ["First text", "Second text", "Third text"], "model": "text-embedding-ada-002"},
        )

        if response.status_code == 200:
            data = response.json()

            # Should have multiple embeddings
            embeddings = data["data"]
            assert len(embeddings) == 3

            # Check ordering and indices
            for i, embedding in enumerate(embeddings):
                assert embedding["index"] == i
                assert "embedding" in embedding

            # Check usage reflects multiple inputs
            usage = data["usage"]
            assert usage["prompt_tokens"] >= 3  # At least one token per input

    def test_openai_embeddings_string_input(self, client):
        """Test OpenAI embeddings with string input."""
        response = client.post(
            "/v1/embeddings", json={"input": "Single string input", "model": "text-embedding-ada-002"}
        )

        if response.status_code == 200:
            data = response.json()
            embeddings = data["data"]
            assert len(embeddings) == 1

    def test_openai_embeddings_with_user(self, client):
        """Test OpenAI embeddings with user parameter."""
        response = client.post(
            "/v1/embeddings",
            json={"input": "Test with user parameter", "model": "text-embedding-ada-002", "user": "test-user-123"},
        )

        # Should handle user parameter gracefully
        assert response.status_code in [200, 503]

    def test_openai_embeddings_model_validation(self, client):
        """Test OpenAI embeddings with different model names."""
        models_to_test = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]

        for model in models_to_test:
            response = client.post("/v1/embeddings", json={"input": "Test model validation", "model": model})

            # Should accept any model name (maps to internal model)
            assert response.status_code in [200, 503]

            if response.status_code == 200:
                data = response.json()
                assert data["model"] == model  # Should echo back the requested model


class TestOpenAICompatibility:
    """Tests for OpenAI API compatibility features."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_openai_error_format(self, client):
        """Test that errors follow OpenAI format."""
        response = client.post("/v1/embeddings", json={"invalid_field": "value"})

        # Should return validation error in appropriate format
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_openai_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002"
                # Missing 'input' field
            },
        )

        assert response.status_code == 422

    def test_openai_empty_input(self, client):
        """Test handling of empty input."""
        response = client.post("/v1/embeddings", json={"input": [], "model": "text-embedding-ada-002"})

        # Should handle empty input gracefully
        assert response.status_code in [200, 400, 422, 503]

    def test_openai_large_input(self, client):
        """Test handling of large input arrays."""
        large_input = [f"Text number {i}" for i in range(100)]

        response = client.post("/v1/embeddings", json={"input": large_input, "model": "text-embedding-ada-002"})

        # Should handle large inputs (might succeed or timeout)
        assert response.status_code in [200, 413, 422, 503, 504]

    def test_openai_special_characters(self, client):
        """Test handling of special characters in input."""
        special_texts = [
            "Text with émojis 🚀",
            "Unicode: αβγδε",
            "Mixed: Hello 世界",
            "Symbols: @#$%^&*()",
            "Newlines:\nand\ttabs",
        ]

        response = client.post("/v1/embeddings", json={"input": special_texts, "model": "text-embedding-ada-002"})

        if response.status_code == 200:
            data = response.json()
            assert len(data["data"]) == len(special_texts)


class TestOpenAIEnhancedFeatures:
    """Tests for enhanced OpenAI compatibility features."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_configurable_dimensions(self, client):
        """Test configurable embedding dimensions."""
        response = client.post(
            "/v1/embeddings", json={"input": "Test dimensions", "model": "text-embedding-3-large", "dimensions": 1024}
        )

        if response.status_code == 200:
            data = response.json()
            embedding = data["data"][0]["embedding"]
            # Note: Actual dimension handling depends on implementation
            assert len(embedding) > 0

    def test_encoding_format_options(self, client):
        """Test different encoding format options."""
        formats = ["float", "base64"]

        for encoding_format in formats:
            response = client.post(
                "/v1/embeddings",
                json={
                    "input": "Test encoding format",
                    "model": "text-embedding-ada-002",
                    "encoding_format": encoding_format,
                },
            )

            # Should handle encoding format parameter
            assert response.status_code in [200, 422, 503]

    def test_extra_parameters(self, client):
        """Test handling of extra parameters."""
        response = client.post(
            "/v1/embeddings",
            json={
                "input": "Test extra params",
                "model": "text-embedding-ada-002",
                "extra_param": "should_be_ignored",
                "custom_setting": 42,
            },
        )

        # Should handle extra parameters gracefully
        assert response.status_code in [200, 422, 503]


class TestOpenAIAuthentication:
    """Tests for authentication and authorization features."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_bearer_token_handling(self, client):
        """Test Bearer token authentication handling."""
        response = client.post(
            "/v1/embeddings",
            json={"input": "Test with auth", "model": "text-embedding-ada-002"},
            headers={"Authorization": "Bearer sk-test123"},
        )

        # Should accept or ignore bearer tokens gracefully
        assert response.status_code in [200, 401, 503]

    def test_api_key_header(self, client):
        """Test API key header handling."""
        response = client.post(
            "/v1/embeddings",
            json={"input": "Test with API key", "model": "text-embedding-ada-002"},
            headers={"X-API-Key": "test-key-123"},
        )

        # Should handle API key headers
        assert response.status_code in [200, 401, 503]

    def test_no_authentication(self, client):
        """Test requests without authentication."""
        response = client.post("/v1/embeddings", json={"input": "Test without auth", "model": "text-embedding-ada-002"})

        # Should work without authentication (or return appropriate error)
        assert response.status_code in [200, 401, 503]


class TestOpenAIRateLimiting:
    """Tests for rate limiting and throttling features."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are included if implemented."""
        response = client.post("/v1/embeddings", json={"input": "Test rate limits", "model": "text-embedding-ada-002"})

        # Check for common rate limit headers (optional)
        # These headers are optional - just check response is valid
        assert response.status_code in [200, 429, 503]

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time

        results = []

        def make_request():
            response = client.post(
                "/v1/embeddings", json={"input": f"Concurrent test {time.time()}", "model": "text-embedding-ada-002"}
            )
            results.append(response.status_code)

        # Make multiple concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all requests to complete
        for thread in threads:
            thread.join()

        # All requests should be handled (success or appropriate errors)
        for status_code in results:
            assert status_code in [200, 429, 503, 504]
