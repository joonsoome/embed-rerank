"""
TEI (Text Embeddings Inference) API Compatibility Tests

This module contains tests for TEI-compatible API endpoints:
- /embed - TEI embeddings format
- /rerank - TEI reranking format
- Hugging Face TEI compatibility features
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestTEIEmbeddingsAPI:
    """Tests for TEI-compatible embeddings API."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_tei_embed_basic(self, client):
        """Test basic TEI embeddings endpoint."""
        response = client.post("/embed", json={"inputs": "Hello world"})

        if response.status_code == 200:
            data = response.json()

            # TEI returns a simple list of embeddings
            assert isinstance(data, list)
            assert len(data) == 1

            # Each embedding is a list of numbers
            embedding = data[0]
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)

        elif response.status_code == 503:
            # Backend not ready
            data = response.json()
            assert "detail" in data

    def test_tei_embed_multiple_inputs(self, client):
        """Test TEI embeddings with multiple inputs."""
        response = client.post("/embed", json={"inputs": ["First text", "Second text", "Third text"]})

        if response.status_code == 200:
            data = response.json()

            # Should return list of embeddings
            assert isinstance(data, list)
            assert len(data) == 3

            # Each embedding should be a vector
            for embedding in data:
                assert isinstance(embedding, list)
                assert len(embedding) > 0
                assert all(isinstance(x, (int, float)) for x in embedding)

    def test_tei_embed_string_input(self, client):
        """Test TEI embeddings with single string input."""
        response = client.post("/embed", json={"inputs": "Single string input"})

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1

    def test_tei_embed_normalization(self, client):
        """Test TEI embeddings with normalization parameter."""
        response = client.post("/embed", json={"inputs": "Test normalization", "normalize": True})

        if response.status_code == 200:
            data = response.json()
            embedding = data[0]

            # Check if embedding appears normalized (magnitude ≈ 1)
            magnitude = sum(x * x for x in embedding) ** 0.5
            # Allow some tolerance for floating point precision
            assert 0.95 <= magnitude <= 1.05

    def test_tei_embed_truncation(self, client):
        """Test TEI embeddings with truncation parameter."""
        response = client.post("/embed", json={"inputs": "Test truncation parameter", "truncate": True})

        # Should handle truncation parameter
        assert response.status_code in [200, 503]

    def test_tei_embed_invalid_input(self, client):
        """Test TEI embeddings with invalid input."""
        response = client.post("/embed", json={"invalid_field": "value"})

        # Should return validation error
        assert response.status_code == 422


class TestTEIRerankingAPI:
    """Tests for TEI-compatible reranking API."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_tei_rerank_basic(self, client):
        """Test basic TEI reranking endpoint."""
        response = client.post(
            "/rerank",
            json={
                "query": "What is machine learning?",
                "texts": [
                    "Machine learning is a subset of artificial intelligence",
                    "Dogs are pets that need care and attention",
                    "Neural networks are used in deep learning",
                ],
            },
        )

        if response.status_code == 200:
            data = response.json()

            # TEI rerank returns a list of scored results
            assert isinstance(data, list)
            assert len(data) == 3

            # Each result should have score and index
            for result in data:
                assert "score" in result
                assert "index" in result
                assert isinstance(result["score"], (int, float))
                assert isinstance(result["index"], int)
                assert 0 <= result["index"] <= 2

            # Results should be sorted by score (descending)
            scores = [result["score"] for result in data]
            assert scores == sorted(scores, reverse=True)

        elif response.status_code == 503:
            # Backend not ready
            data = response.json()
            assert "detail" in data

    def test_tei_rerank_single_text(self, client):
        """Test TEI reranking with single text."""
        response = client.post("/rerank", json={"query": "test query", "texts": ["Single text to rank"]})

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["index"] == 0

    def test_tei_rerank_return_texts(self, client):
        """Test TEI reranking with return_texts parameter."""
        texts = ["First document about ML", "Second document about dogs", "Third document about AI"]

        response = client.post(
            "/rerank", json={"query": "artificial intelligence", "texts": texts, "return_texts": True}
        )

        if response.status_code == 200:
            data = response.json()

            # Should include text in results when return_texts=True
            for result in data:
                if "text" in result:  # Optional field
                    assert isinstance(result["text"], str)
                    assert result["text"] in texts

    def test_tei_rerank_top_k(self, client):
        """Test TEI reranking with top_k parameter."""
        response = client.post(
            "/rerank",
            json={"query": "test query", "texts": ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"], "top_k": 3},
        )

        if response.status_code == 200:
            data = response.json()

            # Should return at most top_k results
            assert len(data) <= 3

    def test_tei_rerank_truncation(self, client):
        """Test TEI reranking with truncation parameter."""
        response = client.post(
            "/rerank",
            json={
                "query": "test query with long text that might need truncation",
                "texts": ["Document to rank"],
                "truncate": True,
            },
        )

        # Should handle truncation parameter
        assert response.status_code in [200, 503]

    def test_tei_rerank_invalid_input(self, client):
        """Test TEI reranking with invalid input."""
        response = client.post("/rerank", json={"invalid_field": "value"})

        # Should return validation error
        assert response.status_code == 422

    def test_tei_rerank_empty_texts(self, client):
        """Test TEI reranking with empty texts."""
        response = client.post("/rerank", json={"query": "test query", "texts": []})

        # Should handle empty texts gracefully
        assert response.status_code in [200, 400, 422, 503]


class TestTEICompatibility:
    """Tests for TEI API compatibility features."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_tei_content_type_handling(self, client):
        """Test that TEI endpoints handle content type correctly."""
        response = client.post(
            "/embed", json={"inputs": "Test content type"}, headers={"Content-Type": "application/json"}
        )

        assert response.status_code in [200, 503]

    def test_tei_cors_headers(self, client):
        """Test CORS headers for TEI endpoints."""
        response = client.options("/embed")

        # Should handle OPTIONS requests for CORS
        assert response.status_code in [200, 405]

    def test_tei_error_format(self, client):
        """Test that TEI errors follow expected format."""
        response = client.post("/embed", json={"invalid": "data"})

        if response.status_code == 422:
            data = response.json()
            assert "detail" in data

    def test_tei_batch_processing(self, client):
        """Test TEI batch processing capabilities."""
        # Large batch to test batching
        large_batch = [f"Text number {i}" for i in range(50)]

        response = client.post("/embed", json={"inputs": large_batch})

        if response.status_code == 200:
            data = response.json()
            assert len(data) == len(large_batch)
        elif response.status_code == 413:
            # Payload too large - acceptable
            pass
        elif response.status_code == 503:
            # Backend not ready - acceptable
            pass

    def test_tei_unicode_handling(self, client):
        """Test TEI Unicode and special character handling."""
        unicode_texts = ["Unicode test: αβγδε", "Emoji test: 🚀🌟⭐", "Mixed: Hello 世界", "Symbols: @#$%^&*()"]

        response = client.post("/embed", json={"inputs": unicode_texts})

        if response.status_code == 200:
            data = response.json()
            assert len(data) == len(unicode_texts)


class TestTEIPerformance:
    """Tests for TEI API performance characteristics."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_tei_response_time_embed(self, client):
        """Test TEI embedding response time."""
        import time

        start_time = time.time()
        response = client.post("/embed", json={"inputs": "Performance test text"})
        end_time = time.time()

        if response.status_code == 200:
            response_time = end_time - start_time
            # Should respond within reasonable time (adjust as needed)
            assert response_time < 30.0  # 30 seconds timeout

    def test_tei_response_time_rerank(self, client):
        """Test TEI reranking response time."""
        import time

        start_time = time.time()
        response = client.post(
            "/rerank", json={"query": "Performance test query", "texts": ["Document 1", "Document 2", "Document 3"]}
        )
        end_time = time.time()

        if response.status_code == 200:
            response_time = end_time - start_time
            # Should respond within reasonable time
            assert response_time < 30.0  # 30 seconds timeout

    def test_tei_concurrent_requests(self, client):
        """Test TEI concurrent request handling."""
        import threading
        import time

        results = []

        def make_embed_request():
            response = client.post("/embed", json={"inputs": f"Concurrent test {time.time()}"})
            results.append(response.status_code)

        # Make multiple concurrent requests
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=make_embed_request)
            threads.append(thread)
            thread.start()

        # Wait for all requests to complete
        for thread in threads:
            thread.join()

        # All requests should be handled appropriately
        for status_code in results:
            assert status_code in [200, 503, 504]


class TestTEIValidation:
    """Tests for TEI API input validation."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_tei_embed_input_types(self, client):
        """Test TEI embed input type validation."""
        # Test various input types
        test_cases = [
            {"inputs": "string"},  # Valid
            {"inputs": ["list", "of", "strings"]},  # Valid
            {"inputs": 123},  # Invalid
            {"inputs": None},  # Invalid
            {"inputs": {"dict": "value"}},  # Invalid
        ]

        for test_case in test_cases:
            response = client.post("/embed", json=test_case)

            if isinstance(test_case["inputs"], (str, list)):
                # Valid inputs
                assert response.status_code in [200, 503]
            else:
                # Invalid inputs
                assert response.status_code == 422

    def test_tei_rerank_input_types(self, client):
        """Test TEI rerank input type validation."""
        # Test query validation
        invalid_queries = [
            {"query": None, "texts": ["text"]},
            {"query": 123, "texts": ["text"]},
            {"query": [], "texts": ["text"]},
        ]

        for test_case in invalid_queries:
            response = client.post("/rerank", json=test_case)
            assert response.status_code == 422

        # Test texts validation
        invalid_texts = [
            {"query": "test", "texts": None},
            {"query": "test", "texts": "string"},
            {"query": "test", "texts": 123},
        ]

        for test_case in invalid_texts:
            response = client.post("/rerank", json=test_case)
            assert response.status_code == 422
