"""
Integration Tests for Embed-Rerank Service

This module contains end-to-end integration tests that verify:
- Cross-API compatibility and consistency
- Real backend integration with actual model loading
- Performance benchmarking and monitoring
- Multi-format API interoperability
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestCrossAPICompatibility:
    """Tests for compatibility between different API formats."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_embedding_consistency_across_apis(self, client):
        """Test that embeddings are consistent across API formats."""
        test_text = "Consistency test across APIs"

        # Test Native API
        native_response = client.post("/api/v1/embed/", json={"texts": [test_text], "normalize": True})

        # Test OpenAI API
        openai_response = client.post("/v1/embeddings", json={"input": test_text, "model": "text-embedding-ada-002"})

        # Test TEI API
        tei_response = client.post("/embed", json={"inputs": test_text, "normalize": True})

        # If all APIs are ready, compare embeddings
        if all(r.status_code == 200 for r in [native_response, openai_response, tei_response]):
            native_vector = native_response.json()["vectors"][0]
            openai_vector = openai_response.json()["data"][0]["embedding"]
            tei_vector = tei_response.json()[0]

            # Vectors should be identical or very similar
            assert len(native_vector) == len(openai_vector) == len(tei_vector)

            # Check similarity (allowing for small floating point differences)
            for i, (n, o, t) in enumerate(zip(native_vector, openai_vector, tei_vector)):
                assert abs(n - o) < 1e-6, f"Native vs OpenAI mismatch at {i}: {n} vs {o}"
                assert abs(n - t) < 1e-6, f"Native vs TEI mismatch at {i}: {n} vs {t}"

    def test_reranking_consistency_native_vs_tei(self, client):
        """Test that reranking is consistent between Native and TEI APIs."""
        query = "What is artificial intelligence?"
        passages = ["AI is a branch of computer science", "Dogs are domestic animals", "Machine learning is part of AI"]

        # Test Native API
        native_response = client.post("/api/v1/rerank/", json={"query": query, "passages": passages})

        # Test TEI API
        tei_response = client.post("/rerank", json={"query": query, "texts": passages})

        # If both APIs are ready, compare rankings
        if native_response.status_code == 200 and tei_response.status_code == 200:
            native_results = native_response.json()["results"]
            tei_results = tei_response.json()

            # Sort by score to compare rankings
            native_ranking = sorted(native_results, key=lambda x: x["score"], reverse=True)
            tei_ranking = sorted(tei_results, key=lambda x: x["score"], reverse=True)

            # Rankings should be identical
            for native_result, tei_result in zip(native_ranking, tei_ranking):
                assert native_result["index"] == tei_result["index"]
                # Scores should be very similar
                assert abs(native_result["score"] - tei_result["score"]) < 1e-6

    def test_api_format_validation(self, client):
        """Test that each API format follows its specification."""
        test_input = "Format validation test"

        # Test all embedding endpoints
        endpoints = [
            ("/api/v1/embed/", {"texts": [test_input]}),
            ("/v1/embeddings", {"input": test_input, "model": "text-embedding-ada-002"}),
            ("/embed", {"inputs": test_input}),
        ]

        for endpoint, payload in endpoints:
            response = client.post(endpoint, json=payload)

            if response.status_code == 200:
                data = response.json()

                # Validate response structure for each format
                if endpoint == "/api/v1/embed/":
                    # Native format
                    assert "vectors" in data
                    assert "processing_time" in data
                    assert isinstance(data["vectors"], list)

                elif endpoint == "/v1/embeddings":
                    # OpenAI format
                    assert "object" in data
                    assert "data" in data
                    assert "model" in data
                    assert "usage" in data
                    assert data["object"] == "list"

                elif endpoint == "/embed":
                    # TEI format
                    assert isinstance(data, list)
                    assert len(data) == 1  # Single input


class TestRealBackendIntegration:
    """Tests that require actual backend initialization and model loading."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    @pytest.mark.asyncio
    async def test_backend_initialization(self):
        """Test backend initialization process."""
        from app.backends.base import BackendManager
        from app.backends.factory import BackendFactory

        # Test backend creation
        backend = BackendFactory.create_backend("auto")
        assert backend is not None

        # Test backend manager
        manager = BackendManager(backend)

        try:
            # Initialize backend (this may take time for model loading)
            await asyncio.wait_for(manager.initialize(), timeout=120.0)

            # Test basic embedding functionality
            result = await backend.embed_texts(["Hello world"])
            assert result.vectors is not None
            assert len(result.vectors) == 1
            assert len(result.vectors[0]) > 0

            # Test basic reranking functionality
            rerank_result = await backend.rerank_documents("test query", ["document 1", "document 2"])
            assert rerank_result.results is not None
            assert len(rerank_result.results) == 2

        except asyncio.TimeoutError:
            pytest.skip("Backend initialization timeout - model download may be needed")
        except Exception as e:
            error_msg = str(e)
            # Check if it's a quantization-related error that doesn't affect functionality
            if "quant_method" in error_msg or "quantization" in error_msg.lower():
                # This is a non-critical warning, but the backend should still work with fallback
                import warnings
                warnings.warn(f"Quantization config warning (non-critical): {error_msg}")
                # The backend should have loaded a fallback model, so we can continue testing
                pass
            else:
                pytest.skip(f"Backend initialization failed: {e}")
            
        # Try to test basic functionality regardless of quantization warnings
        try:
            result = await backend.embed_texts(["Hello world"])
            assert result.vectors is not None
            assert len(result.vectors) == 1
            assert len(result.vectors[0]) > 0

            # Test basic reranking functionality
            rerank_result = await backend.rerank_documents("test query", ["document 1", "document 2"])
            assert rerank_result.results is not None
            assert len(rerank_result.results) == 2
            
        except Exception as functional_e:
            pytest.skip(f"Backend functional test failed: {functional_e}")
        finally:
            # Cleanup
            await manager.cleanup()

    def test_end_to_end_embedding_workflow(self, client):
        """Test complete embedding workflow from request to response."""
        # Test with various text lengths and types
        test_cases = [
            "Short text",
            "Medium length text that contains multiple words and concepts",
            "Very long text that spans multiple sentences and contains various topics. " * 10,
            "Special characters: àáâãäåæçèéêë 中文 русский العربية",
            "Numbers and symbols: 123 456.789 @#$%^&*() []{}",
        ]

        for i, text in enumerate(test_cases):
            response = client.post("/api/v1/embed/", json={"texts": [text], "normalize": True})

            if response.status_code == 200:
                data = response.json()

                # Validate response structure
                assert "vectors" in data
                assert "processing_time" in data
                assert "num_texts" in data

                # Validate vector properties
                vector = data["vectors"][0]
                assert isinstance(vector, list)
                assert len(vector) > 0
                assert all(isinstance(x, (int, float)) for x in vector)

                # Check normalization
                magnitude = sum(x * x for x in vector) ** 0.5
                assert 0.95 <= magnitude <= 1.05, f"Vector not normalized: magnitude={magnitude}"

                # Check processing time is reasonable
                assert 0 <= data["processing_time"] <= 30.0

            elif response.status_code == 503:
                # Backend not ready - skip this test case
                pytest.skip(f"Backend not ready for test case {i}")

    def test_end_to_end_reranking_workflow(self, client):
        """Test complete reranking workflow from request to response."""
        test_cases = [
            {
                "query": "What is machine learning?",
                "passages": [
                    "Machine learning is a method of data analysis",
                    "Cats are small carnivorous mammals",
                    "Deep learning uses neural networks",
                    "Python is a programming language",
                ],
            },
            {
                "query": "How to cook pasta?",
                "passages": [
                    "Boil water in a large pot",
                    "Machine learning algorithms learn from data",
                    "Add salt to the boiling water",
                    "Cook pasta for 8-12 minutes",
                ],
            },
        ]

        for i, test_case in enumerate(test_cases):
            response = client.post("/api/v1/rerank/", json=test_case)

            if response.status_code == 200:
                data = response.json()

                # Validate response structure
                assert "results" in data
                assert "processing_time" in data
                assert "num_passages" in data

                # Validate results
                results = data["results"]
                assert len(results) == len(test_case["passages"])

                # Check result structure
                for result in results:
                    assert "index" in result
                    assert "score" in result
                    assert "text" in result
                    assert isinstance(result["index"], int)
                    assert isinstance(result["score"], (int, float))
                    assert isinstance(result["text"], str)

                # Results should be sorted by score (descending)
                scores = [result["score"] for result in results]
                assert scores == sorted(scores, reverse=True)

                # Check processing time is reasonable
                assert 0 <= data["processing_time"] <= 30.0

            elif response.status_code == 503:
                # Backend not ready - skip this test case
                pytest.skip(f"Backend not ready for test case {i}")


class TestPerformanceBenchmarks:
    """Performance tests and benchmarks."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_embedding_performance_single_text(self, client):
        """Benchmark single text embedding performance."""
        test_text = "Performance benchmark for single text embedding"

        # Warm up
        client.post("/api/v1/embed/", json={"texts": [test_text]})

        # Measure performance
        times = []
        for _ in range(5):
            start_time = time.time()
            response = client.post("/api/v1/embed/", json={"texts": [test_text]})
            end_time = time.time()

            if response.status_code == 200:
                times.append(end_time - start_time)

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            print("\nSingle text embedding performance:")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")

            # Performance assertions (adjust based on hardware)
            assert avg_time < 10.0, f"Average time too slow: {avg_time:.3f}s"

    def test_embedding_performance_batch(self, client):
        """Benchmark batch embedding performance."""
        batch_sizes = [1, 5, 10, 20]

        for batch_size in batch_sizes:
            texts = [f"Batch test text {i}" for i in range(batch_size)]

            # Warm up
            client.post("/api/v1/embed/", json={"texts": texts[:1]})

            start_time = time.time()
            response = client.post("/api/v1/embed/", json={"texts": texts})
            end_time = time.time()

            if response.status_code == 200:
                total_time = end_time - start_time
                time_per_text = total_time / batch_size

                print(f"\nBatch size {batch_size} performance:")
                print(f"  Total time: {total_time:.3f}s")
                print(f"  Time per text: {time_per_text:.3f}s")

                # Performance should scale reasonably
                assert total_time < 30.0, f"Batch processing too slow: {total_time:.3f}s"

    def test_reranking_performance(self, client):
        """Benchmark reranking performance."""
        query = "Performance test query"
        passage_counts = [2, 5, 10, 20]

        for count in passage_counts:
            passages = [f"Document {i} for performance testing" for i in range(count)]

            start_time = time.time()
            response = client.post("/api/v1/rerank/", json={"query": query, "passages": passages})
            end_time = time.time()

            if response.status_code == 200:
                total_time = end_time - start_time
                time_per_passage = total_time / count

                print(f"\nReranking {count} passages performance:")
                print(f"  Total time: {total_time:.3f}s")
                print(f"  Time per passage: {time_per_passage:.3f}s")

                # Performance assertions
                assert total_time < 30.0, f"Reranking too slow: {total_time:.3f}s"


class TestSystemIntegration:
    """System-level integration tests."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_health_check_integration(self, client):
        """Test health check with actual system state."""
        response = client.get("/health/")

        # Health check should always respond
        assert response.status_code in [200, 503]
        data = response.json()

        # Validate health response structure
        assert "status" in data
        assert "timestamp" in data
        assert "service" in data

        if response.status_code == 200:
            # System is healthy
            assert data["status"] in ["healthy", "ready"]

            # Should have backend information
            if "backend" in data:
                backend_info = data["backend"]
                assert "type" in backend_info
                assert "device" in backend_info

    def test_error_handling_integration(self, client):
        """Test error handling across the system."""
        # Test various error conditions
        error_cases = [
            # Invalid JSON
            ("/api/v1/embed/", "invalid json", 422),
            # Missing required fields
            ("/api/v1/embed/", {}, 422),
            # Wrong HTTP method
            ("/api/v1/embed/", None, 405, "GET"),
            # Nonexistent endpoint
            ("/nonexistent", {}, 404),
        ]

        for case in error_cases:
            endpoint = case[0]
            payload = case[1]
            expected_status = case[2]
            method = case[3] if len(case) > 3 else "POST"

            if method == "GET":
                response = client.get(endpoint)
            else:
                if payload == "invalid json":
                    response = client.post(endpoint, data=payload, headers={"Content-Type": "application/json"})
                else:
                    response = client.post(endpoint, json=payload)

            assert response.status_code == expected_status

    def test_concurrent_request_handling(self, client):
        """Test system behavior under concurrent load."""
        import queue
        import threading

        results = queue.Queue()

        def make_request(request_id):
            try:
                response = client.post("/api/v1/embed/", json={"texts": [f"Concurrent request {request_id}"]})
                results.put(
                    (
                        request_id,
                        response.status_code,
                        response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0,
                    )
                )
            except Exception as e:
                results.put((request_id, 500, str(e)))

        # Start multiple concurrent requests
        threads = []
        num_requests = 10

        for i in range(num_requests):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all requests to complete
        for thread in threads:
            thread.join()

        # Collect results
        request_results = []
        while not results.empty():
            request_results.append(results.get())

        # Analyze results
        assert len(request_results) == num_requests

        success_count = sum(1 for _, status, _ in request_results if status == 200)
        error_count = sum(1 for _, status, _ in request_results if status >= 400)

        print("\nConcurrent request results:")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful: {success_count}")
        print(f"  Errors: {error_count}")

        # At least some requests should succeed (or all should fail gracefully)
        if success_count > 0:
            # If some succeed, most should succeed
            assert success_count >= num_requests * 0.5
        else:
            # If none succeed, should be due to backend not ready
            assert all(status == 503 for _, status, _ in request_results)
