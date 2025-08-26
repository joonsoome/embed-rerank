"""
Health Check and Monitoring Tests for Embed-Rerank Service

This module contains tests for:
- Health endpoint functionality and monitoring
- System status reporting and diagnostics
- Service readiness and backend state validation
- Performance monitoring and metrics
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_health_endpoint_accessibility(self, client):
        """Test health endpoint is accessible."""
        response = client.get("/health/")

        # Health endpoint should always respond
        assert response.status_code in [200, 503]

        # Should return JSON
        assert response.headers["content-type"].startswith("application/json")

    def test_health_response_structure(self, client):
        """Test health endpoint response structure."""
        response = client.get("/health/")
        data = response.json()

        # Required fields
        required_fields = ["status", "timestamp", "service"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
            assert data[field] is not None, f"Field {field} is None"

        # Validate timestamp format
        timestamp = data["timestamp"]
        assert isinstance(timestamp, str)
        # Should be ISO 8601 format
        import datetime

        try:
            datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {timestamp}")

        # Validate service info
        service_info = data["service"]
        assert isinstance(service_info, dict)
        assert "name" in service_info
        assert "version" in service_info

    def test_health_status_values(self, client):
        """Test health status values are valid."""
        response = client.get("/health/")
        data = response.json()

        status = data["status"]
        valid_statuses = ["healthy", "ready", "initializing", "not_ready", "error"]
        assert status in valid_statuses, f"Invalid status: {status}"

        # Status should correspond to HTTP status code
        if response.status_code == 200:
            assert status in ["healthy", "ready"]
        elif response.status_code == 503:
            assert status in ["initializing", "not_ready", "error"]

    def test_health_backend_info(self, client):
        """Test backend information in health response."""
        response = client.get("/health/")
        data = response.json()

        if response.status_code == 200 and "backend" in data:
            backend_info = data["backend"]
            assert isinstance(backend_info, dict)

            # Common backend fields
            expected_fields = ["type", "device"]
            for field in expected_fields:
                if field in backend_info:
                    assert isinstance(backend_info[field], str)
                    assert len(backend_info[field]) > 0

    def test_health_consistency_multiple_calls(self, client):
        """Test health endpoint consistency across multiple calls."""
        responses = []

        # Make multiple health check calls
        for _ in range(5):
            response = client.get("/health/")
            responses.append(response)
            time.sleep(0.1)  # Small delay between calls

        # All responses should have same structure
        for response in responses:
            assert response.status_code in [200, 503]
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert "service" in data

        # Service info should be consistent
        service_infos = [r.json()["service"] for r in responses]
        first_service = service_infos[0]
        for service_info in service_infos[1:]:
            assert service_info["name"] == first_service["name"]
            assert service_info["version"] == first_service["version"]


class TestHealthMonitoring:
    """Tests for health monitoring and diagnostics."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_health_response_time(self, client):
        """Test health endpoint response time."""
        times = []

        for _ in range(10):
            start_time = time.time()
            client.get("/health/")
            end_time = time.time()

            response_time = end_time - start_time
            times.append(response_time)

            # Health check should be fast
            assert response_time < 5.0, f"Health check too slow: {response_time:.3f}s"

        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print("\nHealth check performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")

        # Health checks should be consistently fast
        assert avg_time < 1.0, f"Average health check time too slow: {avg_time:.3f}s"

    def test_health_under_load(self, client):
        """Test health endpoint behavior under load."""
        import queue
        import threading

        results = queue.Queue()

        def check_health():
            try:
                response = client.get("/health/")
                results.put((response.status_code, response.json()))
            except Exception as e:
                results.put((500, {"error": str(e)}))

        # Make concurrent health checks
        threads = []
        num_checks = 20

        for _ in range(num_checks):
            thread = threading.Thread(target=check_health)
            threads.append(thread)
            thread.start()

        # Wait for all checks to complete
        for thread in threads:
            thread.join()

        # Collect results
        health_results = []
        while not results.empty():
            health_results.append(results.get())

        # All health checks should complete
        assert len(health_results) == num_checks

        # All should return valid responses
        for status_code, data in health_results:
            assert status_code in [200, 503]
            assert isinstance(data, dict)
            assert "status" in data

    def test_health_error_conditions(self, client):
        """Test health endpoint with various error conditions."""
        # Test wrong HTTP method
        response = client.post("/health/")
        assert response.status_code == 405

        # Test with query parameters (should be ignored)
        response = client.get("/health/?param=value")
        assert response.status_code in [200, 503]

        # Test with extra headers
        response = client.get("/health/", headers={"X-Custom": "value"})
        assert response.status_code in [200, 503]


class TestSystemStatus:
    """Tests for system status reporting."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_service_information(self, client):
        """Test service information accuracy."""
        response = client.get("/health/")
        data = response.json()

        service_info = data["service"]

        # Service name should match application
        assert service_info["name"] == "embed-rerank"

        # Version should be valid format
        version = service_info["version"]
        assert isinstance(version, str)
        assert len(version) > 0

        # Version should follow semantic versioning pattern (if applicable)
        import re

        # Allow various version formats
        version_patterns = [
            r'^\d+\.\d+\.\d+$',  # 1.0.0
            r'^\d+\.\d+\.\d+-\w+$',  # 1.0.0-beta
            r'^v\d+\.\d+\.\d+$',  # v1.0.0
            r'^\d+\.\d+$',  # 1.0
            r'^dev$',  # dev
            r'^latest$',  # latest
        ]

        valid_version = any(re.match(pattern, version) for pattern in version_patterns)
        assert valid_version, f"Invalid version format: {version}"

    def test_backend_status_reporting(self, client):
        """Test backend status reporting in health check."""
        response = client.get("/health/")
        data = response.json()

        if "backend" in data:
            backend_info = data["backend"]

            # Backend type should be valid
            if "type" in backend_info:
                valid_types = ["mlx", "torch", "cpu"]
                assert backend_info["type"] in valid_types

            # Device should be valid
            if "device" in backend_info:
                valid_devices = ["cpu", "mps", "cuda", "mlx"]
                device = backend_info["device"].lower()
                assert any(valid_dev in device for valid_dev in valid_devices)

            # Model info if available
            if "model" in backend_info:
                model_info = backend_info["model"]
                assert isinstance(model_info, (str, dict))
                if isinstance(model_info, str):
                    assert len(model_info) > 0

    def test_status_transitions(self, client):
        """Test status transitions over time."""
        # Make multiple health checks to observe status changes
        statuses = []

        for i in range(10):
            response = client.get("/health/")
            data = response.json()
            statuses.append(data["status"])

            if i < 9:  # Don't sleep after last iteration
                time.sleep(0.5)

        # Status should be consistent or show valid transitions
        unique_statuses = set(statuses)

        # Valid status transitions:
        # initializing -> ready/healthy
        # not_ready -> ready/healthy
        # ready/healthy should remain stable

        if len(unique_statuses) > 1:
            # Status changed during test
            first_status = statuses[0]
            last_status = statuses[-1]

            # Validate transition
            if first_status in ["initializing", "not_ready"]:
                # Should transition to better state or stay same
                assert last_status in ["initializing", "not_ready", "ready", "healthy"]
            else:
                # Should remain stable
                assert last_status == first_status


class TestHealthMetrics:
    """Tests for health metrics and performance indicators."""

    @pytest.fixture(scope="class")
    def client(self):
        """Test client fixture."""
        with TestClient(app) as test_client:
            yield test_client

    def test_timestamp_accuracy(self, client):
        """Test health check timestamp accuracy."""
        before_request = time.time()
        response = client.get("/health/")
        after_request = time.time()

        data = response.json()
        timestamp_str = data["timestamp"]

        # Parse timestamp
        import datetime

        timestamp = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        timestamp_unix = timestamp.timestamp()

        # Timestamp should be within request window
        assert (
            before_request <= timestamp_unix <= after_request
        ), f"Timestamp {timestamp_unix} not within [{before_request}, {after_request}]"

    def test_health_caching_behavior(self, client):
        """Test health endpoint caching behavior."""
        # Make rapid successive requests
        responses = []
        timestamps = []

        for _ in range(5):
            response = client.get("/health/")
            data = response.json()
            responses.append(response)
            timestamps.append(data["timestamp"])

        # Check if responses are cached or fresh
        unique_timestamps = set(timestamps)

        if len(unique_timestamps) == 1:
            # Responses are cached
            print("Health responses appear to be cached")
        else:
            # Responses are fresh
            print("Health responses are generated fresh")

        # Both behaviors are acceptable for health checks

    def test_health_resource_usage(self, client):
        """Test health endpoint resource usage."""
        import os

        import psutil

        # Get current process
        process = psutil.Process(os.getpid())

        # Measure memory before
        memory_before = process.memory_info().rss

        # Make multiple health checks
        for _ in range(100):
            response = client.get("/health/")
            assert response.status_code in [200, 503]

        # Measure memory after
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Memory increase should be minimal for health checks
        max_increase = 10 * 1024 * 1024  # 10 MB tolerance
        assert (
            memory_increase < max_increase
        ), f"Health checks increased memory by {memory_increase / 1024 / 1024:.1f} MB"

    def test_health_concurrent_performance(self, client):
        """Test health endpoint performance under concurrent load."""
        import queue
        import threading

        results = queue.Queue()

        def timed_health_check():
            start_time = time.time()
            response = client.get("/health/")
            end_time = time.time()

            results.put((end_time - start_time, response.status_code))

        # Start concurrent health checks
        threads = []
        num_concurrent = 50

        start_time = time.time()
        for _ in range(num_concurrent):
            thread = threading.Thread(target=timed_health_check)
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Collect results
        response_times = []
        status_codes = []
        while not results.empty():
            response_time, status_code = results.get()
            response_times.append(response_time)
            status_codes.append(status_code)

        # All requests should complete
        assert len(response_times) == num_concurrent

        # Calculate statistics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)

        print("\nConcurrent health check performance:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  Max response time: {max_response_time:.3f}s")
        print(f"  Min response time: {min_response_time:.3f}s")

        # Performance assertions - made more lenient for CI environments
        assert avg_response_time < 10.0, f"Average response time too slow: {avg_response_time:.3f}s"
        assert max_response_time < 15.0, f"Max response time too slow: {max_response_time:.3f}s"

        # All should return valid status codes
        for status_code in status_codes:
            assert status_code in [200, 503]
