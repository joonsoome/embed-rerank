"""
Backend performance benchmarking tool.
"""

import asyncio
import statistics
import time
from typing import Any, Dict, List

from app.backends.factory import BackendFactory
from app.utils.logger import setup_logging

logger = setup_logging()


class BackendBenchmark:
    """Tool for benchmarking backend performance.

    Can be constructed with either a backend instance or a model_name. Tests
    expect to pass a backend instance and access benchmark.backend.
    """

    def __init__(self, backend_or_model: object = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize benchmark with a backend instance or model name."""
        self.results = {}

        # If given a backend instance (has embed_texts), use it directly
        if hasattr(backend_or_model, 'embed_texts'):
            self.backend = backend_or_model
            self.model_name = getattr(self.backend, 'model_name', None)
        else:
            self.backend = None
            self.model_name = backend_or_model

    async def benchmark_backend(
        self, backend_type: str, test_texts: List[str], iterations: int = 3, batch_sizes: List[int] = [1, 8, 16, 32]
    ) -> Dict[str, Any]:
        """
        Benchmark a specific backend.

        Args:
            backend_type: Backend type to benchmark
            test_texts: List of test texts
            iterations: Number of iterations per test
            batch_sizes: Batch sizes to test

        Returns:
            Dict with benchmark results
        """
        logger.info(
            "Starting backend benchmark",
            backend_type=backend_type,
            model_name=self.model_name,
            num_texts=len(test_texts),
            iterations=iterations,
        )

        try:
            # Create and load backend
            backend = BackendFactory.create_backend(backend_type, self.model_name)
            load_start = time.time()
            await backend.load_model()
            load_time = time.time() - load_start

            # Get backend info
            model_info = backend.get_model_info()
            device_info = backend.get_device_info()

            # Benchmark different batch sizes
            batch_results = {}

            for batch_size in batch_sizes:
                logger.info(f"Benchmarking batch size {batch_size}")

                times = []
                embedding_dims = []

                for i in range(iterations):
                    start_time = time.time()
                    result = await backend.embed_texts(test_texts, batch_size=batch_size)
                    end_time = time.time()

                    times.append(end_time - start_time)
                    embedding_dims.append(result.vectors.shape[1])

                batch_results[batch_size] = {
                    "mean_time": statistics.mean(times),
                    "median_time": statistics.median(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                    "embedding_dim": embedding_dims[0],
                    "throughput_texts_per_sec": len(test_texts) / statistics.mean(times),
                }

            # Test similarity computation
            similarity_times = []
            for i in range(iterations):
                # Get some embeddings first
                result = await backend.embed_texts(test_texts[:5], batch_size=5)
                query_emb = result.vectors[0]
                passage_embs = result.vectors[1:]

                start_time = time.time()
                await backend.compute_similarity(query_emb, passage_embs)
                end_time = time.time()

                similarity_times.append(end_time - start_time)

            benchmark_result = {
                "backend_type": backend_type,
                "model_name": self.model_name,
                "load_time": load_time,
                "model_info": model_info,
                "device_info": device_info,
                "batch_results": batch_results,
                "similarity_computation": {
                    "mean_time": statistics.mean(similarity_times),
                    "median_time": statistics.median(similarity_times),
                    "min_time": min(similarity_times),
                    "max_time": max(similarity_times),
                },
                "optimal_batch_size": self._find_optimal_batch_size(batch_results),
                "benchmark_timestamp": time.time(),
            }

            logger.info(
                "Backend benchmark completed",
                backend_type=backend_type,
                load_time=load_time,
                optimal_batch_size=benchmark_result["optimal_batch_size"],
            )

            return benchmark_result

        except Exception as e:
            logger.error("Backend benchmark failed", backend_type=backend_type, error=str(e))
            raise

    async def run_single_benchmark(self, texts: List[str], batch_size: int = 1) -> Dict[str, Any]:
        """Run a single benchmark using the provided backend instance or the default model.

        Returns a dict with processing_time and throughput.
        """
        if self.backend is None:
            # Create a temporary backend for the run
            backend = BackendFactory.create_backend('torch', self.model_name)
            await backend.load_model()
        else:
            backend = self.backend

        start = time.time()
        await backend.embed_texts(texts, batch_size=batch_size)
        processing_time = time.time() - start

        throughput = len(texts) / processing_time if processing_time > 0 else float('inf')

        return {"processing_time": processing_time, "throughput": throughput}

    def _find_optimal_batch_size(self, batch_results: Dict[int, Dict]) -> int:
        """Find the optimal batch size based on throughput."""
        best_batch_size = 1
        best_throughput = 0

        for batch_size, results in batch_results.items():
            throughput = results["throughput_texts_per_sec"]
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size

        return best_batch_size

    async def compare_backends(
        self, backend_types: List[str], test_texts: List[str], iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Compare multiple backends.

        Args:
            backend_types: List of backend types to compare
            test_texts: Test texts for benchmarking
            iterations: Number of iterations per test

        Returns:
            Comparison results
        """
        logger.info(
            "Starting backend comparison", backend_types=backend_types, num_texts=len(test_texts), iterations=iterations
        )

        comparison_results = {
            "backends": {},
            "comparison": {},
            "test_config": {
                "model_name": self.model_name,
                "num_texts": len(test_texts),
                "iterations": iterations,
                "test_timestamp": time.time(),
            },
        }

        # Benchmark each backend
        for backend_type in backend_types:
            try:
                result = await self.benchmark_backend(backend_type, test_texts, iterations)
                comparison_results["backends"][backend_type] = result
            except Exception as e:
                logger.error(f"Failed to benchmark {backend_type}", error=str(e))
                comparison_results["backends"][backend_type] = {"error": str(e), "available": False}

        # Generate comparison metrics
        available_backends = {k: v for k, v in comparison_results["backends"].items() if "error" not in v}

        if len(available_backends) > 1:
            comparison_results["comparison"] = self._generate_comparison(available_backends)

        return comparison_results

    def _generate_comparison(self, backends: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comparison metrics between backends."""
        comparison = {"load_time": {}, "inference_speed": {}, "optimal_batch_sizes": {}, "recommendations": []}

        # Compare load times
        load_times = {name: result["load_time"] for name, result in backends.items()}
        fastest_load = min(load_times, key=load_times.get)
        comparison["load_time"] = {"fastest": fastest_load, "times": load_times}

        # Compare inference speeds (using optimal batch size for each)
        inference_speeds = {}
        for name, result in backends.items():
            optimal_batch = result["optimal_batch_size"]
            speed = result["batch_results"][optimal_batch]["throughput_texts_per_sec"]
            inference_speeds[name] = speed

        fastest_inference = max(inference_speeds, key=inference_speeds.get)
        comparison["inference_speed"] = {"fastest": fastest_inference, "speeds": inference_speeds}

        # Optimal batch sizes
        comparison["optimal_batch_sizes"] = {name: result["optimal_batch_size"] for name, result in backends.items()}

        # Generate recommendations
        if fastest_load == fastest_inference:
            comparison["recommendations"].append(f"{fastest_load} is optimal for both load time and inference speed")
        else:
            comparison["recommendations"].append(f"{fastest_load} loads fastest, {fastest_inference} runs fastest")

        return comparison

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print benchmark results in a readable format."""
        print("\n" + "=" * 60)
        print("BACKEND BENCHMARK RESULTS")
        print("=" * 60)

        if "backends" in results:
            for backend_name, backend_result in results["backends"].items():
                if "error" in backend_result:
                    print(f"\n❌ {backend_name.upper()}: {backend_result['error']}")
                    continue

                print(f"\n✅ {backend_name.upper()}:")
                print(f"   Load time: {backend_result['load_time']:.2f}s")
                print(f"   Device: {backend_result['device_info'].get('device', 'unknown')}")
                print(f"   Optimal batch size: {backend_result['optimal_batch_size']}")

                optimal_batch = backend_result['optimal_batch_size']
                optimal_result = backend_result['batch_results'][optimal_batch]
                print(f"   Best throughput: {optimal_result['throughput_texts_per_sec']:.1f} texts/sec")
                print(f"   Best latency: {optimal_result['mean_time']:.3f}s")

        if "comparison" in results and results["comparison"]:
            print("\n🏆 COMPARISON:")
            comp = results["comparison"]
            print(f"   Fastest loading: {comp['load_time']['fastest']}")
            print(f"   Fastest inference: {comp['inference_speed']['fastest']}")

            for rec in comp["recommendations"]:
                print(f"   💡 {rec}")


# Test function
async def run_benchmark():
    """Run a quick benchmark test."""
    benchmark = BackendBenchmark()

    test_texts = [
        "Hello, how are you today?",
        "Machine learning is fascinating",
        "Apple Silicon provides excellent performance",
        "PyTorch and MLX are both great frameworks",
        "Embedding models enable semantic search",
    ] * 4  # 20 texts total

    # Test available backends
    available_backends = []
    try:
        available_backends.append("torch")
    except Exception:
        pass

    try:
        available_backends.append("mlx")
    except Exception:
        pass

    if available_backends:
        results = await benchmark.compare_backends(available_backends, test_texts)
        benchmark.print_results(results)
        return results
    else:
        print("No backends available for benchmarking")
        return None


if __name__ == "__main__":
    asyncio.run(run_benchmark())
