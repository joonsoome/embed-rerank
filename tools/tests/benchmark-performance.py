#!/usr/bin/env python3
"""
⚡ MLX Performance Benchmark Suite

Comprehensive performance testing for your MLX-powered embedding and reranking API.
Tests throughput, latency, and stability under different workloads.

Usage:
    python3 tools/tests/benchmark-performance.py
    python3 tools/tests/benchmark-performance.py --url http://localhost:11436 --output perf.json
"""

import requests
import json
import time
import statistics
import argparse
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import sys

# Color codes for pretty output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class PerformanceBenchmark:
    """⚡ MLX Performance Benchmark Suite"""
    
    def __init__(self, base_url: str = "http://localhost:11436"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
    def print_section(self, title: str):
        """Print a colorful section header"""
        print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}{title}{Colors.END}")
        print(f"{Colors.CYAN}{'='*60}{Colors.END}")
    
    def print_metric(self, name: str, value: str, unit: str = ""):
        """Print performance metric with colors"""
        print(f"{Colors.GREEN}✅{Colors.END} {name:<25} {Colors.BOLD}{value}{Colors.END} {unit}")
    
    def check_server_health(self) -> Dict[str, Any]:
        """🏥 Check server health before benchmarking"""
        self.print_section("🏥 Server Health Check")
        
        try:
            resp = self.session.get(f"{self.base_url}/health/", timeout=10)
            if resp.status_code != 200:
                raise Exception(f"Server unhealthy: HTTP {resp.status_code}")
            
            health_data = resp.json()
            backend_name = health_data.get('backend', {}).get('name', 'Unknown')
            model_name = health_data.get('backend', {}).get('model_name', 'Unknown')
            device = health_data.get('backend', {}).get('device', 'Unknown')
            
            self.print_metric("Backend", backend_name)
            self.print_metric("Model", model_name)
            self.print_metric("Device", device)
            
            return {
                "status": "healthy",
                "backend": backend_name,
                "model": model_name,
                "device": device,
                "full_info": health_data
            }
            
        except Exception as e:
            print(f"{Colors.RED}❌ Server health check failed: {str(e)}{Colors.END}")
            raise
    
    def warm_up_server(self, iterations: int = 5) -> float:
        """🔥 Warm up the server to ensure consistent performance"""
        print(f"{Colors.YELLOW}🔥 Warming up server ({iterations} iterations)...{Colors.END}")
        
        warm_up_text = "Warmup text for server initialization and model loading optimization."
        
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            payload = {"inputs": [warm_up_text]}
            resp = self.session.post(f"{self.base_url}/embed", json=payload, timeout=30)
            if resp.status_code == 200:
                duration = time.perf_counter() - start
                times.append(duration)
                print(f"  Warmup {i+1}/{iterations}: {duration*1000:.1f}ms")
            else:
                print(f"  Warmup {i+1}/{iterations}: ❌ HTTP {resp.status_code}")
        
        if times:
            avg_warmup = statistics.mean(times)
            print(f"{Colors.GREEN}✅ Warmup complete. Average: {avg_warmup*1000:.1f}ms{Colors.END}")
            return avg_warmup
        else:
            raise Exception("Warmup failed - no successful requests")
    
    def benchmark_embedding_latency(self, iterations: int = 100) -> Dict[str, Any]:
        """📊 Benchmark single-text embedding latency"""
        self.print_section("📊 Embedding Latency Benchmark")
        
        test_text = "Performance benchmark test sentence for precise latency measurement and optimization analysis."
        payload = {"inputs": [test_text]}
        
        latencies = []
        successful_requests = 0
        
        print(f"Running {iterations} iterations...")
        
        for i in range(iterations):
            try:
                start = time.perf_counter()
                resp = self.session.post(f"{self.base_url}/embed", json=payload, timeout=30)
                end = time.perf_counter()
                
                if resp.status_code == 200:
                    latency_ms = (end - start) * 1000
                    latencies.append(latency_ms)
                    successful_requests += 1
                    
                    if (i + 1) % 20 == 0:
                        print(f"  Progress: {i+1}/{iterations} ({latency_ms:.2f}ms)")
                else:
                    print(f"  Request {i+1}: ❌ HTTP {resp.status_code}")
                    
            except Exception as e:
                print(f"  Request {i+1}: ❌ {str(e)}")
        
        if not latencies:
            raise Exception("No successful latency measurements")
        
        # Calculate statistics
        mean_lat = statistics.mean(latencies)
        median_lat = statistics.median(latencies)
        p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        p99_lat = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        # Print results
        self.print_metric("Success Rate", f"{successful_requests}/{iterations} ({successful_requests/iterations*100:.1f}%)")
        self.print_metric("Mean Latency", f"{mean_lat:.2f}", "ms")
        self.print_metric("Median Latency", f"{median_lat:.2f}", "ms")
        self.print_metric("P95 Latency", f"{p95_lat:.2f}", "ms")
        self.print_metric("P99 Latency", f"{p99_lat:.2f}", "ms")
        self.print_metric("Min/Max Latency", f"{min_lat:.2f}/{max_lat:.2f}", "ms")
        self.print_metric("Std Deviation", f"{std_lat:.2f}", "ms")
        
        return {
            "successful_requests": successful_requests,
            "total_requests": iterations,
            "success_rate": successful_requests / iterations,
            "mean_latency_ms": round(mean_lat, 2),
            "median_latency_ms": round(median_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "p99_latency_ms": round(p99_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "std_deviation_ms": round(std_lat, 2),
            "all_latencies": [round(lat, 2) for lat in latencies]
        }
    
    def benchmark_throughput(self, max_concurrent: int = 20) -> Dict[str, Any]:
        """🚀 Benchmark throughput with different batch sizes and concurrency"""
        self.print_section("🚀 Throughput Benchmark")
        
        batch_sizes = [1, 5, 10, 20, 50]
        concurrency_levels = [1, 2, 4, 8] if max_concurrent >= 8 else [1, 2]
        
        results = []
        
        for batch_size in batch_sizes:
            for concurrency in concurrency_levels:
                if concurrency > max_concurrent:
                    continue
                    
                print(f"Testing batch_size={batch_size}, concurrency={concurrency}")
                
                # Generate test texts
                texts = [f"Throughput test batch {batch_size} concurrent {concurrency} text number {i}" 
                        for i in range(batch_size)]
                payload = {"inputs": texts}
                
                # Run benchmark
                def make_request():
                    start = time.perf_counter()
                    resp = self.session.post(f"{self.base_url}/embed", json=payload, timeout=60)
                    end = time.perf_counter()
                    return resp.status_code == 200, end - start, batch_size
                
                successful_requests = 0
                total_texts_processed = 0
                request_times = []
                
                # Measure for 10 seconds or 50 requests, whichever comes first
                test_duration = 10
                max_requests = 50
                
                start_time = time.perf_counter()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    requests_made = 0
                    
                    while (time.perf_counter() - start_time < test_duration and 
                           requests_made < max_requests):
                        
                        # Submit batch of concurrent requests
                        futures = []
                        for _ in range(min(concurrency, max_requests - requests_made)):
                            if requests_made >= max_requests:
                                break
                            futures.append(executor.submit(make_request))
                            requests_made += 1
                        
                        # Collect results
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                success, duration, texts_in_batch = future.result()
                                if success:
                                    successful_requests += 1
                                    total_texts_processed += texts_in_batch
                                    request_times.append(duration)
                            except Exception as e:
                                print(f"    Request failed: {e}")
                
                total_time = time.perf_counter() - start_time
                
                if request_times:
                    avg_request_time = statistics.mean(request_times)
                    texts_per_second = total_texts_processed / total_time
                    requests_per_second = successful_requests / total_time
                    
                    result = {
                        "batch_size": batch_size,
                        "concurrency": concurrency,
                        "successful_requests": successful_requests,
                        "total_texts_processed": total_texts_processed,
                        "test_duration_seconds": round(total_time, 2),
                        "texts_per_second": round(texts_per_second, 1),
                        "requests_per_second": round(requests_per_second, 1),
                        "avg_request_time_ms": round(avg_request_time * 1000, 2)
                    }
                    
                    results.append(result)
                    
                    print(f"  ✅ {texts_per_second:.1f} texts/sec, {requests_per_second:.1f} req/sec")
                else:
                    print(f"  ❌ No successful requests")
        
        # Find optimal configurations
        best_throughput = max(results, key=lambda x: x["texts_per_second"]) if results else None
        
        if best_throughput:
            self.print_metric("Peak Throughput", f"{best_throughput['texts_per_second']:.1f}", "texts/sec")
            self.print_metric("Optimal Batch Size", str(best_throughput['batch_size']))
            self.print_metric("Optimal Concurrency", str(best_throughput['concurrency']))
        
        return {
            "all_results": results,
            "peak_throughput": best_throughput
        }
    
    def benchmark_reranking_performance(self, iterations: int = 50) -> Dict[str, Any]:
        """🔄 Benchmark reranking performance with various passage counts"""
        self.print_section("🔄 Reranking Performance Benchmark")
        
        passage_counts = [5, 10, 20, 50]
        results = []
        
        for passage_count in passage_counts:
            print(f"Testing reranking with {passage_count} passages...")
            
            # Generate test data
            query = f"Performance benchmark query for reranking {passage_count} passages efficiently."
            passages = [f"Passage number {i} for reranking performance benchmark testing with multiple documents."
                       for i in range(passage_count)]
            payload = {"query": query, "passages": passages}
            
            latencies = []
            successful_requests = 0
            
            for i in range(iterations):
                try:
                    start = time.perf_counter()
                    resp = self.session.post(f"{self.base_url}/api/v1/rerank/", json=payload, timeout=60)
                    end = time.perf_counter()
                    
                    if resp.status_code == 200:
                        latency_ms = (end - start) * 1000
                        latencies.append(latency_ms)
                        successful_requests += 1
                        
                        if (i + 1) % 10 == 0:
                            print(f"    Progress: {i+1}/{iterations} ({latency_ms:.2f}ms)")
                    else:
                        print(f"    Request {i+1}: ❌ HTTP {resp.status_code}")
                        
                except Exception as e:
                    print(f"    Request {i+1}: ❌ {str(e)}")
            
            if latencies:
                mean_lat = statistics.mean(latencies)
                median_lat = statistics.median(latencies)
                passages_per_ms = passage_count / mean_lat
                
                result = {
                    "passage_count": passage_count,
                    "successful_requests": successful_requests,
                    "total_requests": iterations,
                    "mean_latency_ms": round(mean_lat, 2),
                    "median_latency_ms": round(median_lat, 2),
                    "passages_per_ms": round(passages_per_ms, 4),
                    "passages_per_second": round(passages_per_ms * 1000, 1)
                }
                
                results.append(result)
                
                print(f"  ✅ {mean_lat:.2f}ms avg, {passages_per_ms*1000:.1f} passages/sec")
            else:
                print(f"  ❌ No successful requests for {passage_count} passages")
        
        return {"results": results}
    
    def stress_test(self, duration_seconds: int = 60, max_concurrency: int = 10) -> Dict[str, Any]:
        """💪 Stress test to check stability under sustained load"""
        self.print_section(f"💪 Stress Test ({duration_seconds}s)")
        
        print(f"Running sustained load test for {duration_seconds} seconds with {max_concurrency} concurrent requests...")
        
        test_text = "Stress test sentence for sustained load testing and stability verification."
        payload = {"inputs": [test_text]}
        
        results = {
            "successful_requests": 0,
            "failed_requests": 0,
            "latencies": [],
            "error_types": {},
            "throughput_over_time": []
        }
        
        def make_request():
            try:
                start = time.perf_counter()
                resp = self.session.post(f"{self.base_url}/embed", json=payload, timeout=30)
                end = time.perf_counter()
                
                latency = (end - start) * 1000
                
                if resp.status_code == 200:
                    return True, latency, None
                else:
                    return False, latency, f"HTTP_{resp.status_code}"
            except Exception as e:
                return False, None, str(type(e).__name__)
        
        start_time = time.perf_counter()
        last_throughput_check = start_time
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            # Keep submitting requests for the duration
            while time.perf_counter() - start_time < duration_seconds:
                # Submit batch of requests
                futures = []
                for _ in range(max_concurrency):
                    futures.append(executor.submit(make_request))
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    success, latency, error = future.result()
                    
                    if success:
                        results["successful_requests"] += 1
                        results["latencies"].append(latency)
                    else:
                        results["failed_requests"] += 1
                        if error:
                            results["error_types"][error] = results["error_types"].get(error, 0) + 1
                
                # Calculate throughput every 10 seconds
                current_time = time.perf_counter()
                if current_time - last_throughput_check >= 10:
                    elapsed = current_time - start_time
                    throughput = results["successful_requests"] / elapsed
                    results["throughput_over_time"].append({
                        "time_elapsed": round(elapsed, 1),
                        "requests_per_second": round(throughput, 1)
                    })
                    last_throughput_check = current_time
                    print(f"  {elapsed:.0f}s: {throughput:.1f} req/sec")
        
        total_time = time.perf_counter() - start_time
        total_requests = results["successful_requests"] + results["failed_requests"]
        
        # Calculate final statistics
        if results["latencies"]:
            mean_latency = statistics.mean(results["latencies"])
            p95_latency = (statistics.quantiles(results["latencies"], n=20)[18] 
                          if len(results["latencies"]) >= 20 else max(results["latencies"]))
        else:
            mean_latency = 0
            p95_latency = 0
        
        success_rate = results["successful_requests"] / total_requests if total_requests > 0 else 0
        overall_throughput = results["successful_requests"] / total_time
        
        # Print final results
        self.print_metric("Total Requests", str(total_requests))
        self.print_metric("Success Rate", f"{success_rate*100:.1f}%")
        self.print_metric("Overall Throughput", f"{overall_throughput:.1f}", "req/sec")
        self.print_metric("Mean Latency", f"{mean_latency:.2f}", "ms")
        self.print_metric("P95 Latency", f"{p95_latency:.2f}", "ms")
        
        if results["error_types"]:
            print(f"{Colors.YELLOW}⚠️ Error Types:{Colors.END}")
            for error_type, count in results["error_types"].items():
                print(f"  {error_type}: {count}")
        
        results.update({
            "total_requests": total_requests,
            "success_rate": success_rate,
            "overall_throughput": overall_throughput,
            "mean_latency_ms": round(mean_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "test_duration_seconds": round(total_time, 2)
        })
        
        return results
    
    def run_full_benchmark(self, stress_duration: int = 60) -> Dict[str, Any]:
        """🎯 Run complete performance benchmark suite"""
        start_time = time.time()
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}")
        print("⚡ MLX Performance Benchmark Suite")
        print("Testing your actual running server performance")
        print(f"{Colors.END}")
        
        results = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "server_url": self.base_url,
                "benchmark_version": "1.0.0"
            }
        }
        
        try:
            # Check server health
            results["server_health"] = self.check_server_health()
            
            # Warm up
            warmup_time = self.warm_up_server()
            results["warmup_time"] = warmup_time
            
            # Run benchmarks
            results["embedding_latency"] = self.benchmark_embedding_latency()
            results["throughput"] = self.benchmark_throughput()
            results["reranking_performance"] = self.benchmark_reranking_performance()
            results["stress_test"] = self.stress_test(duration_seconds=stress_duration)
            
            total_time = time.time() - start_time
            
            # Generate summary
            embedding_results = results["embedding_latency"]
            throughput_results = results["throughput"]
            stress_results = results["stress_test"]
            
            peak_throughput = (throughput_results["peak_throughput"]["texts_per_second"] 
                             if throughput_results["peak_throughput"] else 0)
            
            results["benchmark_summary"] = {
                "total_benchmark_time": f"{total_time:.2f}s",
                "mean_latency_ms": embedding_results["mean_latency_ms"],
                "p95_latency_ms": embedding_results["p95_latency_ms"],
                "peak_throughput_texts_per_sec": peak_throughput,
                "stress_test_success_rate": f"{stress_results['success_rate']*100:.1f}%",
                "overall_status": "🎉 PERFORMANCE EXCELLENT" if (
                    embedding_results["mean_latency_ms"] < 100 and
                    stress_results["success_rate"] > 0.95 and
                    peak_throughput > 100
                ) else "⚠️ PERFORMANCE NEEDS REVIEW"
            }
            
            # Print summary
            self.print_section("🎯 Performance Summary")
            summary = results["benchmark_summary"]
            
            print(f"{Colors.BOLD}{summary['overall_status']}{Colors.END}")
            print(f"Mean Latency: {summary['mean_latency_ms']}ms")
            print(f"P95 Latency: {summary['p95_latency_ms']}ms")
            print(f"Peak Throughput: {summary['peak_throughput_texts_per_sec']} texts/sec")
            print(f"Stress Test Success: {summary['stress_test_success_rate']}")
            print(f"Total Benchmark Time: {summary['total_benchmark_time']}")
            
            return results
            
        except Exception as e:
            results["benchmark_summary"] = {
                "overall_status": f"❌ BENCHMARK FAILED: {str(e)}",
                "error": str(e)
            }
            print(f"\n{Colors.RED}❌ Benchmark failed: {str(e)}{Colors.END}")
            return results

def main():
    parser = argparse.ArgumentParser(
        description="⚡ MLX Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/tests/benchmark-performance.py                                      # Full benchmark
  python3 tools/tests/benchmark-performance.py --url http://localhost:8080          # Custom URL
  python3 tools/tests/benchmark-performance.py --stress-duration 30                 # Shorter stress test
  python3 tools/tests/benchmark-performance.py --output benchmark-results.json      # Save results
        """
    )
    
    parser.add_argument("--url", default="http://localhost:11436",
                       help="API server URL (default: http://localhost:11436)")
    parser.add_argument("--output", type=Path,
                       help="Save benchmark results to JSON file")
    parser.add_argument("--stress-duration", type=int, default=60,
                       help="Stress test duration in seconds (default: 60)")
    parser.add_argument("--latency-iterations", type=int, default=100,
                       help="Number of latency test iterations (default: 100)")
    
    args = parser.parse_args()
    
    try:
        benchmark = PerformanceBenchmark(args.url)
        results = benchmark.run_full_benchmark(stress_duration=args.stress_duration)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to {args.output}")
        
        # Exit code based on results
        summary = results.get("benchmark_summary", {})
        if "EXCELLENT" in summary.get("overall_status", ""):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ Benchmark interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Benchmark error: {str(e)}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()
