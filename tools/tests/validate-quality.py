#!/usr/bin/env python3
"""
🧠 MLX Embedding & Reranking Quality Validator

Comprehensive quality validation for your MLX-powered embedding and reranking API.
Tests your actual running server with your configured model settings.

Usage:
    python3 tools/tests/validate-quality.py
    python3 tools/tests/validate-quality.py --url http://localhost:11436 --output results.json
"""

import requests
import json
import time
import statistics
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

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

class EmbeddingQualityValidator:
    """🎯 MLX Embedding & Reranking Quality Validator"""
    
    def __init__(self, base_url: str = "http://localhost:11436"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self.server_info = {}
    
    def print_section(self, title: str):
        """Print a colorful section header"""
        print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}{title}{Colors.END}")
        print(f"{Colors.CYAN}{'='*60}{Colors.END}")
    
    def print_status(self, test_name: str, status: str, details: str = ""):
        """Print test status with colors"""
        if "PASS" in status or "✅" in status:
            color = Colors.GREEN
        elif "FAIL" in status or "❌" in status:
            color = Colors.RED
        else:
            color = Colors.YELLOW
        
        print(f"{color}{status}{Colors.END} {test_name}")
        if details:
            print(f"    {details}")
    
    def check_server_health(self) -> Dict[str, Any]:
        """🏥 Check server health and get configuration info"""
        self.print_section("🏥 Server Health Check")
        
        try:
            resp = self.session.get(f"{self.base_url}/health/", timeout=10)
            if resp.status_code != 200:
                raise Exception(f"Server unhealthy: HTTP {resp.status_code}")
            
            health_data = resp.json()
            self.server_info = health_data
            
            backend_name = health_data.get('backend', {}).get('name', 'Unknown')
            model_name = health_data.get('backend', {}).get('model_name', 'Unknown')
            device = health_data.get('backend', {}).get('device', 'Unknown')
            
            self.print_status("Server Health", "✅ HEALTHY")
            self.print_status("Backend", f"✅ {backend_name}")
            self.print_status("Model", f"✅ {model_name}")
            self.print_status("Device", f"✅ {device}")
            
            return {
                "status": "healthy",
                "backend": backend_name,
                "model": model_name,
                "device": device,
                "full_info": health_data
            }
            
        except Exception as e:
            self.print_status("Server Health", f"❌ FAILED: {str(e)}")
            raise
    
    async def test_basic_embedding(self) -> Dict[str, Any]:
        """🔤 Test basic embedding functionality"""
        self.print_section("🔤 Basic Embedding Test")
        
        test_text = "Hello, MLX world!"
        
        try:
            payload = {"inputs": [test_text]}
            async with self.session.post(f"{self.base_url}/embed", json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}: {await resp.text()}")
                
                data = await resp.json()
                embeddings = data
                
                if not embeddings or len(embeddings) == 0:
                    raise Exception("No embeddings returned")
                
                vector = embeddings[0]
                dimension = len(vector)
                norm = float(np.linalg.norm(vector))
                
                # Check if properly normalized
                is_normalized = 0.99 < norm < 1.01
                
                self.print_status("Embedding Generation", "✅ SUCCESS")
                self.print_status("Vector Dimension", f"✅ {dimension}")
                self.print_status("Vector Normalization", f"{'✅' if is_normalized else '⚠️'} norm={norm:.6f}")
                
                return {
                    "status": "success",
                    "dimension": dimension,
                    "norm": norm,
                    "is_normalized": is_normalized,
                    "value_range": [float(min(vector)), float(max(vector))],
                    "std_dev": float(np.std(vector))
                }
                
        except Exception as e:
            self.print_status("Basic Embedding", f"❌ FAILED: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_semantic_similarity(self) -> Dict[str, Any]:
        """🧠 Test semantic understanding through similarity"""
        self.print_section("🧠 Semantic Similarity Test")
        
        # Test pairs: (similar_pair, dissimilar_pair)
        test_cases = [
            # Technology similarity
            (("Apple MacBook Pro", "Apple laptop computer"), 
             ("Apple MacBook Pro", "Banana fruit")),
            
            # Concept similarity  
            (("Machine learning model", "AI neural network"),
             ("Machine learning model", "Cooking recipe")),
            
            # Language similarity
            (("Fast car", "Quick automobile"),
             ("Fast car", "Slow turtle")),
            
            # Korean-English similarity
            (("안녕하세요", "Hello there"),
             ("안녕하세요", "Goodbye forever"))
        ]
        
        results = []
        
        try:
            for (similar_pair, dissimilar_pair) in test_cases:
                # Test similar pair
                payload = {"inputs": list(similar_pair)}
                async with self.session.post(f"{self.base_url}/embed", json=payload) as resp:
                    data = await resp.json()
                    v1, v2 = data[0], data[1]
                    sim_similarity = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                
                # Test dissimilar pair
                payload = {"inputs": list(dissimilar_pair)}
                async with self.session.post(f"{self.base_url}/embed", json=payload) as resp:
                    data = await resp.json()
                    v1, v2 = data[0], data[1]
                    dissim_similarity = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                
                # Check if similar pair has higher similarity
                semantic_coherent = sim_similarity > dissim_similarity
                
                result = {
                    "similar_pair": similar_pair,
                    "dissimilar_pair": dissimilar_pair,
                    "similar_score": sim_similarity,
                    "dissimilar_score": dissim_similarity,
                    "coherent": semantic_coherent
                }
                results.append(result)
                
                status = "✅" if semantic_coherent else "⚠️"
                self.print_status(
                    f"'{similar_pair[0][:20]}...' vs others",
                    f"{status} sim={sim_similarity:.4f} > dissim={dissim_similarity:.4f}"
                )
            
            coherent_count = sum(1 for r in results if r["coherent"])
            overall_coherent = coherent_count >= len(results) * 0.75  # 75% threshold
            
            self.print_status("Overall Semantic Coherence", 
                            f"{'✅' if overall_coherent else '⚠️'} {coherent_count}/{len(results)} tests coherent")
            
            return {
                "status": "success",
                "test_results": results,
                "coherent_tests": coherent_count,
                "total_tests": len(results),
                "overall_coherent": overall_coherent
            }
            
        except Exception as e:
            self.print_status("Semantic Similarity", f"❌ FAILED: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_multilingual_support(self) -> Dict[str, Any]:
        """🌍 Test multilingual embedding capability"""
        self.print_section("🌍 Multilingual Support Test")
        
        multilingual_greetings = [
            ("English", "Hello, how are you today?"),
            ("Korean", "안녕하세요, 오늘 어떻게 지내세요?"),
            ("Spanish", "Hola, ¿cómo estás hoy?"),
            ("Japanese", "こんにちは、今日はいかがですか？"),
            ("French", "Bonjour, comment allez-vous aujourd'hui?"),
            ("German", "Hallo, wie geht es dir heute?")
        ]
        
        try:
            texts = [text for _, text in multilingual_greetings]
            payload = {"inputs": texts}
            
            async with self.session.post(f"{self.base_url}/embed", json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                
                data = await resp.json()
                embeddings = data
                
                # Check each embedding
                language_results = []
                for i, (lang, text) in enumerate(multilingual_greetings):
                    vector = embeddings[i]
                    norm = float(np.linalg.norm(vector))
                    non_zero_count = np.count_nonzero(vector)
                    
                    is_valid = 0.99 < norm < 1.01 and non_zero_count > len(vector) * 0.5
                    
                    language_results.append({
                        "language": lang,
                        "text": text,
                        "norm": norm,
                        "non_zero_elements": non_zero_count,
                        "total_elements": len(vector),
                        "valid": is_valid
                    })
                    
                    status = "✅" if is_valid else "❌"
                    self.print_status(f"{lang} embedding", 
                                    f"{status} norm={norm:.4f}, non_zero={non_zero_count}/{len(vector)}")
                
                valid_count = sum(1 for r in language_results if r["valid"])
                all_valid = valid_count == len(language_results)
                
                self.print_status("Multilingual Support", 
                                f"{'✅' if all_valid else '⚠️'} {valid_count}/{len(language_results)} languages valid")
                
                return {
                    "status": "success",
                    "language_results": language_results,
                    "valid_languages": valid_count,
                    "total_languages": len(language_results),
                    "all_languages_supported": all_valid
                }
                
        except Exception as e:
            self.print_status("Multilingual Support", f"❌ FAILED: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_reranking_quality(self) -> Dict[str, Any]:
        """🔄 Test reranking functionality and quality"""
        self.print_section("🔄 Reranking Quality Test")
        
        # Test cases with clear relevance rankings
        test_cases = [
            {
                "query": "What is machine learning?",
                "passages": [
                    "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",  # Most relevant
                    "Deep learning uses neural networks with multiple layers to solve complex problems.",  # Relevant
                    "Python is a programming language used for data science and AI development.",  # Somewhat relevant
                    "Apple makes great computers and smartphones for everyday use.",  # Not relevant
                    "The weather today is sunny with a chance of rain in the afternoon."  # Not relevant
                ],
                "expected_top_keywords": ["machine learning", "artificial intelligence", "learn"]
            },
            {
                "query": "Apple Silicon M-series processor performance",
                "passages": [
                    "Apple's M-series processors deliver exceptional performance with unified memory architecture.",  # Most relevant
                    "The M1 and M2 chips revolutionized Mac performance with ARM-based design.",  # Relevant
                    "Silicon Valley is home to many technology companies including Apple.",  # Somewhat relevant
                    "Apples are a healthy fruit rich in vitamins and fiber.",  # Not relevant
                    "Machine learning algorithms require significant computational power to train."  # Not relevant
                ],
                "expected_top_keywords": ["apple", "m-series", "processor", "performance", "m1", "m2"]
            }
        ]
        
        results = []
        
        try:
            for i, test_case in enumerate(test_cases):
                query = test_case["query"]
                passages = test_case["passages"]
                expected_keywords = test_case["expected_top_keywords"]
                
                payload = {
                    "query": query,
                    "passages": passages,
                    "top_k": len(passages)
                }
                
                async with self.session.post(f"{self.base_url}/api/v1/rerank/", json=payload) as resp:
                    if resp.status != 200:
                        raise Exception(f"HTTP {resp.status}: {await resp.text()}")
                    
                    data = await resp.json()
                    rerank_results = data.get("results", [])
                    
                    if not rerank_results:
                        raise Exception("No reranking results returned")
                    
                    # Check if top result contains expected keywords
                    top_result = rerank_results[0]
                    top_passage = top_result["text"].lower()
                    
                    keyword_matches = sum(1 for keyword in expected_keywords 
                                        if keyword.lower() in top_passage)
                    has_relevant_top = keyword_matches > 0
                    
                    # Check score ordering (descending)
                    scores = [r["score"] for r in rerank_results]
                    properly_ordered = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
                    
                    result = {
                        "test_case": i + 1,
                        "query": query,
                        "top_passage_index": top_result["index"],
                        "top_score": top_result["score"],
                        "keyword_matches": keyword_matches,
                        "has_relevant_top": has_relevant_top,
                        "properly_ordered": properly_ordered,
                        "all_scores": scores
                    }
                    results.append(result)
                    
                    status = "✅" if has_relevant_top and properly_ordered else "⚠️"
                    self.print_status(f"Test case {i+1}", 
                                    f"{status} relevant_top={has_relevant_top}, ordered={properly_ordered}")
            
            successful_tests = sum(1 for r in results if r["has_relevant_top"] and r["properly_ordered"])
            overall_success = successful_tests >= len(results) * 0.5  # 50% threshold
            
            self.print_status("Overall Reranking Quality", 
                            f"{'✅' if overall_success else '⚠️'} {successful_tests}/{len(results)} tests successful")
            
            return {
                "status": "success",
                "test_results": results,
                "successful_tests": successful_tests,
                "total_tests": len(results),
                "overall_quality": overall_success
            }
            
        except Exception as e:
            self.print_status("Reranking Quality", f"❌ FAILED: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_performance_benchmark(self) -> Dict[str, Any]:
        """⚡ Performance benchmarking"""
        self.print_section("⚡ Performance Benchmark")
        
        # Embedding latency test
        print(f"{Colors.BLUE}Testing embedding latency...{Colors.END}")
        latency_results = await self._benchmark_embedding_latency()
        
        # Throughput test
        print(f"{Colors.BLUE}Testing throughput across batch sizes...{Colors.END}")
        throughput_results = await self._benchmark_throughput()
        
        # Reranking performance
        print(f"{Colors.BLUE}Testing reranking performance...{Colors.END}")
        reranking_results = await self._benchmark_reranking()
        
        self.print_status("Embedding Latency", f"✅ {latency_results['mean_latency_ms']:.2f}ms avg")
        self.print_status("Peak Throughput", f"✅ {throughput_results['peak_throughput']:.1f} texts/sec")
        self.print_status("Reranking Latency", f"✅ {reranking_results['mean_latency_ms']:.2f}ms avg")
        
        return {
            "status": "success",
            "embedding_latency": latency_results,
            "throughput": throughput_results,
            "reranking_performance": reranking_results
        }
    
    async def _benchmark_embedding_latency(self, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark embedding latency"""
        latencies = []
        test_text = "Performance benchmark test sentence for latency measurement."
        
        # Warmup
        for _ in range(5):
            payload = {"inputs": [test_text]}
            async with self.session.post(f"{self.base_url}/embed", json=payload) as resp:
                await resp.json()
        
        # Actual benchmark
        for _ in range(iterations):
            start_time = time.perf_counter()
            payload = {"inputs": [test_text]}
            async with self.session.post(f"{self.base_url}/embed", json=payload) as resp:
                await resp.json()
            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)
        
        return {
            "mean_latency_ms": round(statistics.mean(latencies), 2),
            "median_latency_ms": round(statistics.median(latencies), 2),
            "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2)
        }
    
    async def _benchmark_throughput(self) -> Dict[str, Any]:
        """Benchmark throughput with different batch sizes"""
        batch_sizes = [1, 5, 10, 20]
        results = []
        
        for batch_size in batch_sizes:
            texts = [f"Throughput test text number {i}" for i in range(batch_size)]
            
            # Multiple runs for accuracy
            run_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                payload = {"inputs": texts}
                async with self.session.post(f"{self.base_url}/embed", json=payload) as resp:
                    await resp.json()
                run_time = time.perf_counter() - start_time
                run_times.append(run_time)
            
            avg_time = statistics.mean(run_times)
            throughput = batch_size / avg_time
            
            results.append({
                "batch_size": batch_size,
                "avg_time_seconds": round(avg_time, 4),
                "throughput_texts_per_second": round(throughput, 1)
            })
        
        return {
            "batch_results": results,
            "peak_throughput": max(r["throughput_texts_per_second"] for r in results),
            "optimal_batch_size": max(results, key=lambda x: x["throughput_texts_per_second"])["batch_size"]
        }
    
    async def _benchmark_reranking(self, iterations: int = 20) -> Dict[str, Any]:
        """Benchmark reranking performance"""
        latencies = []
        
        query = "Test query for reranking performance"
        passages = [
            "This is test passage number one for reranking.",
            "This is test passage number two for reranking.", 
            "This is test passage number three for reranking.",
            "This is test passage number four for reranking.",
            "This is test passage number five for reranking."
        ]
        
        # Warmup
        for _ in range(3):
            payload = {"query": query, "passages": passages}
            async with self.session.post(f"{self.base_url}/api/v1/rerank/", json=payload) as resp:
                await resp.json()
        
        # Benchmark
        for _ in range(iterations):
            start_time = time.perf_counter()
            payload = {"query": query, "passages": passages}
            async with self.session.post(f"{self.base_url}/api/v1/rerank/", json=payload) as resp:
                await resp.json()
            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)
        
        return {
            "mean_latency_ms": round(statistics.mean(latencies), 2),
            "median_latency_ms": round(statistics.median(latencies), 2),
            "passages_per_query": len(passages)
        }
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """🎯 Run complete validation suite"""
        start_time = time.time()
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}")
        print("🧠 MLX Embedding & Reranking Quality Validator")
        print("Testing your actual running server with configured model")
        print(f"{Colors.END}")
        
        results = {
            "validation_info": {
                "timestamp": datetime.now().isoformat(),
                "server_url": self.base_url,
                "validator_version": "1.0.0"
            }
        }
        
        try:
            # Run all tests
            results["server_health"] = await self.check_server_health()
            results["basic_embedding"] = await self.test_basic_embedding()
            results["semantic_similarity"] = await self.test_semantic_similarity()
            results["multilingual_support"] = await self.test_multilingual_support()
            results["reranking_quality"] = await self.test_reranking_quality()
            results["performance_benchmark"] = await self.test_performance_benchmark()
            
            # Calculate summary
            test_results = [
                results["basic_embedding"],
                results["semantic_similarity"],
                results["multilingual_support"],
                results["reranking_quality"],
                results["performance_benchmark"]
            ]
            
            passed_tests = sum(1 for test in test_results if test.get("status") == "success")
            total_tests = len(test_results)
            
            total_time = time.time() - start_time
            
            results["validation_summary"] = {
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "success_rate": f"{passed_tests/total_tests*100:.1f}%",
                "overall_status": "🎉 ALL TESTS PASSED" if passed_tests == total_tests else 
                                f"⚠️ {passed_tests}/{total_tests} TESTS PASSED",
                "total_validation_time": f"{total_time:.2f}s"
            }
            
            # Print summary
            self.print_section("🎯 Validation Summary")
            summary = results["validation_summary"]
            
            status_color = Colors.GREEN if passed_tests == total_tests else Colors.YELLOW
            print(f"{status_color}{summary['overall_status']}{Colors.END}")
            print(f"Success Rate: {summary['success_rate']}")
            print(f"Total Time: {summary['total_validation_time']}")
            
            if results["server_health"]["status"] == "healthy":
                info = results["server_health"]
                print(f"Backend: {info['backend']}")
                print(f"Model: {info['model']}")
                print(f"Device: {info['device']}")
            
            return results
            
        except Exception as e:
            results["validation_summary"] = {
                "overall_status": f"❌ VALIDATION FAILED: {str(e)}",
                "error": str(e)
            }
            print(f"\n{Colors.RED}❌ Validation failed: {str(e)}{Colors.END}")
            return results

async def main():
    parser = argparse.ArgumentParser(
        description="🧠 MLX Embedding & Reranking Quality Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/tests/validate-quality.py                                    # Test localhost:11436
  python3 tools/tests/validate-quality.py --url http://localhost:8080        # Custom URL
  python3 tools/tests/validate-quality.py --output validation-results.json   # Save results
  python3 tools/tests/validate-quality.py --quiet                           # Minimal output
        """
    )
    
    parser.add_argument("--url", default="http://localhost:11436",
                       help="API server URL (default: http://localhost:11436)")
    parser.add_argument("--output", type=Path,
                       help="Save validation results to JSON file")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output (errors only)")
    
    args = parser.parse_args()
    
    if args.quiet:
        # Redirect stdout to minimize output
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    try:
        async with EmbeddingQualityValidator(args.url) as validator:
            results = await validator.run_full_validation()
            
            if args.quiet:
                sys.stdout.close()
                sys.stdout = original_stdout
                
                # Print only summary in quiet mode
                summary = results.get("validation_summary", {})
                print(summary.get("overall_status", "❌ UNKNOWN STATUS"))
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Results saved to {args.output}")
            
            # Exit code based on results
            summary = results.get("validation_summary", {})
            if "ALL TESTS PASSED" in summary.get("overall_status", ""):
                sys.exit(0)
            else:
                sys.exit(1)
                
    except KeyboardInterrupt:
        if args.quiet:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"\n{Colors.YELLOW}⚠️ Validation interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        if args.quiet:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"\n{Colors.RED}❌ Validation error: {str(e)}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
