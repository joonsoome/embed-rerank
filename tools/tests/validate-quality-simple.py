#!/usr/bin/env python3
"""
🧠 MLX Embedding & Reranking Quality Validator

Comprehensive quality validation for your MLX-powered embedding and reranking API.
Tests your actual running server with your configured model settings.

Usage:
    python3 tools/tests/validate-quality-simple.py
    python3 tools/tests/validate-quality-simple.py --url http://localhost:11436 --output results.json
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

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0

def vector_norm(vector: List[float]) -> float:
    """Calculate L2 norm of a vector"""
    return math.sqrt(sum(x * x for x in vector))

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
    
    def test_basic_embedding(self) -> Dict[str, Any]:
        """🔤 Test basic embedding functionality"""
        self.print_section("🔤 Basic Embedding Test")
        
        test_text = "Hello, MLX world!"
        
        try:
            payload = {"inputs": [test_text]}
            resp = self.session.post(f"{self.base_url}/embed", json=payload, timeout=30)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text}")
            
            embeddings = resp.json()
            
            if not embeddings or len(embeddings) == 0:
                raise Exception("No embeddings returned")
            
            vector = embeddings[0]
            dimension = len(vector)
            norm = vector_norm(vector)
            
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
                "std_dev": float(statistics.stdev(vector)) if len(vector) > 1 else 0.0
            }
            
        except Exception as e:
            self.print_status("Basic Embedding", f"❌ FAILED: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def test_semantic_similarity(self) -> Dict[str, Any]:
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
                resp = self.session.post(f"{self.base_url}/embed", json=payload, timeout=30)
                data = resp.json()
                v1, v2 = data[0], data[1]
                sim_similarity = cosine_similarity(v1, v2)
                
                # Test dissimilar pair
                payload = {"inputs": list(dissimilar_pair)}
                resp = self.session.post(f"{self.base_url}/embed", json=payload, timeout=30)
                data = resp.json()
                v1, v2 = data[0], data[1]
                dissim_similarity = cosine_similarity(v1, v2)
                
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
    
    def test_multilingual_support(self) -> Dict[str, Any]:
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
            
            resp = self.session.post(f"{self.base_url}/embed", json=payload, timeout=30)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}")
            
            embeddings = resp.json()
            
            # Check each embedding
            language_results = []
            for i, (lang, text) in enumerate(multilingual_greetings):
                vector = embeddings[i]
                norm = vector_norm(vector)
                non_zero_count = sum(1 for x in vector if abs(x) > 1e-10)
                
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
    
    def test_reranking_quality(self) -> Dict[str, Any]:
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
                
                resp = self.session.post(f"{self.base_url}/api/v1/rerank/", json=payload, timeout=30)
                if resp.status_code != 200:
                    raise Exception(f"HTTP {resp.status_code}: {resp.text}")
                
                data = resp.json()
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
    
    def test_performance_quick(self) -> Dict[str, Any]:
        """⚡ Quick performance test"""
        self.print_section("⚡ Quick Performance Test")
        
        # Embedding latency test
        print(f"{Colors.BLUE}Testing embedding latency...{Colors.END}")
        
        test_text = "Performance benchmark test sentence for latency measurement."
        payload = {"inputs": [test_text]}
        
        # Warmup
        for _ in range(3):
            self.session.post(f"{self.base_url}/embed", json=payload, timeout=30)
        
        # Test latency
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            resp = self.session.post(f"{self.base_url}/embed", json=payload, timeout=30)
            if resp.status_code == 200:
                latency = (time.perf_counter() - start) * 1000  # ms
                latencies.append(latency)
        
        if latencies:
            mean_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            self.print_status("Embedding Latency", f"✅ {mean_latency:.2f}ms avg ({min_latency:.2f}-{max_latency:.2f}ms)")
        
        # Throughput test
        print(f"{Colors.BLUE}Testing throughput...{Colors.END}")
        
        batch_sizes = [1, 5, 10]
        throughput_results = []
        
        for batch_size in batch_sizes:
            texts = [f"Test text {i}" for i in range(batch_size)]
            payload = {"inputs": texts}
            
            # Measure multiple runs
            times = []
            for _ in range(5):
                start = time.perf_counter()
                resp = self.session.post(f"{self.base_url}/embed", json=payload, timeout=30)
                if resp.status_code == 200:
                    times.append(time.perf_counter() - start)
            
            if times:
                avg_time = statistics.mean(times)
                throughput = batch_size / avg_time
                throughput_results.append(throughput)
                self.print_status(f"Batch size {batch_size}", f"✅ {throughput:.1f} texts/sec")
        
        peak_throughput = max(throughput_results) if throughput_results else 0
        
        return {
            "status": "success",
            "mean_latency_ms": round(mean_latency, 2) if latencies else 0,
            "peak_throughput": round(peak_throughput, 1)
        }
    
    def run_full_validation(self) -> Dict[str, Any]:
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
            results["server_health"] = self.check_server_health()
            results["basic_embedding"] = self.test_basic_embedding()
            results["semantic_similarity"] = self.test_semantic_similarity()
            results["multilingual_support"] = self.test_multilingual_support()
            results["reranking_quality"] = self.test_reranking_quality()
            results["performance_quick"] = self.test_performance_quick()
            
            # Calculate summary
            test_results = [
                results["basic_embedding"],
                results["semantic_similarity"],
                results["multilingual_support"],
                results["reranking_quality"],
                results["performance_quick"]
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

def main():
    parser = argparse.ArgumentParser(
        description="🧠 MLX Embedding & Reranking Quality Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/tests/validate-quality-simple.py                                    # Test localhost:11436
  python3 tools/tests/validate-quality-simple.py --url http://localhost:8080        # Custom URL
  python3 tools/tests/validate-quality-simple.py --output validation-results.json   # Save results
  python3 tools/tests/validate-quality-simple.py --quiet                           # Minimal output
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
        validator = EmbeddingQualityValidator(args.url)
        results = validator.run_full_validation()
        
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
    main()
