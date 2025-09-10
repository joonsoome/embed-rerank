"""
Test script for Cohere-compatible reranking endpoints.

🚀 Testing MLX-powered Cohere compatibility!
"""

import json
import os
import pytest
import requests
import time
from typing import Dict, Any

# Server configuration - use environment variable or default
BASE_URL = os.getenv("TEST_SERVER_URL", "http://localhost:9000")

def test_cohere_v1_rerank():
    """Test Cohere v1 rerank endpoint."""
    
    url = f"{BASE_URL}/v1/rerank"
    payload = {
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing helps computers understand text.",
            "Cats are fluffy animals that like to sleep.",
            "Python is a popular programming language."
        ],
        "top_n": 3,
        "return_documents": True
    }
    
    response = requests.post(url, json=payload, timeout=30)
    
    # Check response status
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    result = response.json()
    
    # Validate Cohere format
    assert "results" in result, "Response missing 'results' field"
    assert "meta" in result, "Response missing 'meta' field"
    assert len(result["results"]) == 3, f"Expected 3 results, got {len(result['results'])}"
    
    for i, res in enumerate(result["results"]):
        assert "index" in res, f"Result {i} missing 'index' field"
        assert "relevance_score" in res, f"Result {i} missing 'relevance_score' field"
        assert isinstance(res["index"], int), f"Result {i} index should be int"
        assert isinstance(res["relevance_score"], (int, float)), f"Result {i} relevance_score should be numeric"


def test_cohere_v2_rerank():
    """Test Cohere v2 rerank endpoint."""
    
    url = f"{BASE_URL}/v2/rerank"
    payload = {
        "query": "Apple Silicon performance",
        "documents": [
            "Apple Silicon chips offer exceptional performance per watt.",
            "MLX framework is optimized for Apple Silicon.",
            "Embedding models run efficiently on M1 and M2 chips.",
            "Traditional x86 processors consume more power.",
            "GPU acceleration is important for machine learning."
        ],
        "top_n": 2,
        "return_documents": False
    }
    
    response = requests.post(url, json=payload, timeout=30)
    
    # Check response status
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    result = response.json()
    
    # Validate Cohere format
    assert "results" in result, "Response missing 'results' field"
    assert "meta" in result, "Response missing 'meta' field"
    assert len(result["results"]) == 2, f"Expected 2 results, got {len(result['results'])}"
    
    for i, res in enumerate(result["results"]):
        assert "index" in res, f"Result {i} missing 'index' field"
        assert "relevance_score" in res, f"Result {i} missing 'relevance_score' field"
        # Should not have document field when return_documents=False
        assert res.get("document") is None, f"Result {i} should not have document field when return_documents=False"
        assert isinstance(res["index"], int), f"Result {i} index should be int"
        assert isinstance(res["relevance_score"], (int, float)), f"Result {i} relevance_score should be numeric"


def test_cohere_return_documents():
    """Test Cohere API with return_documents=True."""
    url = f"{BASE_URL}/v1/rerank"
    payload = {
        "query": "machine learning",
        "documents": [
            "Machine learning is AI subset",
            "Deep learning uses neural networks"
        ],
        "top_n": 2,
        "return_documents": True
    }
    
    response = requests.post(url, json=payload, timeout=30)
    assert response.status_code == 200
    
    result = response.json()
    assert "results" in result
    assert len(result["results"]) == 2
    
    for res in result["results"]:
        assert "document" in res, "Expected document field when return_documents=True"
        assert "text" in res["document"], "Document should have text field"


def test_performance_comparison():
    """Compare performance across different API formats."""
    
    query = "Natural language processing and machine learning"
    documents = [
        "Natural language processing (NLP) is a branch of AI that helps computers understand human language.",
        "Machine learning algorithms can be trained on large datasets to make predictions.",
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Reinforcement learning involves training agents through rewards and penalties.",
        "Statistical methods form the foundation of many machine learning techniques.",
        "Python is widely used for machine learning and data science applications.",
        "The cat sat on the mat in the sunny garden.",
        "Today is a beautiful day for going to the beach.",
        "Quantum computing could revolutionize certain computational problems."
    ]
    
    # Test different endpoints
    endpoints = [
        ("/api/v1/rerank/", {
            "query": query,
            "passages": documents,
            "top_k": 5
        }),
        ("/v1/rerank", {
            "query": query,
            "documents": documents,
            "top_n": 5
        }),
        ("/v2/rerank", {
            "query": query,
            "documents": documents,
            "top_n": 5
        })
    ]
    
    for endpoint, payload in endpoints:
        url = f"{BASE_URL}{endpoint}"
        times = []
        
        # Warm up
        requests.post(url, json=payload)
        
        # Measure performance
        for _ in range(3):
            start_time = time.time()
            response = requests.post(url, json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                times.append((end_time - start_time) * 1000)
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"   🚀 {endpoint}: {avg_time:.2f}ms average")
        else:
            print(f"   ❌ {endpoint}: Failed")


if __name__ == "__main__":
    # Support running as standalone script for development
    import sys
    import os
    
    # Set base URL from environment or use default
    BASE_URL = os.getenv("TEST_SERVER_URL", "http://localhost:9000")
    
    print("🎯 Cohere API Compatibility Test Suite")
    print(f"🔗 Testing server: {BASE_URL}")
    print("=" * 50)
    
    try:
        test_cohere_v1_rerank()
        print("✅ Cohere v1 test passed")
        
        test_cohere_v2_rerank()
        print("✅ Cohere v2 test passed")
        
        test_cohere_return_documents()
        print("✅ Cohere return_documents test passed")
        
        test_performance_comparison()
        print("✅ Performance comparison completed")
        
        print("\n🎉 All Cohere compatibility tests passed! MLX + Cohere = ⚡")
    
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
