#!/usr/bin/env python3
"""
Debug script to test reranking API directly
"""
import requests
import json

def test_reranking():
    url = "http://192.168.0.140:11436/api/v1/rerank/"
    payload = {
        "query": "machine learning",
        "passages": ["AI and ML are fascinating", "I love pizza", "Deep learning is a subset of ML"]
    }
    
    print("🔍 Testing reranking API...")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Response received!")
            print(f"Response keys: {list(data.keys())}")
            print(f"Has 'results' field: {'results' in data}")
            if 'results' in data:
                print(f"Results count: {len(data['results'])}")
                print(f"Results == 3: {len(data['results']) == 3}")
                
            print("\nFull response:")
            print(json.dumps(data, indent=2))
            
            # Test the exact condition from the test
            if "results" in data and len(data["results"]) == 3:
                print("\n🎉 TEST CONDITION PASSED!")
                print("The reranking test should work!")
            else:
                print("\n❌ TEST CONDITION FAILED!")
                print("This is why the test is failing.")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_reranking()
