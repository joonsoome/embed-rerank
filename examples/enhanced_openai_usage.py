#!/usr/bin/env python3
"""
🚀 Enhanced OpenAI-Compatible API Usage Examples

This script demonstrates how to use the enhanced OpenAI-compatible embeddings
endpoint with configurable Apple MLX arguments while maintaining full
OpenAI SDK compatibility.

✨ Features Demonstrated:
- 🔧 Configurable batch sizes
- 🎯 Normalization control
- 🧠 Backend preferences
- ⚡ Device preferences  
- ⏱️ Detailed timing metrics
- 🌐 Custom header support
"""

import asyncio
import time
from openai import OpenAI
import httpx
import json


# 🔗 Configure client for local Apple MLX service
BASE_URL = "http://localhost:9000/v1"

def basic_openai_compatibility():
    """
    🎯 Basic OpenAI SDK Usage (No Changes Needed)
    
    Your existing OpenAI code works unchanged - just point to our endpoint!
    """
    print("🔄 Testing Basic OpenAI Compatibility...")
    
    client = OpenAI(
        base_url=BASE_URL,
        api_key="dummy-key"  # Not needed for local service
    )
    
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=["Hello Apple MLX!", "Blazing fast embeddings"]
    )
    
    print(f"✅ Basic embeddings generated: {len(response.data)} vectors")
    print(f"📏 Vector dimension: {len(response.data[0].embedding)}")
    print(f"🔢 Tokens used: {response.usage.total_tokens}")
    print()


def enhanced_openai_with_args():
    """
    🚀 Enhanced OpenAI Usage with Apple MLX Configuration
    
    Use additional fields in the request body to control MLX behavior
    while maintaining OpenAI SDK compatibility.
    """
    print("🔧 Testing Enhanced OpenAI with MLX Arguments...")
    
    client = OpenAI(
        base_url=BASE_URL,
        api_key="dummy-key"
    )
    
    # 🌟 Enhanced request with MLX-specific options
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[
            "Apple Silicon delivers incredible performance",
            "MLX framework revolutionizes on-device AI",
            "Unified memory architecture enables fast inference"
        ],
        # 🚀 Enhanced MLX Arguments (Optional, Non-Breaking)
        extra_body={
            "batch_size": 16,           # 📦 Custom batch size
            "normalize": True,          # 🎯 Normalization control
            "backend_preference": "mlx", # 🧠 Prefer MLX backend
            "device_preference": "mps", # ⚡ Prefer Apple Silicon
            "return_timing": True,      # ⏱️ Include timing metrics
            "max_tokens_per_text": 512  # 📏 Token limit per text
        }
    )
    
    print(f"✅ Enhanced embeddings generated: {len(response.data)} vectors")
    print(f"📏 Vector dimension: {len(response.data[0].embedding)}")
    print(f"🔢 Tokens used: {response.usage.total_tokens}")
    
    # 🚀 Check for enhanced metrics (if return_timing=True)
    if hasattr(response.usage, 'mlx_processing_time'):
        print(f"⚡ MLX processing time: {response.usage.mlx_processing_time:.4f}s")
        print(f"🕐 Total processing time: {response.usage.total_processing_time:.4f}s")
        print(f"🧠 Backend used: {response.usage.backend_used}")
        print(f"💻 Device used: {response.usage.device_used}")
        print(f"📦 Batch size used: {response.usage.batch_size_used}")
    print()


def custom_headers_approach():
    """
    🌐 Using Custom Headers for MLX Configuration
    
    Alternative approach using X-MLX-* headers for enterprise integration
    while keeping request body perfectly OpenAI-compatible.
    """
    print("🌐 Testing Custom Headers Approach...")
    
    # 🔗 Use httpx for direct HTTP control
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy-key",
        # 🚀 MLX-specific configuration headers
        "X-MLX-Batch-Size": "64",
        "X-MLX-Normalize": "false", 
        "X-MLX-Backend": "mlx",
        "X-MLX-Device": "mps"
    }
    
    payload = {
        "model": "text-embedding-ada-002",
        "input": ["Custom headers enable enterprise integration"],
        "return_timing": True
    }
    
    response = httpx.post(
        f"{BASE_URL}/embeddings",
        headers=headers,
        json=payload,
        timeout=30.0
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Custom headers embeddings: {len(data['data'])} vectors")
        print(f"📏 Vector dimension: {len(data['data'][0]['embedding'])}")
        
        usage = data['usage']
        if 'backend_used' in usage:
            print(f"🧠 Backend used: {usage['backend_used']}")
            print(f"📦 Batch size used: {usage['batch_size_used']}")
    else:
        print(f"❌ Request failed: {response.status_code}")
    print()


def performance_comparison():
    """
    ⚡ Performance Comparison with Different Configurations
    
    Compare performance with different batch sizes and configurations
    to find optimal settings for your use case.
    """
    print("⚡ Testing Performance with Different Configurations...")
    
    client = OpenAI(base_url=BASE_URL, api_key="dummy-key")
    
    # 📝 Test data
    texts = [
        f"Performance test text number {i} for Apple MLX benchmarking"
        for i in range(50)
    ]
    
    configs = [
        {"batch_size": 8, "name": "Small Batch"},
        {"batch_size": 32, "name": "Medium Batch"}, 
        {"batch_size": 64, "name": "Large Batch"}
    ]
    
    for config in configs:
        start_time = time.time()
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts,
            extra_body={
                "batch_size": config["batch_size"],
                "return_timing": True
            }
        )
        
        total_time = time.time() - start_time
        
        print(f"🔧 {config['name']} (batch_size={config['batch_size']}):")
        print(f"   📊 Total time: {total_time:.4f}s")
        print(f"   ⚡ MLX time: {response.usage.mlx_processing_time:.4f}s")
        print(f"   📈 Throughput: {len(texts)/total_time:.1f} texts/sec")
    print()


async def async_usage_example():
    """
    🔄 Async Usage Example
    
    Demonstrate async usage for high-throughput applications.
    """
    print("🔄 Testing Async Usage...")
    
    async with httpx.AsyncClient() as client:
        tasks = []
        
        for i in range(5):
            payload = {
                "model": "text-embedding-ada-002",
                "input": [f"Async request {i} for Apple MLX"],
                "batch_size": 16,
                "return_timing": True
            }
            
            task = client.post(
                f"{BASE_URL}/embeddings",
                json=payload,
                timeout=30.0
            )
            tasks.append(task)
        
        # ⚡ Execute all requests concurrently
        responses = await asyncio.gather(*tasks)
        
        print(f"✅ Completed {len(responses)} async requests")
        for i, response in enumerate(responses):
            if response.status_code == 200:
                data = response.json()
                mlx_time = data['usage'].get('mlx_processing_time', 0)
                print(f"   Request {i}: {mlx_time:.4f}s MLX processing")
    print()


def main():
    """🚀 Run all examples"""
    print("🍎 Apple MLX Enhanced OpenAI Compatibility Examples")
    print("=" * 60)
    print()
    
    try:
        # 🔄 Basic compatibility test
        basic_openai_compatibility()
        
        # 🚀 Enhanced features test
        enhanced_openai_with_args()
        
        # 🌐 Custom headers test
        custom_headers_approach()
        
        # ⚡ Performance comparison
        performance_comparison()
        
        # 🔄 Async usage
        asyncio.run(async_usage_example())
        
        print("✅ All examples completed successfully!")
        print("🎯 Your OpenAI-compatible endpoint with MLX arguments is working perfectly!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("🔧 Make sure your Apple MLX service is running on localhost:9000")


if __name__ == "__main__":
    main()
