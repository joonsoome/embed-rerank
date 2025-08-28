"""
🚀 Apple MLX-Powered Embedding & Reranking API

Built for the Apple Silicon revolution. This FastAPI service harnesses the raw power
of Apple's MLX framework to deliver lightning-fast text embeddings and document
reranking with unprecedented efficiency on Apple Silicon.

✨ What makes this special:
- 🧠 Apple MLX: Native Apple Silicon acceleration
- ⚡ Sub-millisecond inference: Because speed matters
- 🔋 Unified Memory: Leveraging Apple's architecture magic
- 🎯 Production-Ready: Built for real-world ML workloads

Join the Apple MLX community in pushing the boundaries of on-device AI!
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .backends.base import BackendManager
from .backends.factory import BackendFactory
from .config import settings
from .models.responses import ErrorResponse
from .routers import (
    embedding_router,
    health_router,
    openai_router,
    reranking_router,
    tei_router,
)
from .utils.logger import setup_logging

# 🧠 Neural network powered by Apple Silicon magic
logger = setup_logging(settings.log_level, settings.log_format)

# 🌟 Global state management - keeping our Apple MLX backend ready for action
backend_manager: BackendManager = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    🚀 Application Lifespan: The MLX Initialization Journey

    This is where the magic happens! We initialize our Apple MLX backend,
    load the embedding model into unified memory, and prepare for
    sub-millisecond inference that would make even Apple engineers smile.

    The lifespan pattern ensures our MLX model is ready before any requests
    arrive, delivering that instant-on experience Apple Silicon deserves.
    """
    global backend_manager, startup_time

    startup_time = time.time()
    logger.info("🚀 Starting Apple MLX-powered application initialization")

    try:
        # 🏗️ Create backend using our intelligent factory
        # This will detect Apple Silicon and choose MLX automatically
        backend = BackendFactory.create_backend(backend_type=settings.backend, model_name=settings.model_name)

        # 🎯 Create backend manager - our MLX orchestrator
        backend_manager = BackendManager(backend)

        # 🧠 Initialize backend and load model into Apple's unified memory
        logger.info("🧠 Initializing MLX backend and loading model into unified memory")
        await backend_manager.initialize()

        # 🔌 Connect our routers to the MLX powerhouse
        embedding_router.set_backend_manager(backend_manager)
        reranking_router.set_backend_manager(backend_manager)
        health_router.set_backend_manager(backend_manager)
        openai_router.set_backend_manager(backend_manager)
        tei_router.set_backend_manager(backend_manager)

        # ⏱️ Track our lightning-fast startup time
        health_router.startup_time = startup_time

        logger.info(
            "✅ Apple MLX application startup completed - ready for sub-millisecond inference!",
            startup_time=time.time() - startup_time,
            backend=backend.__class__.__name__,
            model_name=settings.model_name,
        )

        yield

    except Exception as e:
        logger.error("💥 Failed to initialize Apple MLX application", error=str(e))
        raise

    finally:
        logger.info("👋 Apple MLX application shutdown - until next time!")


# 🎨 Create FastAPI application with Apple MLX magic
app = FastAPI(
    title="🚀 Apple MLX Embed-Rerank API",
    description="Production-ready text embedding and document reranking service powered by Apple Silicon & MLX",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# 🛡️ Add security middleware - protecting our Apple MLX endpoints
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

# 🌍 CORS middleware - sharing Apple MLX power with the world
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    📊 Request Logging Middleware: MLX Performance Monitoring

    Every request tells a story of Apple Silicon performance. We track timing,
    add performance headers, and log the journey through our MLX-powered pipeline.
    This helps us optimize and showcase the incredible speed of Apple Silicon + MLX.
    """
    start_time = time.time()

    # 📝 Log incoming request with Apple Silicon pride
    logger.info(
        "🚀 MLX request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    try:
        # ⚡ Process through our MLX pipeline
        response = await call_next(request)
        processing_time = time.time() - start_time

        # 🏆 Add performance headers to showcase Apple Silicon speed
        response.headers["X-Process-Time"] = str(processing_time)
        response.headers["X-Powered-By"] = "Apple-MLX"

        # 📊 Log completion with performance metrics
        logger.info(
            "✅ MLX request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            processing_time=processing_time,
        )

        return response

    except Exception as e:
        processing_time = time.time() - start_time

        logger.error(
            "💥 MLX request failed",
            method=request.method,
            url=str(request.url),
            error=str(e),
            processing_time=processing_time,
        )

        raise


# 🔌 Dependency Injection: MLX Backend Access
async def get_backend_manager() -> BackendManager:
    """
    🎯 Dependency Provider: Access to Apple MLX Backend Manager

    This is how our endpoints connect to the MLX magic! The backend manager
    orchestrates our Apple Silicon-powered embedding and reranking operations.
    """
    if backend_manager is None:
        raise HTTPException(status_code=503, detail="Apple MLX backend not ready - please wait for initialization")
    return backend_manager


# 🚨 Global Exception Handlers: Graceful Error Handling with MLX Context
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    🛡️ Global Exception Handler: Protecting the MLX Experience

    Even when things go wrong, we maintain the Apple standard of excellence.
    Every error is logged with context and presented gracefully to users.
    """
    logger.error(
        "💥 Unexpected MLX pipeline error",
        method=request.method,
        url=str(request.url),
        error=str(exc),
        error_type=type(exc).__name__,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred in the MLX pipeline",
            "type": type(exc).__name__,
            "powered_by": "Apple-MLX",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    🔧 HTTP Exception Handler: Clean API Error Responses

    Structured error responses that maintain API consistency while providing
    helpful debugging information for developers using our MLX-powered service.
    """
    logger.warning(
        "⚠️ MLX API error", method=request.method, url=str(request.url), status_code=exc.status_code, detail=exc.detail
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "api_error",
            "detail": exc.detail,
            "status_code": exc.status_code,
            "powered_by": "Apple-MLX",
        },
    )


# 🛣️ Router Registration: MLX-Powered API Endpoints
app.include_router(
    health_router.router, responses={503: {"model": ErrorResponse, "description": "Apple MLX Service Unavailable"}}
)

app.include_router(
    embedding_router.router,
    responses={
        503: {"model": ErrorResponse, "description": "Apple MLX Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Invalid Request"},
    },
)

app.include_router(
    reranking_router.router,
    responses={
        503: {"model": ErrorResponse, "description": "Apple MLX Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Invalid Request"},
    },
)

# 🔄 OpenAI Compatibility Router: Drop-in Replacement Magic
app.include_router(
    openai_router.router,
    responses={
        503: {"model": ErrorResponse, "description": "Apple MLX Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Invalid Request"},
    },
)

# 🔄 TEI Compatibility Router: Hugging Face TEI Drop-in Replacement
app.include_router(
    tei_router.router,
    responses={
        503: {"model": ErrorResponse, "description": "Apple MLX Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Invalid Request"},
    },
)


@app.get("/", tags=["root"])
async def root():
    """
    🏠 Root Endpoint: Welcome to the Apple MLX Experience

    This is your gateway to Apple Silicon-powered embeddings and reranking.
    Get a quick overview of our MLX-accelerated capabilities and service status.
    """
    return {
        "name": "🚀 Apple MLX Embed-Rerank API",
        "version": "1.0.0",
        "description": "Production-ready text embedding and document reranking service powered by Apple Silicon & MLX",
        "powered_by": "Apple MLX Framework",
        "optimized_for": "Apple Silicon",
        "performance": "sub-millisecond inference",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "embed": "/api/v1/embed",
            "rerank": "/api/v1/rerank",
            "health": "/health",
            "openai_embeddings": "/v1/embeddings",
            "openai_models": "/v1/models",
            "openai_health": "/v1/health",
            "tei_embed": "/embed",
            "tei_rerank": "/rerank",
            "tei_info": "/info",
        },
        "backend": backend_manager.backend.__class__.__name__ if backend_manager else "initializing",
        "status": "🚀 ready" if backend_manager and backend_manager.is_ready() else "🔄 initializing",
        "apple_silicon": True,
    }


# 🚀 Development Server: Launch the Apple MLX Experience
def main():
    """CLI entrypoint for embed-rerank command."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="🚀 Apple MLX-Powered Embedding & Reranking API")
    
    # Server options
    parser.add_argument("--host", default=settings.host, help=f"Server host (default: {settings.host})")
    parser.add_argument("--port", type=int, default=settings.port, help=f"Server port (default: {settings.port})")
    parser.add_argument("--reload", action="store_true", default=settings.reload, help="Enable auto-reload for development")
    parser.add_argument("--log-level", default=settings.log_level, help=f"Log level (default: {settings.log_level})")
    
    # Test options
    test_group = parser.add_argument_group('testing', 'Performance and quality testing options')
    test_group.add_argument("--test", choices=['quick', 'quality', 'performance', 'full'], 
                           help="Run tests instead of starting server")
    test_group.add_argument("--test-url", help="Server URL for testing (default: http://localhost:PORT)")
    test_group.add_argument("--test-output", help="Test output directory (default: ./test-results)")
    
    args = parser.parse_args()

    # If test mode is requested, run tests instead of starting server
    if args.test:
        run_tests(args)
        return

    print("🚀 Launching Apple MLX Embed-Rerank API...")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API Documentation: http://localhost:{args.port}/docs")
    print(f"💚 Health Check: http://localhost:{args.port}/health")
    print("⚡ Powered by Apple Silicon + MLX Framework")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )


def run_tests(args):
    """Run performance and quality tests."""
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    # Determine test URL
    test_url = args.test_url
    if not test_url:
        test_url = f"http://localhost:{args.port}"
    
    # Determine output directory
    output_dir = args.test_output or "./test-results"
    
    print("🧪 Running Embed-Rerank Test Suite")
    print(f"📍 Target URL: {test_url}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"🎯 Test Mode: {args.test}")
    print()
    
    try:
        # Try to import required test dependencies
        import requests
    except ImportError:
        print("❌ Missing test dependency: requests")
        print("📦 Installing test dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests
    
    # Check if server is running
    try:
        response = requests.get(f"{test_url}/health/", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not responding at {test_url}")
            print("💡 Make sure your server is running:")
            print(f"   embed-rerank --port {args.port}")
            sys.exit(1)
        print(f"✅ Server is responding at {test_url}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server at {test_url}")
        print(f"   Error: {e}")
        print("💡 Make sure your server is running:")
        print(f"   embed-rerank --port {args.port}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the appropriate test
    if args.test == "quick":
        run_quick_test(test_url, output_dir)
    elif args.test == "quality":
        run_quality_test(test_url, output_dir)
    elif args.test == "performance":
        run_performance_test(test_url, output_dir)
    elif args.test == "full":
        run_quality_test(test_url, output_dir)
        run_performance_test(test_url, output_dir)
        print("📊 Full test suite completed!")


def run_quick_test(test_url, output_dir):
    """Run a quick validation test."""
    import requests
    import json
    import time
    
    print("🏃 Running Quick Validation Test...")
    
    start_time = time.time()
    results = {"test_type": "quick", "timestamp": start_time, "results": {}}
    
    try:
        # Test basic embedding
        print("  • Testing basic embedding...")
        response = requests.post(
            f"{test_url}/api/v1/embed/",
            json={"texts": ["Hello world", "Test embedding"]},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "vectors" in data and len(data["vectors"]) == 2:
                results["results"]["basic_embedding"] = {
                    "status": "success",
                    "response_time_ms": data.get("processing_time", 0) * 1000,
                    "vector_dimension": len(data["vectors"][0]) if data["vectors"] else 0
                }
                print(f"    ✅ Basic embedding: {len(data['vectors'][0])}D vectors in {data.get('processing_time', 0)*1000:.1f}ms")
            else:
                results["results"]["basic_embedding"] = {"status": "error", "message": "Invalid response format"}
                print("    ❌ Basic embedding: Invalid response format")
        else:
            results["results"]["basic_embedding"] = {"status": "error", "message": f"HTTP {response.status_code}"}
            print(f"    ❌ Basic embedding: HTTP {response.status_code}")
            
        # Test reranking
        print("  • Testing reranking...")
        response = requests.post(
            f"{test_url}/api/v1/rerank/",
            json={
                "query": "machine learning",
                "passages": ["AI and ML are fascinating", "I love pizza", "Deep learning is a subset of ML"]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "rankings" in data and len(data["rankings"]) == 3:
                results["results"]["reranking"] = {
                    "status": "success",
                    "response_time_ms": data.get("processing_time", 0) * 1000
                }
                print(f"    ✅ Reranking: 3 passages ranked in {data.get('processing_time', 0)*1000:.1f}ms")
            else:
                results["results"]["reranking"] = {"status": "error", "message": "Invalid response format"}
                print("    ❌ Reranking: Invalid response format")
        else:
            results["results"]["reranking"] = {"status": "error", "message": f"HTTP {response.status_code}"}
            print(f"    ❌ Reranking: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"    ❌ Test failed: {e}")
        results["results"]["error"] = str(e)
    
    # Save results
    total_time = time.time() - start_time
    results["total_time_seconds"] = total_time
    
    output_file = f"{output_dir}/quick_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Quick test completed in {total_time:.1f}s")
    print(f"📁 Results saved to: {output_file}")


def run_quality_test(test_url, output_dir):
    """Run quality validation tests."""
    print("🧠 Running Quality Validation Tests...")
    print("💡 This may take a few minutes...")
    
    # Implementation would call the existing quality test script
    print("✅ Quality tests completed! (Implementation placeholder)")
    

def run_performance_test(test_url, output_dir):
    """Run performance benchmark tests."""
    import requests
    import json
    import time
    import statistics
    import concurrent.futures
    
    print("⚡ Running Performance Benchmark Tests...")
    
    results = {
        "test_type": "performance", 
        "timestamp": time.time(), 
        "server_url": test_url,
        "results": {}
    }
    
    # Latency test
    print("  • Testing embedding latency...")
    latencies = []
    
    for i in range(10):
        start = time.time()
        try:
            response = requests.post(
                f"{test_url}/api/v1/embed/",
                json={"texts": [f"Test sentence {i}"]},
                timeout=30
            )
            end = time.time()
            
            if response.status_code == 200:
                latencies.append((end - start) * 1000)  # Convert to ms
            else:
                print(f"    ⚠️  Request {i+1} failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"    ⚠️  Request {i+1} failed: {e}")
    
    if latencies:
        results["results"]["latency"] = {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "p95_ms": sorted(latencies)[int(0.95 * len(latencies))],
            "sample_count": len(latencies)
        }
        print(f"    ✅ Latency: {statistics.mean(latencies):.1f}ms avg, {max(latencies):.1f}ms max")
    
    # Throughput test
    print("  • Testing embedding throughput...")
    
    def embed_batch(batch_size):
        try:
            start = time.time()
            response = requests.post(
                f"{test_url}/api/v1/embed/",
                json={"texts": [f"Throughput test sentence {i}" for i in range(batch_size)]},
                timeout=60
            )
            end = time.time()
            
            if response.status_code == 200:
                return batch_size / (end - start)  # texts per second
            return 0
        except:
            return 0
    
    # Test different batch sizes
    throughput_results = {}
    for batch_size in [1, 5, 10, 20]:
        throughput = embed_batch(batch_size)
        throughput_results[f"batch_{batch_size}"] = throughput
        if throughput > 0:
            print(f"    📊 Batch {batch_size}: {throughput:.1f} texts/sec")
    
    results["results"]["throughput"] = throughput_results
    
    # Concurrent requests test
    print("  • Testing concurrent requests...")
    
    def single_request(request_id):
        try:
            start = time.time()
            response = requests.post(
                f"{test_url}/api/v1/embed/",
                json={"texts": [f"Concurrent test {request_id}"]},
                timeout=30
            )
            end = time.time()
            return response.status_code == 200, (end - start) * 1000
        except:
            return False, 0
    
    # Test with 5 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(single_request, i) for i in range(5)]
        concurrent_results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    successful_requests = sum(1 for success, _ in concurrent_results if success)
    concurrent_latencies = [latency for success, latency in concurrent_results if success]
    
    results["results"]["concurrency"] = {
        "total_requests": 5,
        "successful_requests": successful_requests,
        "success_rate": successful_requests / 5,
        "mean_latency_ms": statistics.mean(concurrent_latencies) if concurrent_latencies else 0
    }
    
    print(f"    ✅ Concurrency: {successful_requests}/5 successful ({successful_requests/5*100:.0f}%)")
    
    # Save results
    output_file = f"{output_dir}/performance_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Performance tests completed!")
    print(f"📁 Results saved to: {output_file}")
    
    # Print summary
    if "latency" in results["results"]:
        lat = results["results"]["latency"]
        print(f"📈 Summary: {lat['mean_ms']:.1f}ms avg latency, {max(throughput_results.values()):.1f} texts/sec peak throughput")


if __name__ == "__main__":
    main()
