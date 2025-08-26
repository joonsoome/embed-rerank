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
if __name__ == "__main__":
    import uvicorn

    print("🚀 Launching Apple MLX Embed-Rerank API...")
    print(f"📍 Server will be available at: http://{settings.host}:{settings.port}")
    print("📚 API Documentation: http://localhost:9000/docs")
    print("💚 Health Check: http://localhost:9000/health")
    print("⚡ Powered by Apple Silicon + MLX Framework")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
