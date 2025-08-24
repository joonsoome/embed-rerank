"""
FastAPI application for embed-rerank API service.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog

from .config import settings
from .backends.base import BackendManager
from .backends.factory import BackendFactory
from .routers import embedding_router, reranking_router, health_router
from .models.responses import ErrorResponse
from .utils.logger import setup_logging

# Setup logging
logger = setup_logging(settings.log_level, settings.log_format)

# Global state
backend_manager: BackendManager = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with backend initialization."""
    global backend_manager, startup_time

    startup_time = time.time()
    logger.info("Starting application initialization")

    try:
        # Create backend using factory
        backend = BackendFactory.create_backend(backend_type=settings.backend, model_name=settings.model_name)

        # Create backend manager
        backend_manager = BackendManager(backend)

        # Initialize backend (load model)
        logger.info("Initializing backend and loading model")
        await backend_manager.initialize()

        # Set backend manager in routers
        embedding_router.set_backend_manager(backend_manager)
        reranking_router.set_backend_manager(backend_manager)
        health_router.set_backend_manager(backend_manager)

        # Set startup time in health router
        health_router.startup_time = startup_time

        logger.info(
            "Application startup completed",
            startup_time=time.time() - startup_time,
            backend=backend.__class__.__name__,
            model_name=settings.model_name,
        )

        yield

    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise

    finally:
        logger.info("Application shutdown")


# Create FastAPI application with Context7 patterns
app = FastAPI(
    title="Embed-Rerank API",
    description="Production-ready text embedding and document reranking service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add middleware with Context7 recommended patterns
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log request and response details with processing time."""
    start_time = time.time()

    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    # Process request
    try:
        response = await call_next(request)
        processing_time = time.time() - start_time

        # Add processing time header
        response.headers["X-Process-Time"] = str(processing_time)

        # Log response
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            processing_time=processing_time,
        )

        return response

    except Exception as e:
        processing_time = time.time() - start_time

        logger.error(
            "Request failed", method=request.method, url=str(request.url), error=str(e), processing_time=processing_time
        )

        raise


# Dependency providers using Context7 patterns
async def get_backend_manager() -> BackendManager:
    """Dependency to get the backend manager instance."""
    if backend_manager is None:
        raise HTTPException(status_code=503, detail="Backend manager not initialized")
    return backend_manager


# Global exception handlers with Context7 patterns
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with structured logging."""
    logger.error(
        "Unhandled exception",
        method=request.method,
        url=str(request.url),
        error=str(exc),
        error_type=type(exc).__name__,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred",
            "type": type(exc).__name__,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.warning(
        "HTTP exception", method=request.method, url=str(request.url), status_code=exc.status_code, detail=exc.detail
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail, "status_code": exc.status_code},
    )


# Include routers with Context7 organization patterns
app.include_router(
    health_router.router, responses={503: {"model": ErrorResponse, "description": "Service Unavailable"}}
)

app.include_router(
    embedding_router.router,
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Bad Request"},
    },
)

app.include_router(
    reranking_router.router,
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Bad Request"},
    },
)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Embed-Rerank API",
        "version": "1.0.0",
        "description": "Production-ready text embedding and document reranking service",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {"embed": "/api/v1/embed", "rerank": "/api/v1/rerank", "health": "/health"},
        "backend": backend_manager.backend.__class__.__name__ if backend_manager else "not_initialized",
        "status": "ready" if backend_manager and backend_manager.is_ready() else "initializing",
    }


# Development server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
