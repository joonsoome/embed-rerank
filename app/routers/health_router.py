"""
🩺 Apple MLX Health Check Router: System Monitoring at Silicon Speed

This router provides comprehensive health monitoring for our Apple MLX-powered
service. Real-time metrics showcase the incredible performance of Apple Silicon
unified memory architecture and MLX framework optimization.

🔍 Monitoring Features:
- Backend health with MLX performance metrics
- Dynamic model metadata exposure
- Apple Silicon resource utilization
- Sub-millisecond response time tracking

Monitor the power of Apple MLX in action! ⚡
"""

import datetime
import time

import psutil
from . import openai_router as _openai_router
from fastapi import APIRouter, Depends, HTTPException

from ..backends.base import BackendManager
from .. import __version__
from ..models.responses import ErrorResponse, HealthResponse

router = APIRouter(
    prefix="/health",
    tags=["🩺 Apple MLX Health"],
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)

# This will be set by the main app
_backend_manager: BackendManager = None
_embedding_service = None  # 🚀 Dynamic service reference
_reranker_backend_manager: BackendManager = None


def set_backend_manager(manager: BackendManager):
    """Set the backend manager instance."""
    global _backend_manager
    _backend_manager = manager


def set_embedding_service(service):
    """🚀 Set the configured embedding service for metadata access"""
    global _embedding_service
    _embedding_service = service


def set_reranker_backend_manager(manager: BackendManager):
    """Set the dedicated reranker backend manager (if available)."""
    global _reranker_backend_manager
    _reranker_backend_manager = manager


async def get_backend_manager() -> BackendManager:
    """Dependency to get the backend manager."""
    if _backend_manager is None:
        raise HTTPException(status_code=503, detail="Backend manager not initialized")
    return _backend_manager


@router.get("/", response_model=HealthResponse)
async def health_check(manager: BackendManager = Depends(get_backend_manager)):
    """
    🩺 Comprehensive Apple MLX Health Check

    Real-time monitoring of our Apple Silicon-powered embedding service.
    Get detailed insights into MLX performance, model metadata, and
    system resource utilization optimized for Apple's unified memory.

    🎯 Health Metrics Include:
    - Backend status with MLX optimization details
    - Dynamic model metadata (dimensions, tokens, etc.)
    - Apple Silicon resource utilization
    - Performance benchmarks and response times

    Returns:
        HealthResponse with comprehensive Apple MLX service status
    """
    try:
        # Get backend health status
        backend_health = await manager.health_check()

        # 🚀 Get dynamic service metadata if available
        service_metadata = {}
        if _embedding_service is not None:
            try:
                service_health = await _embedding_service.health_check()
                service_metadata = service_health.get("model_metadata", {})
            except Exception:
                pass  # Fallback gracefully

    # Get system resource information
        memory_info = psutil.virtual_memory()
        cpu_info = psutil.cpu_percent(interval=0.1)

        # Normalize backend type for compatibility
        backend_name = backend_health.get("backend", "unknown")
        if "MLX" in backend_name:
            backend_type = "mlx"
        elif "torch" in backend_name.lower():
            backend_type = "torch"
        else:
            backend_type = "cpu"

        # Top-level service information with MLX branding
        service_info = {
            "name": "embed-rerank",
            "version": __version__,
            "description": "Apple MLX-powered embedding & reranking service",
            "powered_by": "Apple MLX Framework",
            "optimized_for": "Apple Silicon",
        }

        # 🎯 Dynamic metadata integration
        embedding_dimension = (
            service_metadata.get("embedding_dimension")
            or backend_health.get("embedding_dim")
            or backend_health.get("embedding_dimension")
            or 4096
        )

        health_data = {
            "status": "healthy" if manager.is_ready() else "initializing",
            "uptime": time.time() - startup_time if 'startup_time' in globals() else 0,
            "timestamp": datetime.datetime.now(),
            "service": service_info,
            "backend": {
                "name": backend_health.get("backend", "unknown"),
                "type": backend_type,  # Use normalized type
                "status": backend_health.get("status", "unknown"),
                "model_loaded": backend_health.get("model_loaded", False),
                "model_name": backend_health.get("model_name"),
                "device": backend_health.get("device"),
                "load_time": backend_health.get("load_time"),
                "apple_mlx_optimized": backend_type == "mlx",
            },
            "system": {
                "cpu_percent": cpu_info,
                "memory_percent": memory_info.percent,
                "memory_available_gb": round(memory_info.available / (1024**3), 2),
                "memory_total_gb": round(memory_info.total / (1024**3), 2),
                "apple_silicon": backend_type == "mlx",
            },
            "performance": {
                "test_embedding_time": backend_health.get("test_embedding_time"),
                "embedding_dimension": embedding_dimension,
                "embedding_dim": embedding_dimension,  # Compatibility
                # 🚀 Dynamic metadata
                "max_tokens": service_metadata.get("max_tokens", 8192),
                "recommended_max_tokens": service_metadata.get("recommended_max_tokens", 2048),
                "warning_threshold": service_metadata.get("warning_threshold", 4096),
                "optimal_batch_size": service_metadata.get("optimal_batch_size", 32),
                "text_preprocessing": "enabled" if service_metadata else "disabled",
                "dynamic_config": "enabled" if service_metadata else "disabled",
            },
        }

        # ➕ Include reranker backend info if configured and ready
        try:
            # Prefer local reranker manager; fall back to OpenAI router's if available
            r_manager = _reranker_backend_manager
            if (r_manager is None or not r_manager.is_ready()) and hasattr(_openai_router, 'reranker_backend_manager'):
                r_manager = _openai_router.reranker_backend_manager

            if r_manager is not None and r_manager.is_ready():
                rerank_info = r_manager.get_current_backend_info()
                # Normalize backend type
                r_backend = rerank_info.get("backend") or rerank_info.get("name", "unknown")
                r_type = "mlx" if isinstance(r_backend, str) and "mlx" in r_backend.lower() else (
                    "torch" if isinstance(r_backend, str) and "torch" in r_backend.lower() else "cpu"
                )
                health_data["reranker"] = {
                    "name": rerank_info.get("name"),
                    "type": r_type,
                    "status": rerank_info.get("status"),
                    "model_name": rerank_info.get("model_name"),
                    "device": rerank_info.get("device"),
                    # Surface MLX experimental options when present
                    "pooling": rerank_info.get("pooling"),
                    "score_norm": rerank_info.get("score_norm"),
                    "method": rerank_info.get("rerank_method"),
                }
        except Exception:
            pass

        # 🎯 Add detailed model metadata if available
        if service_metadata:
            health_data["model_metadata"] = service_metadata

        # Overall status determination
        if not manager.is_ready():
            health_data["status"] = "not_ready"
        elif backend_health.get("status") == "unhealthy":
            health_data["status"] = "unhealthy"
        elif memory_info.percent > 90:
            health_data["status"] = "warning"

        return HealthResponse(**health_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/ready")
async def readiness_check(manager: BackendManager = Depends(get_backend_manager)):
    """
    Readiness probe for container orchestration.

    Returns:
        Simple status indicating if service is ready to handle requests
    """
    if manager.is_ready():
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    Liveness probe for container orchestration.

    Returns:
        Simple status indicating if service is alive
    """
    return {"status": "alive"}


# Global startup time tracking
startup_time = time.time()
