"""
Health check router for monitoring system status.
"""

import datetime
import time

import psutil
from fastapi import APIRouter, Depends, HTTPException

from ..backends.base import BackendManager
from ..models.responses import ErrorResponse, HealthResponse

router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)

# This will be set by the main app
_backend_manager: BackendManager = None


def set_backend_manager(manager: BackendManager):
    """Set the backend manager instance."""
    global _backend_manager
    _backend_manager = manager


async def get_backend_manager() -> BackendManager:
    """Dependency to get the backend manager."""
    if _backend_manager is None:
        raise HTTPException(status_code=503, detail="Backend manager not initialized")
    return _backend_manager


@router.get("/", response_model=HealthResponse)
async def health_check(manager: BackendManager = Depends(get_backend_manager)):
    """
    Comprehensive health check endpoint.

    Returns:
        HealthResponse with system status, backend info, and resource usage
    """
    try:
        # Get backend health status
        backend_health = await manager.health_check()

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

        # Top-level service information
        service_info = {
            "name": "embed-rerank",
            "version": "1.0.0",
            "description": "Embedding & reranking service",
        }

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
            },
            "system": {
                "cpu_percent": cpu_info,
                "memory_percent": memory_info.percent,
                "memory_available_gb": round(memory_info.available / (1024**3), 2),
                "memory_total_gb": round(memory_info.total / (1024**3), 2),
            },
            "performance": {
                "test_embedding_time": backend_health.get("test_embedding_time"),
                # Provide both keys for compatibility
                "embedding_dimension": backend_health.get("embedding_dim") or backend_health.get("embedding_dimension"),
                "embedding_dim": backend_health.get("embedding_dim") or backend_health.get("embedding_dimension"),
            },
        }

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
