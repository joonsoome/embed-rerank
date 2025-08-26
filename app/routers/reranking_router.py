"""
Reranking router for document reranking operations.
"""

import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..backends.base import BackendManager
from ..models.requests import RerankRequest
from ..models.responses import ErrorResponse, RerankResponse
from ..services.reranking_service import RerankingService


def json_encoder(obj):
    """Custom JSON encoder for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def filter_none_values(data):
    """Recursively filter None values from a dictionary."""
    if isinstance(data, dict):
        return {k: filter_none_values(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [filter_none_values(item) for item in data]
    else:
        return data


router = APIRouter(
    prefix="/api/v1/rerank",
    tags=["reranking"],
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
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


async def get_reranking_service(manager: BackendManager = Depends(get_backend_manager)) -> RerankingService:
    """Dependency to get the reranking service."""
    if not manager.is_ready():
        raise HTTPException(status_code=503, detail="Backend not ready. Please wait for model initialization.")
    return RerankingService(manager)


@router.post("/")
async def rerank_passages(request: RerankRequest, service: RerankingService = Depends(get_reranking_service)):
    """
    Rerank passages based on relevance to the query.

    Args:
        request: RerankRequest containing query, passages, and options
        service: RerankingService dependency

    Returns:
        RerankResponse with ranked results and metadata

    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Perform reranking using the service
        response = await service.rerank_passages(request)

        # Convert to dict and filter None values
        response_dict = response.model_dump()
        filtered_response = filter_none_values(response_dict)

        # Add backward-compatible num_passages field
        filtered_response["num_passages"] = len(filtered_response.get("results", []))

        # Create JSON with custom encoder for datetime
        json_content = json.dumps(filtered_response, default=json_encoder)
        return JSONResponse(content=json.loads(json_content))

    except ValueError as e:
        # Input validation errors
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    except RuntimeError as e:
        # Backend/model errors
        raise HTTPException(status_code=503, detail=f"Service error: {str(e)}")

    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/batch", response_model=List[RerankResponse])
async def batch_rerank_passages(
    requests: List[RerankRequest], service: RerankingService = Depends(get_reranking_service)
):
    """
    Perform batch reranking for multiple queries.

    Args:
        requests: List of RerankRequest objects
        service: RerankingService dependency

    Returns:
        List of RerankResponse objects

    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Validate batch size
        if len(requests) > 10:  # Reasonable batch limit
            raise ValueError("Batch size too large. Maximum 10 requests per batch.")

        # Process each request
        responses = []
        for i, request in enumerate(requests):
            try:
                response = await service.rerank_passages(request)
                responses.append(response)
            except Exception as e:
                raise ValueError(f"Error in batch item {i}: {str(e)}")

        return responses

    except ValueError as e:
        # Input validation errors
        raise HTTPException(status_code=400, detail=f"Invalid batch input: {str(e)}")

    except RuntimeError as e:
        # Backend/model errors
        raise HTTPException(status_code=503, detail=f"Service error: {str(e)}")

    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/info")
async def get_reranking_info(service: RerankingService = Depends(get_reranking_service)):
    """
    Get information about the reranking service and model.

    Returns:
        Dictionary with model information, capabilities, and status
    """
    try:
        info = await service.get_service_info()
        return info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service info: {str(e)}")
