"""
Routes for credit scoring feature.

This module defines the FastAPI routes for credit risk scoring operations.
"""

from fastapi import APIRouter

from app.credit_scoring.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse: Status of the service.
    """
    return HealthResponse(status="healthy")
