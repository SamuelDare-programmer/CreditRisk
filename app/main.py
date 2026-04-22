"""
Main FastAPI application entry point.

This module initializes the FastAPI application, includes routers,
and configures middleware and dependencies.
"""

from fastapi import FastAPI

from app.core.config import settings
from app.credit_scoring.routes import router as credit_scoring_router


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: The configured application instance.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
    )

    # Include routers
    app.include_router(credit_scoring_router, prefix=settings.API_V1_STR)

    return app


app = create_application()
