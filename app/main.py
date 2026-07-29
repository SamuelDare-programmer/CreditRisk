"""Main FastAPI application entry point.

This module initializes the FastAPI application, includes routers,
configures middleware, CORS, and database lifecycle management.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import init_db, settings
from app.core.middleware import LoggingMiddleware
from app.credit_scoring.routes import router as credit_scoring_router

# Import models so SQLAlchemy registers them before create_all
import app.models.sqlalchemy  # noqa: F401

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Initialises the SQLite database tables on startup.
    """
    logger.info("Starting up — initialising database…")
    await init_db()
    logger.info("Database initialised successfully.")
    yield
    logger.info("Shutting down…")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: The configured application instance.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        lifespan=lifespan,
    )

    # --- Middleware ---
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Streamlit frontend
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers ---
    app.include_router(credit_scoring_router, prefix=settings.API_V1_STR)

    return app


app = create_application()
