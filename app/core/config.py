"""
Core configuration for the application.

This module defines application settings using Pydantic BaseSettings
for environment variable management and validation.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    PROJECT_NAME: str = "Credit Risk Scoring System"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "RESTful API for credit risk scoring with interactive SHAP explanations"
    )
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "dev-secret-key-change-in-prod"
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/credit_risk.db"

    @property
    def async_database_url(self) -> str:
        """Convert standard postgres urls to asyncpg urls."""
        if self.DATABASE_URL.startswith("postgres://"):
            return self.DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self.DATABASE_URL
    DEBUG: bool = True

    # --- ML Artifact Paths ---
    MODEL_PATH: str = "ml/artefacts/model.pkl"
    PREPROCESSOR_PATH: str = "ml/artefacts/preprocessor.pkl"
    EXPLAINER_PATH: str = "ml/artefacts/explainer.pkl"

    class Config:
        """Pydantic configuration."""
        env_file = ".env"


settings = Settings()


# --- SQLAlchemy Async Setup ---

class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


# Ensure the data directory exists for SQLite
os.makedirs("data", exist_ok=True)

engine = create_async_engine(settings.async_database_url, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session via dependency injection."""
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    """Create all database tables on application startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
