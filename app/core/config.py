"""
Core configuration for the application.

This module defines application settings using Pydantic BaseSettings
for environment variable management and validation.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes:
        PROJECT_NAME: Name of the project.
        VERSION: Version of the application.
        DESCRIPTION: Description of the application.
        API_V1_STR: API version string.
        SECRET_KEY: Secret key for JWT or other security.
        DATABASE_URL: URL for the database connection.
        DEBUG: Debug mode flag.
        KAGGLE_USERNAME: Kaggle API username.
        KAGGLE_KEY: Kaggle API key.
    """

    PROJECT_NAME: str = "Credit Risk Scoring System"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "API for credit risk scoring using machine learning"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-here"  # TODO: Move to .env
    DATABASE_URL: Optional[str] = None
    DEBUG: bool = True

    # Kaggle Integration
    KAGGLE_USERNAME: Optional[str] = None
    KAGGLE_KEY: Optional[str] = None

    class Config:
        """Pydantic configuration."""

        env_file = ".env"


settings = Settings()
