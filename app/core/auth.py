"""Authentication utilities for the API.

Implements API key validation using the X-API-Key header.
The key is compared against the SECRET_KEY setting.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str = Security(api_key_header),
) -> str:
    """Validate the API key from the request header.

    Args:
        api_key: Value of the X-API-Key header.

    Returns:
        The validated API key string.

    Raises:
        HTTPException: 403 if the key is missing or invalid.
    """
    if not api_key or api_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )
    return api_key