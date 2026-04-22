"""
Custom exception handlers and error responses.

This module defines application-specific exceptions and error handling.
"""

from fastapi import HTTPException


class CreditRiskException(HTTPException):
    """Base exception for credit risk application errors."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)


class ModelNotFoundError(CreditRiskException):
    """Raised when a required ML model is not found."""

    def __init__(self) -> None:
        super().__init__(status_code=500, detail="ML model not available")


class InvalidInputError(CreditRiskException):
    """Raised when input data is invalid."""

    def __init__(self, detail: str = "Invalid input data") -> None:
        super().__init__(status_code=400, detail=detail)