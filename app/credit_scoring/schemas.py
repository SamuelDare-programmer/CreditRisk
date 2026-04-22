"""
Pydantic schemas for credit scoring feature.

This module defines request and response models for the credit scoring API.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str


class BorrowerProfile(BaseModel):
    """Input model for borrower profile data."""

    # TODO: Define fields based on dataset
    age: int
    income: float
    credit_score: int


class RiskScoreResponse(BaseModel):
    """Response model for risk score prediction."""

    risk_score: float
    risk_category: str
    confidence: float