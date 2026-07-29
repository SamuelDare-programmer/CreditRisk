"""Pydantic schemas for credit scoring feature.

This module defines request and response models for the credit scoring API,
including input validation constraints and Nigerian alternative data fields.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str


class BorrowerProfile(BaseModel):
    """Input model for borrower profile data.

    Supports both SuperLender dataset fields and Nigerian alternative
    data signals. Nigerian fields are optional to maintain backward
    compatibility across different dataset configurations.
    """

    # --- Demographics ---
    birthdate: str = Field(..., description="Date of birth (YYYY-MM-DD)")
    bank_account_type: str = Field(..., description="Account type, e.g. savings, current")
    bank_name_clients: str = Field(..., description="Bank name, e.g. gt bank")
    longitude_gps: float = Field(..., ge=-180, le=180, description="GPS longitude")
    latitude_gps: float = Field(..., ge=-90, le=90, description="GPS latitude")
    employment_status_clients: str = Field(
        ..., description="Employment status, e.g. permanent, self-employed"
    )

    # --- Current Loan Info ---
    loanamount: float = Field(..., gt=0, description="Loan amount requested")
    totaldue: float = Field(..., gt=0, description="Total amount due including fees")
    termdays: int = Field(..., gt=0, description="Loan term in days")

    # --- Request timestamps ---
    approveddate: str = Field(..., description="Approval date (YYYY-MM-DD)")
    creationdate: str = Field(..., description="Application creation date (YYYY-MM-DD)")

    # --- Historical Credit Behaviour (defaults for new customers) ---
    prev_loan_count: int = Field(0, ge=0)
    prev_total_borrowed: float = Field(0.0, ge=0)
    prev_avg_loan_amount: float = Field(0.0, ge=0)
    prev_max_loan_amount: float = Field(0.0, ge=0)
    prev_avg_totaldue: float = Field(0.0, ge=0)
    prev_avg_term_days: float = Field(0.0, ge=0)
    prev_late_repayment_count: int = Field(0, ge=0)
    prev_late_repayment_ratio: float = Field(0.0, ge=0, le=1)
    prev_repayment_speed_avg: float = Field(0.0)

    # --- Nigerian Alternative Data (optional) ---
    has_bvn: Optional[bool] = Field(None, description="BVN verification status")
    ussd_bank_usage: Optional[int] = Field(
        None, ge=0, description="USSD bank transactions in last 30 days"
    )
    monthly_airtime_spend: Optional[float] = Field(
        None, ge=0, description="Monthly airtime spend (NGN)"
    )
    active_betting_account: Optional[bool] = Field(
        None, description="Active betting account flag"
    )
    telco_provider: Optional[str] = Field(
        None, description="Telco provider (MTN, Airtel, Glo, 9mobile)"
    )
    annual_income: Optional[float] = Field(
        None, gt=0, description="Annual income (NGN)"
    )


class RiskFactor(BaseModel):
    """Explains a single feature's impact on credit risk."""

    feature: str
    impact: float
    direction: str  # "increased_risk" or "decreased_risk"


class RiskScoreResponse(BaseModel):
    """Response model for risk score prediction."""

    risk_score: float = Field(..., ge=0, le=1, description="Default probability")
    risk_category: str = Field(..., description="low_risk, medium_risk, or high_risk")
    recommendation: str = Field(..., description="approve or decline")
    risk_factors: List[RiskFactor]
    shap_waterfall_plot: Optional[str] = Field(
        None, description="Base64-encoded PNG of the SHAP waterfall plot"
    )