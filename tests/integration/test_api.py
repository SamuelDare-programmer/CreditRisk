"""Integration tests for the credit risk scoring FastAPI backend.

Tests GET /health and POST /score endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test health check returns status 200 and healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_score_endpoint() -> None:
    """Test credit scoring POST endpoint accepts valid profiles and returns risk scores."""
    # Define a complete borrower profile matching Pydantic schema
    profile_data = {
        "birthdate": "1990-01-01",
        "bank_account_type": "savings",
        "bank_name_clients": "gt bank",
        "longitude_gps": 3.4,
        "latitude_gps": 6.5,
        "employment_status_clients": "permanent",
        "loanamount": 10000.0,
        "totaldue": 11500.0,
        "termdays": 30,
        "approveddate": "2023-01-01",
        "creationdate": "2023-01-01",
        "prev_loan_count": 2,
        "prev_total_borrowed": 20000.0,
        "prev_avg_loan_amount": 10000.0,
        "prev_max_loan_amount": 10000.0,
        "prev_avg_totaldue": 11500.0,
        "prev_avg_term_days": 30.0,
        "prev_late_repayment_count": 0,
        "prev_late_repayment_ratio": 0.0,
        "prev_repayment_speed_avg": 0.0,
    }

    response = client.post("/api/v1/score", json=profile_data)
    
    assert response.status_code == 200
    json_data = response.json()
    
    # Assert expected fields exist
    assert "risk_score" in json_data
    assert "risk_category" in json_data
    assert "recommendation" in json_data
    assert "risk_factors" in json_data
    
    # Assert values are within valid types/bounds
    assert isinstance(json_data["risk_score"], float)
    assert 0.0 <= json_data["risk_score"] <= 1.0
    assert json_data["risk_category"] in ["low_risk", "medium_risk", "high_risk"]
    assert json_data["recommendation"] in ["approve", "decline"]
    
    # Assert risk factors list structure
    assert isinstance(json_data["risk_factors"], list)
    assert len(json_data["risk_factors"]) <= 5
    
    if len(json_data["risk_factors"]) > 0:
        factor = json_data["risk_factors"][0]
        assert "feature" in factor
        assert "impact" in factor
        assert "direction" in factor
        assert factor["direction"] in ["increased_risk", "decreased_risk"]
