"""SQLAlchemy ORM model for the scoring audit log.

Records every scoring request and its outcome to a SQLite database
for audit trail compliance and operational monitoring.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.core.config import Base


class ScoringLog(Base):
    """Audit log entry for a single credit scoring request.

    Attributes:
        id: Auto-incrementing primary key.
        timestamp: UTC timestamp of the scoring request.
        application_data: JSON string of the full borrower profile input.
        risk_score: Predicted default probability (0.0–1.0).
        default_probability: Alias for risk_score stored for clarity.
        risk_category: Categorical risk bucket (low_risk, medium_risk, high_risk).
        recommendation: Underwriting decision (approve or decline).
        shap_values: JSON string of all SHAP risk factors.
        top_factors: JSON string of the top contributing factors.
    """

    __tablename__ = "scoring_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    application_data = Column(Text, nullable=False)
    risk_score = Column(Float, nullable=False)
    default_probability = Column(Float, nullable=False)
    risk_category = Column(String(20), nullable=False)
    recommendation = Column(String(10), nullable=False)
    shap_values = Column(Text, nullable=False)
    top_factors = Column(Text, nullable=False)
