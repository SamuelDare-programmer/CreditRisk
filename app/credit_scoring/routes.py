"""Routes for credit scoring feature.

Defines HTTP routes for health checks and borrower loan risk scoring.
All scoring requests are logged to SQLite for audit compliance.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import verify_api_key
from app.core.config import get_db, settings
from app.credit_scoring.schemas import BorrowerProfile, HealthResponse, RiskScoreResponse
from app.credit_scoring.services import PredictorService
from app.models.sqlalchemy.scoring_log import ScoringLog

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize the predictor service singleton on module import
predictor_service = PredictorService(
    model_path=settings.MODEL_PATH,
    preprocessor_path=settings.PREPROCESSOR_PATH,
    explainer_path=settings.EXPLAINER_PATH,
)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint to verify service readiness.

    Returns:
        HealthResponse: Status of the service ("healthy" or "degraded").
    """
    if predictor_service._model is None:
        return HealthResponse(status="degraded")
    return HealthResponse(status="healthy")


@router.post("/score", response_model=RiskScoreResponse)
async def calculate_credit_score(
    profile: BorrowerProfile,
    api_key: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> RiskScoreResponse:
    """Calculate credit risk score and generate explanations for a borrower profile.

    Orchestrates ingestion, cleaning, feature engineering, and model inference,
    returning default probability, risk category, recommendation, SHAP factors,
    and a SHAP waterfall plot. The request and response are logged to SQLite.

    Args:
        profile: Pydantic request model containing demographics, loan info, and history.
        api_key: Validated API key from the X-API-Key header.
        db: Async SQLAlchemy session for audit logging.

    Returns:
        RiskScoreResponse: Predicted risk assessment and explaining factors.
    """
    try:
        result = predictor_service.predict_risk(profile)
        response = RiskScoreResponse(**result)

        # --- Audit logging to SQLite ---
        try:
            log_entry = ScoringLog(
                application_data=profile.model_dump_json(),
                risk_score=response.risk_score,
                default_probability=response.risk_score,
                risk_category=response.risk_category,
                recommendation=response.recommendation,
                shap_values=json.dumps(
                    [rf.model_dump() for rf in response.risk_factors]
                ),
                top_factors=json.dumps(
                    [rf.model_dump() for rf in response.risk_factors[:5]]
                ),
            )
            db.add(log_entry)
            await db.commit()
            logger.info("Scoring request logged to audit database.")
        except Exception as db_err:
            logger.error("Failed to log scoring request to database: %s", db_err)
            # Don't fail the scoring response if logging fails

        return response

    except RuntimeError as e:
        logger.error("PredictorService runtime error: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Scoring service is currently unavailable. Please check artifact health.",
        )
    except Exception as e:
        logger.error("Failed to process credit scoring request: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing request: {str(e)}",
        )
