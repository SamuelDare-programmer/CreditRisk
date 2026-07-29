"""Services for credit scoring feature.

This module contains the business logic for running the machine learning pipeline
(data cleaning, feature engineering, preprocessing, and SHAP explainability) on
incoming REST API loan applications.
"""

import logging
from typing import Any, Dict

import pandas as pd

from app.credit_scoring.schemas import BorrowerProfile, RiskFactor
from ml.src.config import PipelineConfig
from ml.src.data_cleaner import DataCleaner
from ml.src.feature_engineer import FeatureEngineer
from ml.src.preprocessor import Preprocessor
from ml.src.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)


class PredictorService:
    """Service for making credit risk predictions and generating explanations.

    Encapsulates serialised ML model and preprocessor objects, orchestrating
    the transformation of incoming raw API data into prediction probabilities
    and SHAP risk indicators.
    """

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        explainer_path: str,
    ) -> None:
        """Initialize the PredictorService by loading serialized artifacts.

        Args:
            model_path: File path to joblib-serialized model.
            preprocessor_path: File path to joblib-serialized preprocessor.
            explainer_path: File path to joblib-serialized SHAPExplainer.
        """
        logger.info("Initializing PredictorService and loading ML artifacts...")
        
        # Load the PipelineConfig
        self._config = PipelineConfig()
        
        # Load serialized components
        try:
            self._preprocessor = Preprocessor.load(preprocessor_path, self._config)
            self._explainer = SHAPExplainer.load(explainer_path)
            
            # Loader for the best estimator model
            import joblib
            self._model = joblib.load(model_path)
            logger.info("Successfully loaded all ML artifacts into memory.")
        except Exception as e:
            logger.error("Failed to load ML artifacts: %s. PredictorService is degraded.", e)
            self._model = None
            self._preprocessor = None
            self._explainer = None

    def predict_risk(self, profile: BorrowerProfile) -> Dict[str, Any]:
        """Run the applicant profile through the ML pipeline to predict default risk.

        Orchestrations:
            1. Convert Pydantic request model to single-row pandas DataFrame.
            2. Run raw features through DataCleaner (with dummy target Series).
            3. Run cleaned features through FeatureEngineer.
            4. Transform columns using fitted ColumnTransformer.
            5. Predict default probability (risk_score).
            6. Categorise risk bucket and determine underwriting decision.
            7. Query SHAP values to extract top 5 explaining factors.

        Args:
            profile: Borrower profile from REST request.

        Returns:
            Dictionary matching the RiskScoreResponse schema:
                - risk_score: float (0.0 to 1.0)
                - risk_category: str ("low_risk", "medium_risk", "high_risk")
                - recommendation: str ("approve", "decline")
                - risk_factors: list[RiskFactor]

        Raises:
            RuntimeError: If service is degraded (failed to load artifacts).
        """
        if self._model is None or self._preprocessor is None or self._explainer is None:
            raise RuntimeError("PredictorService is currently unavailable due to artifact loading failure.")

        logger.info("Processing loan scoring request for applicant...")

        # 1. Convert profile to DataFrame
        raw_dict = profile.dict()
        df = pd.DataFrame([raw_dict])

        # 2. Clean features (using a dummy target series of length 1)
        cleaner = DataCleaner(self._config)
        dummy_y = pd.Series([0])
        cleaned_df, _ = cleaner.clean(df, dummy_y)

        # 3. Engineer features
        engineer = FeatureEngineer(self._config)
        featured_df = engineer.engineer(cleaned_df)

        # 4. Transform features using fitted preprocessor
        processed_features = self._preprocessor.transform(featured_df)

        # 5. Predict default probability
        risk_score = float(self._model.predict_proba(processed_features)[:, 1][0])
        logger.info("Scored risk probability: %.4f", risk_score)

        # 6. Categorise risk and recommendation
        if risk_score < 0.30:
            category = "low_risk"
            recommendation = "approve"
        elif risk_score < 0.50:
            category = "medium_risk"
            recommendation = "approve"
        else:
            category = "high_risk"
            recommendation = "decline"

        # 7. Generate SHAP explanations on featured DataFrame (untransformed raw columns)
        logger.info("Generating SHAP explanations...")
        shap_factors = self._explainer.explain(featured_df, top_n=5)
        
        # 8. Generate SHAP waterfall plot
        try:
            logger.info("Generating SHAP waterfall plot...")
            waterfall_b64 = self._explainer.generate_waterfall_plot(featured_df)
        except Exception as e:
            logger.warning("Failed to generate waterfall plot: %s", e)
            waterfall_b64 = None
        
        # Convert SHAP results into Pydantic-compatible RiskFactor dict list
        risk_factors = [
            RiskFactor(
                feature=factor["feature"],
                impact=factor["impact"],
                direction=factor["direction"]
            )
            for factor in shap_factors
        ]

        return {
            "risk_score": risk_score,
            "risk_category": category,
            "recommendation": recommendation,
            "risk_factors": risk_factors,
            "shap_waterfall_plot": waterfall_b64
        }