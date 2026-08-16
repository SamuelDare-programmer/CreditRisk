"""SHAP explainability module for the credit risk ML pipeline.

Provides the SHAPExplainer class to generate per-applicant risk factors
for model transparency and regulatory compliance.
"""

import logging
import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap

from ml.src.config import PipelineConfig
from ml.src.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Computes and formats feature-level SHAP explanations for credit decisions.

    Designed to support both linear models (via LinearExplainer) and tree-based
    models (via TreeExplainer) using the unified shap.Explainer interface.
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Preprocessor,
        background_data: np.ndarray,
    ) -> None:
        """Initialise the SHAPExplainer.

        Args:
            model: Trained classifier model.
            preprocessor: Fitted Preprocessor instance.
            background_data: Representative sample of transformed training features
                used as a reference distribution.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = preprocessor.get_feature_names()

        logger.info("Initializing SHAP Explainer...")
        
        # Check if the model is linear or tree-based to choose the right masker/explainer
        model_name = type(model).__name__
        if "LogisticRegression" in model_name:
            # Linear models require a background reference to establish base expectation
            logger.info("Detected Linear Model. Using shap.Explainer with background data.")
            self.explainer = shap.Explainer(
                model.predict_proba, 
                background_data,
                feature_names=self.feature_names
            )
        else:
            # Tree-based models (LightGBM, XGBoost, RandomForest) can run directly
            logger.info("Detected Tree Ensemble. Using shap.Explainer.")
            self.explainer = shap.Explainer(
                model, 
                feature_names=self.feature_names
            )

    def explain(self, input_df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """Generate ranked SHAP explanations for a single applicant.

        Args:
            input_df: Single-row DataFrame containing raw applicant features.
            top_n: Number of top features to return (default: 5).

        Returns:
            List of dictionaries containing 'feature', 'impact' (SHAP value),
            and 'direction' ('increased_risk' or 'decreased_risk').
        """
        if len(input_df) != 1:
            raise ValueError("SHAP Explainer expects exactly one row for real-time explanations.")

        # Transform inputs using the preprocessor
        X_transformed = self.preprocessor.transform(input_df)

        # Compute SHAP values
        shap_values = self.explainer(X_transformed)
        
        # Extract values for the first (and only) row
        # shap_values can be an Explanation object or array depending on the explainer type
        if hasattr(shap_values, "values"):
            values = shap_values.values[0]
        else:
            values = shap_values[0]

        # Handle multi-class / probability output indexing
        # For binary classification, ensure we explain the probability of class 1 (default)
        if len(values.shape) > 1 and values.shape[-1] == 2:
            values = values[:, 1]  # Select class 1 (Default/Bad)
        elif len(values.shape) > 1:
            values = values[0]

        # Pair features with their SHAP impact
        explanations = []
        for feat_name, val in zip(self.feature_names, values):
            # Map raw feature names (like 'num__loanamount') to user-friendly names
            clean_name = feat_name.split("__")[-1]
            
            # Map direction: positive values increase default risk, negative values decrease it
            direction = "increased_risk" if val > 0 else "decreased_risk"
            
            explanations.append({
                "feature": clean_name,
                "impact": float(abs(val)),
                "direction": direction
            })

        # Sort by absolute impact magnitude descending
        explanations.sort(key=lambda x: x["impact"], reverse=True)

        return explanations[:top_n]

    def generate_waterfall_plot(self, input_df: pd.DataFrame) -> str:
        """Generate a SHAP waterfall plot as a base64-encoded PNG string.

        Creates a visual waterfall plot showing how each feature contributes
        to pushing the model output from the base value (expected value)
        to the actual prediction for a single applicant.

        Args:
            input_df: Single-row DataFrame containing raw applicant features.

        Returns:
            Base64-encoded PNG image string of the SHAP waterfall plot.
        """
        if len(input_df) != 1:
            raise ValueError("Waterfall plot expects exactly one row.")

        import io
        import base64
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Transform inputs using the preprocessor
        X_transformed = self.preprocessor.transform(input_df)

        # Compute SHAP values
        shap_values = self.explainer(X_transformed)

        # Extract the Explanation object for the first (and only) row
        # Handle binary classification: select class 1 (default)
        if hasattr(shap_values, "values") and len(shap_values.values.shape) > 2:
            explanation = shap_values[0, :, 1]
        else:
            explanation = shap_values[0]

        # Generate waterfall plot
        fig = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, max_display=10, show=False)
        plt.title("SHAP Waterfall — Feature Contributions to Default Risk", fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode("utf-8")
        logger.info("Generated SHAP waterfall plot (%d bytes encoded)", len(encoded))
        return encoded

    def save(self, path: str) -> None:
        """Serialise the SHAPExplainer instance to disk.

        Args:
            path: Destination file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info("SHAPExplainer saved successfully to %s", path)

    @classmethod
    def load(cls, path: str) -> "SHAPExplainer":
        """Load a serialised SHAPExplainer instance from disk.

        Args:
            path: Path to the joblib-serialised explainer.

        Returns:
            A restored SHAPExplainer instance.
        """
        instance = joblib.load(path)
        logger.info("SHAPExplainer loaded successfully from %s", path)
        return instance
