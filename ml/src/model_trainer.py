"""Model training and evaluation module for the credit risk ML pipeline.

Trains four classifiers (Logistic Regression, Random Forest, XGBoost,
LightGBM), evaluates them with stratified cross-validation and a held-out
test set, and selects the best performer by AUC-ROC.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from ml.src.config import PipelineConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains, cross-validates, evaluates, and persists credit risk models.

    Four classifiers are initialised with reproducible hyper-parameters and
    benchmarked on the same stratified splits.  After evaluation the best
    model (by test AUC-ROC) can be retrieved and saved as a joblib artefact.

    Attributes:
        config: Pipeline configuration dataclass.
        models: Dictionary mapping model names to sklearn-compatible
            estimator instances.
        cv_results: Per-model cross-validation results (populated after
            ``train_all``).
        evaluation_results: Per-model test-set metrics (populated after
            ``evaluate_all``).
    """

    def __init__(self, config: PipelineConfig, use_class_weights: bool = False) -> None:
        """Initialise ModelTrainer with four pre-configured classifiers.

        Args:
            config: Pipeline configuration providing ``random_state``
                and ``cv_folds``.
            use_class_weights: Whether to use class weights for models
                to handle class imbalance (alternative to SMOTE).
        """
        self.config: PipelineConfig = config
        self.use_class_weights: bool = use_class_weights
        self.models: Dict[str, Any] = self._build_models()
        self.cv_results: Dict[str, Dict[str, Any]] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_unresampled: Optional[np.ndarray] = None,
        y_train_unresampled: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Fit every model on the training set and run cross-validation.

        For each model the method:
            1. Sets dynamic class/imbalance weights if enabled.
            2. Calls ``model.fit(X_train, y_train)``.
            3. Runs ``_cross_validate`` with stratified k-fold.
               To prevent leakage, SMOTE is applied inside each CV fold
               if unresampled training data is provided.
            4. Logs the mean CV AUC-ROC.

        Args:
            X_train: Transformed training features (post-SMOTE).
            y_train: Resampled training labels.
            X_train_unresampled: Transformed training features before SMOTE (optional, for CV).
            y_train_unresampled: Training labels before SMOTE (optional, for CV).

        Returns:
            Dictionary ``{model_name: cv_result_dict}`` where each
            result dict contains *mean*, *std*, and *per_fold_scores*.
        """
        if self.use_class_weights:
            # Estimate negative-to-positive ratio for XGBoost scale_pos_weight
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            ratio = neg_count / max(pos_count, 1)
            if "XGBoost" in self.models:
                self.models["XGBoost"].set_params(scale_pos_weight=ratio)
                logger.info("Set XGBoost scale_pos_weight = %.4f", ratio)

        for name, model in self.models.items():
            logger.info("Training %s …", name)
            model.fit(X_train, y_train)

            # If unresampled data is passed, run CV on it with SMOTE inside each fold
            if X_train_unresampled is not None and y_train_unresampled is not None:
                logger.info("Running CV on unresampled data with SMOTE inside folds for %s", name)
                cv_result = self._cross_validate(
                    model, X_train_unresampled, y_train_unresampled, use_smote=True
                )
            else:
                logger.info("Running standard CV on training data for %s", name)
                cv_result = self._cross_validate(
                    model, X_train, y_train, use_smote=False
                )
                
            self.cv_results[name] = cv_result

            logger.info(
                "%s — CV AUC-ROC: %.4f (± %.4f)",
                name,
                cv_result["mean"],
                cv_result["std"],
            )

        return self.cv_results

    def evaluate_all(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate every trained model on the held-out test set.

        Metrics computed per model:
            - AUC-ROC
            - F1 score
            - Precision
            - Recall
            - Log-loss
            - Confusion matrix

        Args:
            X_test: Transformed test features.
            y_test: True test labels.

        Returns:
            Dictionary ``{model_name: metrics_dict}``.
        """
        for name, model in self.models.items():
            metrics = self._compute_metrics(model, X_test, y_test)
            self.evaluation_results[name] = metrics

        # Log a comparison table
        self._log_comparison_table()

        return self.evaluation_results

    def get_best_model(self) -> Tuple[str, Any]:
        """Select the model with the highest test AUC-ROC.

        Returns:
            Tuple of ``(model_name, model_instance)``.

        Raises:
            RuntimeError: If ``evaluate_all`` has not been called.
        """
        if not self.evaluation_results:
            raise RuntimeError(
                "No evaluation results available. "
                "Call evaluate_all() before get_best_model()."
            )

        best_name: str = max(
            self.evaluation_results,
            key=lambda n: self.evaluation_results[n]["auc_roc"],
        )
        best_model = self.models[best_name]

        logger.info(
            "Best model: %s (AUC-ROC: %.4f)",
            best_name,
            self.evaluation_results[best_name]["auc_roc"],
        )
        return best_name, best_model

    def save_model(self, model: Any, path: Optional[str] = None) -> None:
        """Serialise a trained model to disk with joblib.

        Args:
            model: The sklearn-compatible estimator to persist.
            path: Destination file path.  Falls back to
                ``config.model_save_path`` when *None*.
        """
        save_path: str = path or self.config.model_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)

        file_size_kb = os.path.getsize(save_path) / 1024
        logger.info(
            "Model saved to %s (%.1f KB)", save_path, file_size_kb
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_models(self) -> Dict[str, Any]:
        """Construct the four classifier instances.

        Returns:
            Ordered dictionary of ``{name: estimator}``.
        """
        rs: int = self.config.random_state
        use_cw = self.use_class_weights

        return {
            "LogisticRegression": LogisticRegression(
                C=1.0,
                penalty="l2",
                max_iter=1000,
                solver="lbfgs",
                random_state=rs,
                class_weight="balanced" if use_cw else None,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                random_state=rs,
                class_weight="balanced" if use_cw else None,
            ),
            "XGBoost": XGBClassifier(
                learning_rate=0.1,
                max_depth=6,
                n_estimators=200,
                random_state=rs,
                use_label_encoder=False,
                eval_metric="logloss",
                # scale_pos_weight is set dynamically during training if use_cw is True
            ),
            "LightGBM": LGBMClassifier(
                learning_rate=0.1,
                num_leaves=31,
                n_estimators=200,
                random_state=rs,
                verbose=-1,
                is_unbalance=use_cw,
            ),
        }

    def _cross_validate(
        self, model: Any, X: np.ndarray, y: np.ndarray, use_smote: bool = False
    ) -> Dict[str, Any]:
        """Run stratified k-fold cross-validation for a single model.

        To prevent target leakage, SMOTE is applied only inside each training split
        (not the validation split) if ``use_smote`` is True.

        Args:
            model: An unfitted sklearn-compatible estimator.
            X: Feature array.
            y: Target array.
            use_smote: Whether to apply SMOTE within each CV fold.

        Returns:
            Dictionary with keys *mean*, *std*, and *per_fold_scores*
            representing AUC-ROC across folds.
        """
        skf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        if use_smote:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            
            # Wrap the model in an imblearn Pipeline so SMOTE is applied per-fold
            cv_model = ImbPipeline([
                ("smote", SMOTE(
                    random_state=self.config.random_state,
                    k_neighbors=self.config.smote_k_neighbors
                )),
                ("model", model)
            ])
        else:
            cv_model = model

        scores: np.ndarray = cross_val_score(
            cv_model, X, y, cv=skf, scoring="roc_auc"
        )

        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "per_fold_scores": scores.tolist(),
        }

    @staticmethod
    def _compute_metrics(
        model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Compute classification metrics on a test set.

        Metrics:
            - ``auc_roc``: Area Under the ROC Curve
            - ``f1``: F1 score (binary)
            - ``precision``: Precision (binary)
            - ``recall``: Recall (binary)
            - ``log_loss``: Logarithmic loss
            - ``confusion_matrix``: 2×2 confusion matrix as a list

        Args:
            model: A fitted estimator exposing ``predict`` and
                ``predict_proba``.
            X_test: Test features.
            y_test: True test labels.

        Returns:
            Dictionary of metric names to their values.
        """
        y_pred: np.ndarray = model.predict(X_test)
        y_proba: np.ndarray = model.predict_proba(X_test)[:, 1]

        return {
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "log_loss": float(log_loss(y_test, y_proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    def _log_comparison_table(self) -> None:
        """Log a formatted comparison table of all evaluation results."""
        header = f"{'Model':<22} {'AUC-ROC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'LogLoss':>8}"
        separator = "-" * len(header)

        logger.info("\n%s\n%s", header, separator)

        for name, metrics in self.evaluation_results.items():
            row = (
                f"{name:<22} "
                f"{metrics['auc_roc']:>8.4f} "
                f"{metrics['f1']:>8.4f} "
                f"{metrics['precision']:>8.4f} "
                f"{metrics['recall']:>8.4f} "
                f"{metrics['log_loss']:>8.4f}"
            )
            logger.info(row)

        logger.info(separator)
