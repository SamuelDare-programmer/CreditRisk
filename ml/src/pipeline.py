"""End-to-end credit risk model training pipeline.

Orchestrates data loading, cleaning, feature engineering, preprocessing,
model training, evaluation, and artefact persistence.
"""

import argparse
import logging
from typing import Any, Dict, Optional, Type

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.src.config import PipelineConfig
from ml.src.data_cleaner import DataCleaner
from ml.src.data_loader import (
    AfricanCreditLoader,
    DataLoader,
    SuperLenderLoader,
    SyntheticLoader,
)
from ml.src.feature_engineer import FeatureEngineer
from ml.src.model_trainer import ModelTrainer
from ml.src.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

# Mapping from CLI dataset names to loader classes
_LOADER_REGISTRY: Dict[str, Type[DataLoader]] = {
    "superlender": SuperLenderLoader,
    "african_credit": AfricanCreditLoader,
    "synthetic": SyntheticLoader,
}


class CreditRiskPipeline:
    """Orchestrates the full credit risk model training pipeline.

    Pipeline stages executed by ``run()``:
        1. Load raw data via the dataset-specific ``DataLoader``.
        2. Clean data via ``DataCleaner``.
        3. Engineer features via ``FeatureEngineer``.
        4. Train/test split (stratified on the target).
        5. Preprocess + SMOTE on the training set via ``Preprocessor``.
        6. Transform the test set (no SMOTE).
        7. Train all four models via ``ModelTrainer``.
        8. Evaluate all models on the test set.
        9. Save the best model and the preprocessor to ``ml/artefacts/``.
        10. Log a summary with the best model name and AUC-ROC.

    Attributes:
        dataset_name: Human-readable name of the dataset being used.
        config: Pipeline configuration dataclass.
        loader: Dataset-specific loader instance.
        cleaner: Data cleaning component.
        feature_engineer: Feature engineering component.
        preprocessor: Preprocessing + SMOTE component.
        trainer: Model training and evaluation component.
    """

    def __init__(
        self,
        dataset_name: str,
        config: Optional[PipelineConfig] = None,
        imbalance_strategy: str = "smote",
    ) -> None:
        """Initialise the pipeline with the chosen dataset, config, and imbalance strategy.

        Args:
            dataset_name: One of ``'superlender'``, ``'african_credit'``,
                or ``'synthetic'``.
            config: Pipeline configuration; a default is created when
                *None*.
            imbalance_strategy: Class imbalance strategy to use (``'smote'`` or ``'class_weight'``).
                If ``'smote'``, SMOTE resampling is applied to training folds.
                If ``'class_weight'``, classifier weights are adjusted to compensate for imbalance.

        Raises:
            ValueError: If *dataset_name* is not recognised.
        """
        self.dataset_name: str = dataset_name
        self.config: PipelineConfig = config or PipelineConfig()
        self.imbalance_strategy: str = imbalance_strategy.lower()

        # Instantiate pipeline components
        self.loader: DataLoader = self._get_loader(dataset_name, self.config)
        self.cleaner: DataCleaner = DataCleaner(self.config)
        self.feature_engineer: FeatureEngineer = FeatureEngineer(self.config)
        self.preprocessor: Preprocessor = Preprocessor(self.config)
        
        use_cw = (self.imbalance_strategy == "class_weight")
        self.trainer: ModelTrainer = ModelTrainer(self.config, use_class_weights=use_cw)

        logger.info(
            "Pipeline initialised for dataset '%s' with strategy '%s'",
            self.dataset_name,
            self.imbalance_strategy,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full end-to-end pipeline.

        Returns:
            Dictionary containing:
                - ``cv_results``: Cross-validation metrics per model.
                - ``evaluation_results``: Test-set metrics per model.
                - ``best_model_name``: Name of the top performer.
                - ``best_auc_roc``: AUC-ROC of the best model.
        """
        # 1. Load data
        logger.info("Stage 1/9 — Loading data …")
        X, y = self.loader.load()
        logger.info(
            "Loaded %d samples with %d features", X.shape[0], X.shape[1]
        )

        # 2. Clean data
        logger.info("Stage 2/9 — Cleaning data …")
        X, y = self.cleaner.clean(X, y)
        logger.info("Post-cleaning shape: %s", X.shape)

        # 3. Engineer features
        logger.info("Stage 3/9 — Engineering features …")
        X = self.feature_engineer.engineer(X)
        logger.info("Post-engineering shape: %s", X.shape)

        # 4. Train/test split (stratified)
        logger.info("Stage 4/9 — Splitting data …")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        logger.info(
            "Train: %d samples, Test: %d samples",
            X_train.shape[0],
            X_test.shape[0],
        )

        # 5. Preprocess training data + SMOTE
        logger.info("Stage 5/9 — Preprocessing (train) …")
        use_smote = (self.imbalance_strategy == "smote")
        X_train_processed, y_train_resampled = self.preprocessor.fit_transform(
            X_train, y_train, use_smote=use_smote
        )

        # To avoid data leakage in CV when using SMOTE:
        # Get the unresampled transformed training features
        if use_smote:
            logger.info("Extracting unresampled transformed features for CV to prevent leakage")
            X_train_unresampled = self.preprocessor.column_transformer.transform(X_train)
            y_train_unresampled = y_train.to_numpy()
        else:
            X_train_unresampled = None
            y_train_unresampled = None

        # 6. Transform test data (no SMOTE)
        logger.info("Stage 6/9 — Transforming test data …")
        X_test_processed = self.preprocessor.transform(X_test)

        # 7. Train all models
        logger.info("Stage 7/9 — Training models …")
        cv_results = self.trainer.train_all(
            X_train_processed,
            y_train_resampled,
            X_train_unresampled=X_train_unresampled,
            y_train_unresampled=y_train_unresampled,
        )

        # 8. Evaluate all models
        logger.info("Stage 8/9 — Evaluating models …")
        evaluation_results = self.trainer.evaluate_all(
            X_test_processed, y_test
        )

        # 9. Save best model + preprocessor
        logger.info("Stage 9/9 — Saving artefacts …")
        best_name, best_model = self.trainer.get_best_model()
        self.trainer.save_model(best_model, self.config.model_save_path)
        self.preprocessor.save(self.config.preprocessor_save_path)

        # Final summary
        best_auc = evaluation_results[best_name]["auc_roc"]
        logger.info(
            "Pipeline complete — Best model: %s | AUC-ROC: %.4f",
            best_name,
            best_auc,
        )

        return {
            "cv_results": cv_results,
            "evaluation_results": evaluation_results,
            "best_model_name": best_name,
            "best_auc_roc": best_auc,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_loader(name: str, config: PipelineConfig) -> DataLoader:
        """Factory method mapping a dataset name to its loader class.

        Args:
            name: Dataset identifier (e.g. ``'superlender'``).
            config: Pipeline configuration passed to the loader.

        Returns:
            An instantiated DataLoader subclass.

        Raises:
            ValueError: If *name* does not match a registered loader.
        """
        loader_cls = _LOADER_REGISTRY.get(name)
        if loader_cls is None:
            valid = ", ".join(sorted(_LOADER_REGISTRY.keys()))
            raise ValueError(
                f"Unknown dataset '{name}'. Choose from: {valid}"
            )
        return loader_cls(config)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Configure logging for the entire pipeline
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train and evaluate credit risk models.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["superlender", "african_credit", "synthetic"],
        default="superlender",
        help="Dataset to use for training (default: superlender).",
    )
    parser.add_argument(
        "--imbalance",
        type=str,
        choices=["smote", "class_weight", "none"],
        default="smote",
        help="Imbalance strategy to use (default: smote).",
    )

    args = parser.parse_args()

    pipeline_config = PipelineConfig()
    pipeline = CreditRiskPipeline(
        dataset_name=args.dataset,
        config=pipeline_config,
        imbalance_strategy=args.imbalance,
    )
    results = pipeline.run()

    # Print top-level results to stdout for quick inspection
    logger.info(
        "Final results — Best: %s, AUC-ROC: %.4f",
        results["best_model_name"],
        results["best_auc_roc"],
    )
