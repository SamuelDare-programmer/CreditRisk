"""Preprocessor module for the credit risk ML pipeline.

Handles feature type detection, column transformations (scaling, encoding),
and SMOTE-based class rebalancing for training data.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml.src.config import PipelineConfig

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocesses credit risk data via column transformations and SMOTE.

    Responsibilities:
        - Auto-detect feature types (numeric, categorical, binary) from
          the training DataFrame if not explicitly provided in config.
        - Build a sklearn ColumnTransformer that applies imputation, scaling,
          and encoding:
            * Imputation (median/constant) + StandardScaler to numeric features
            * Imputation (constant) + OneHotEncoder to categorical features
            * OrdinalEncoder to binary features
        - Apply SMOTE resampling to the *training* set only.

    Attributes:
        config: Pipeline configuration dataclass.
        column_transformer: Fitted sklearn ColumnTransformer instance.
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
        binary_features: List of binary column names.
        _is_fitted: Whether the transformer has been fitted.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise the Preprocessor.

        Args:
            config: Pipeline configuration dataclass containing feature
                groups and SMOTE parameters.
        """
        self.config: PipelineConfig = config
        self.column_transformer: Optional[ColumnTransformer] = None
        self.numeric_features: List[str] = list(config.numeric_features)
        self.categorical_features: List[str] = list(config.categorical_features)
        self.binary_features: List[str] = list(config.binary_features)
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: pd.Series, use_smote: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the column transformer on training data, then optionally apply SMOTE.

        Steps:
            1. Auto-detect feature types when lists are empty.
            2. Build and fit the ColumnTransformer on *X_train*.
            3. Transform *X_train*.
            4. Optionally apply SMOTE to the transformed training data.
            5. Log class distributions before and after SMOTE.

        Args:
            X_train: Training feature DataFrame.
            y_train: Training target Series.
            use_smote: Whether to apply SMOTE.

        Returns:
            Tuple of (X_train_transformed, y_train_transformed) as NumPy
            arrays.
        """
        # Step 1 — detect feature types if not already specified
        if not (self.numeric_features or self.categorical_features or self.binary_features):
            self._detect_feature_types(X_train)

        # Filter to only columns present in X_train
        self.numeric_features = [
            col for col in self.numeric_features if col in X_train.columns
        ]
        self.categorical_features = [
            col for col in self.categorical_features if col in X_train.columns
        ]
        self.binary_features = [
            col for col in self.binary_features if col in X_train.columns
        ]

        logger.info(
            "Feature groups — numeric: %d, categorical: %d, binary: %d",
            len(self.numeric_features),
            len(self.categorical_features),
            len(self.binary_features),
        )

        # Step 2 — build and fit ColumnTransformer
        self.column_transformer = self._build_column_transformer()
        X_train_transformed: np.ndarray = self.column_transformer.fit_transform(X_train)
        self._is_fitted = True

        logger.info(
            "Transformed training shape: %s",
            X_train_transformed.shape,
        )

        if not use_smote:
            logger.info("SMOTE is disabled. Returning transformed features directly.")
            self._log_class_distribution(y_train, label="Training Target")
            return X_train_transformed, y_train.to_numpy()

        # Step 3 — log pre-SMOTE class distribution
        self._log_class_distribution(y_train, label="Before SMOTE")

        # Step 4 — apply SMOTE
        smote = SMOTE(
            random_state=self.config.random_state,
            k_neighbors=self.config.smote_k_neighbors,
        )
        X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)

        # Step 5 — log post-SMOTE class distribution
        self._log_class_distribution(y_resampled, label="After SMOTE")

        return X_resampled, y_resampled.to_numpy() if isinstance(y_resampled, pd.Series) else y_resampled

    def transform(self, X_test: pd.DataFrame) -> np.ndarray:
        """Transform test data using the already-fitted ColumnTransformer.

        SMOTE is **not** applied to the test set — this preserves the
        original class distribution for unbiased evaluation.

        Args:
            X_test: Test feature DataFrame.

        Returns:
            Transformed test features as a NumPy array.

        Raises:
            RuntimeError: If called before ``fit_transform``.
        """
        if not self._is_fitted or self.column_transformer is None:
            raise RuntimeError(
                "Preprocessor has not been fitted. "
                "Call fit_transform() before transform()."
            )

        X_test_transformed: np.ndarray = self.column_transformer.transform(X_test)
        logger.info("Transformed test shape: %s", X_test_transformed.shape)
        return X_test_transformed

    def get_feature_names(self) -> List[str]:
        """Return feature names produced by the fitted ColumnTransformer.

        Returns:
            List of output feature names after all transformations.

        Raises:
            RuntimeError: If called before ``fit_transform``.
        """
        if not self._is_fitted or self.column_transformer is None:
            raise RuntimeError(
                "Preprocessor has not been fitted. "
                "Call fit_transform() before get_feature_names()."
            )
        return list(self.column_transformer.get_feature_names_out())

    def save(self, path: Optional[str] = None) -> None:
        """Persist the fitted ColumnTransformer to disk via joblib.

        Only the ColumnTransformer is serialised — the SMOTE object is
        not needed at inference time.

        Args:
            path: Destination file path. Falls back to
                ``config.preprocessor_save_path`` when *None*.

        Raises:
            RuntimeError: If called before ``fit_transform``.
        """
        if not self._is_fitted or self.column_transformer is None:
            raise RuntimeError(
                "Preprocessor has not been fitted. "
                "Call fit_transform() before save()."
            )

        save_path: str = path or self.config.preprocessor_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(
            {
                "column_transformer": self.column_transformer,
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
                "binary_features": self.binary_features,
            },
            save_path,
        )
        file_size_kb = os.path.getsize(save_path) / 1024
        logger.info(
            "Preprocessor saved to %s (%.1f KB)", save_path, file_size_kb
        )

    @classmethod
    def load(cls, path: str, config: Optional[PipelineConfig] = None) -> "Preprocessor":
        """Load a previously saved Preprocessor from disk.

        Args:
            path: Path to the joblib-serialised preprocessor artefact.
            config: Optional pipeline config; a default is created when
                *None*.

        Returns:
            A fully initialised Preprocessor with the fitted
            ColumnTransformer restored.
        """
        artefact: Dict = joblib.load(path)
        resolved_config = config or PipelineConfig()
        instance = cls(resolved_config)
        instance.column_transformer = artefact["column_transformer"]
        instance.numeric_features = artefact["numeric_features"]
        instance.categorical_features = artefact["categorical_features"]
        instance.binary_features = artefact["binary_features"]
        instance._is_fitted = True
        logger.info("Preprocessor loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_feature_types(self, X: pd.DataFrame) -> None:
        """Auto-detect numeric, categorical, and binary feature types.

        Classification rules:
            - **Numeric**: columns with dtype ``int64`` or ``float64``.
            - **Binary**: columns with dtype ``object`` or ``bool`` that
              have ≤ 2 unique non-null values.
            - **Categorical**: columns with dtype ``object`` that have
              > 2 unique non-null values.

        Args:
            X: DataFrame whose columns are to be classified.
        """
        self.numeric_features = []
        self.categorical_features = []
        self.binary_features = []

        for col in X.columns:
            if X[col].dtype in ("int64", "float64"):
                self.numeric_features.append(col)
            elif X[col].dtype == "bool" or (
                X[col].dtype == "object" and X[col].nunique() <= 2
            ):
                self.binary_features.append(col)
            elif X[col].dtype == "object" and X[col].nunique() > 2:
                self.categorical_features.append(col)

        logger.info(
            "Auto-detected features — numeric: %s, categorical: %s, binary: %s",
            self.numeric_features,
            self.categorical_features,
            self.binary_features,
        )

    def _build_column_transformer(self) -> ColumnTransformer:
        """Construct the ColumnTransformer from detected feature groups.

        Transformers applied:
            - Numeric: ``SimpleImputer(median)`` → ``StandardScaler``
            - Categorical: ``SimpleImputer(constant)`` → ``OneHotEncoder``
            - Binary: ``OrdinalEncoder``

        Returns:
            An *unfitted* ColumnTransformer ready for ``fit_transform``.
        """
        transformers = []

        if self.numeric_features:
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(
                ("num", numeric_transformer, self.numeric_features)
            )

        if self.categorical_features:
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="Unknown"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            drop="first", sparse_output=False, handle_unknown="ignore"
                        ),
                    ),
                ]
            )
            transformers.append(
                (
                    "cat",
                    categorical_transformer,
                    self.categorical_features,
                )
            )

        if self.binary_features:
            transformers.append(
                (
                    "bin",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    self.binary_features,
                )
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )

    @staticmethod
    def _log_class_distribution(
        y: np.ndarray | pd.Series, label: str = ""
    ) -> None:
        """Log the value-count distribution of the target variable.

        Args:
            y: Target array or Series.
            label: Descriptive prefix for the log message.
        """
        series = pd.Series(y)
        counts = series.value_counts().to_dict()
        total = len(series)
        logger.info(
            "%s — class distribution: %s (total: %d)", label, counts, total
        )
