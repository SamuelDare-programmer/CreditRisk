"""DataCleaner class for the credit risk ML pipeline.

Handles missing value imputation, outlier treatment via winsorisation,
identifier/sparse column dropping, and data type standardisation.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from ml.src.config import PipelineConfig

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans raw features and target arrays for ML modelling.

    Responsibilities:
        - Drop identifier columns and highly sparse variables.
        - Convert date columns to standard datetimes.
        - Treat outliers in continuous columns via IQR-based winsorisation.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise with pipeline config.

        Args:
            config: Pipeline configuration settings.
        """
        self.config: PipelineConfig = config

    def clean(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Orchestrate the full cleaning pipeline.

        Args:
            X: Input features DataFrame.
            y: Input target Series.

        Returns:
            Tuple of cleaned (X, y).
        """
        initial_shape = X.shape
        logger.info("Cleaning features: starting shape %s", initial_shape)

        # Work on a copy to avoid SettingWithCopy warnings
        X_clean = X.copy()

        # Step 1: Drop sparse columns
        X_clean = self._drop_sparse_columns(X_clean)

        # Step 2: Drop identifiers
        X_clean = self._drop_identifiers(X_clean)

        # Step 3: Handle missing values
        X_clean = self._handle_missing_values(X_clean)

        # Step 4: Parse date columns safely
        X_clean = self._parse_dates(X_clean)

        # Step 5: Treat outliers for continuous numerical features
        X_clean = self._treat_outliers(X_clean)

        # Standardise string values in categorical columns
        for col in X_clean.select_dtypes(include=["object"]).columns:
            X_clean[col] = X_clean[col].astype(str).str.strip().str.lower()

        logger.info(
            "Cleaning complete: shape changed from %s to %s",
            initial_shape,
            X_clean.shape,
        )
        return X_clean, y

    def _drop_sparse_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns listed as sparse in config.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with sparse columns removed.
        """
        cols_to_drop = [col for col in self.config.columns_to_drop if col in df.columns]
        if cols_to_drop:
            logger.info("Dropping sparse columns: %s", cols_to_drop)
            df = df.drop(columns=cols_to_drop)
        return df

    def _drop_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop ID and metadata columns.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with identifiers removed.
        """
        cols_to_drop = [col for col in self.config.identifiers if col in df.columns]
        if cols_to_drop:
            logger.info("Dropping identifier columns: %s", cols_to_drop)
            df = df.drop(columns=cols_to_drop)
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to parse columns containing date information to datetime objects.

        Identifies columns based on name patterns ('date', 'birth', 'creation', 'approved').

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with parsed datetime columns.
        """
        df = df.copy()
        date_keywords = ["date", "birth", "creation", "approved"]

        for col in df.columns:
            if any(kw in col.lower() for kw in date_keywords):
                # Only convert if it's not already a datetime
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    logger.debug("Parsing date column: %s", col)
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

    def _treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply IQR-based winsorisation to cap extreme continuous numerical outliers.

        Only caps columns with > 10 unique values to avoid affecting encoded features
        or binary indicators.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with capped numerical values.
        """
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            # Skip binary/categorical flags represented as numbers
            if df[col].nunique() <= 10:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Cap outliers
                original_min = df[col].min()
                original_max = df[col].max()
                
                df[col] = np.clip(df[col], lower_bound, upper_bound)

                capped_lower = (df[col] == lower_bound).sum()
                capped_upper = (df[col] == upper_bound).sum()

                if capped_lower > 0 or capped_upper > 0:
                    logger.info(
                        "Winsorised '%s' — lower cap: %.2f (%d rows), upper cap: %.2f (%d rows)",
                        col,
                        lower_bound,
                        capped_lower,
                        upper_bound,
                        capped_upper,
                    )
            
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values based on column type.

        Numeric columns are imputed with their median.
        Categorical columns are imputed with the category 'Unknown'.

        Args:
            df: DataFrame to process.

        Returns:
            Imputed DataFrame.
        """
        df = df.copy()

        # Handle numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                median_val = df[col].median()
                # If median is NaN (all values are NaN), fill with 0
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)
                logger.debug("Imputed %d missing values in numeric '%s' with median %s", null_count, col, median_val)

        # Handle categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                df[col] = df[col].fillna("Unknown")
                logger.debug("Imputed %d missing values in categorical '%s' with 'Unknown'", null_count, col)

        return df
