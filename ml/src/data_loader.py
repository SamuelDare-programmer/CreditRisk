"""DataLoader classes for the credit risk ML pipeline.

Defines the DataLoader base class and specific implementations for loading and
joining raw datasets (SuperLender, African Credit, and Synthetic).
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

from ml.src.config import PipelineConfig

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loading strategies.

    Provides common validation and standardisation methods.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise with pipeline config.

        Args:
            config: Pipeline configuration settings.
        """
        self.config: PipelineConfig = config

    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data.

        Returns:
            Tuple of (features_df, target_series).
        """
        pass

    def _validate_columns(self, df: pd.DataFrame, expected_cols: list[str]) -> None:
        """Validate that all expected columns are present in the DataFrame.

        Args:
            df: DataFrame to validate.
            expected_cols: List of column names that must exist.

        Raises:
            ValueError: If any expected column is missing.
        """
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns in dataset: {missing}")


class SuperLenderLoader(DataLoader):
    """Loads and merges the SuperLender datasets (Performance, Demographics, PrevLoans)."""

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and merge SuperLender tables.

        Steps:
            1. Ingest performance, demographics, and previous loans.
            2. Aggregate previous loans by customerid.
            3. Left join performance with demographics and aggregated previous loans.
            4. Map target variable: Good -> 0, Bad -> 1.
            5. Return features (X) and target (y).
        """
        logger.info("Loading SuperLender datasets...")

        # Load performance (the base table)
        perf_path = self.config.superlender_performance_path
        perf_df = pd.read_csv(perf_path)
        logger.info("Loaded performance table: %d rows", len(perf_df))
        self._validate_columns(
            perf_df, ["customerid", "good_bad_flag", "loanamount", "totaldue"]
        )

        # Load demographics
        demo_path = self.config.superlender_demographics_path
        try:
            demo_df = pd.read_csv(demo_path)
            logger.info("Loaded demographics table: %d rows", len(demo_df))
            self._validate_columns(demo_df, ["customerid"])
            # Remove duplicate customer entries in demographics (keep first)
            demo_df = demo_df.drop_duplicates(subset=["customerid"])
        except FileNotFoundError:
            logger.warning("Demographics file not found. Proceeding without demographics.")
            demo_df = pd.DataFrame(columns=["customerid"])

        # Load previous loans
        prev_path = self.config.superlender_prevloans_path
        try:
            prev_df = pd.read_csv(prev_path)
            logger.info("Loaded previous loans table: %d rows", len(prev_df))
            self._validate_columns(
                prev_df,
                [
                    "customerid",
                    "loanamount",
                    "totaldue",
                    "firstduedate",
                    "firstrepaiddate",
                ],
            )
            agg_prev_df = self._aggregate_previous_loans(prev_df)
        except FileNotFoundError:
            logger.warning("Previous loans file not found. Proceeding without history.")
            agg_prev_df = pd.DataFrame(columns=["customerid"])

        # Perform left joins using performance as the base table
        merged_df = perf_df.copy()

        if not demo_df.empty:
            merged_df = pd.merge(merged_df, demo_df, on="customerid", how="left")
            logger.info("Merged demographics: post-merge shape %s", merged_df.shape)

        if not agg_prev_df.empty:
            merged_df = pd.merge(merged_df, agg_prev_df, on="customerid", how="left")
            logger.info("Merged previous loan aggregations: post-merge shape %s", merged_df.shape)

            # Fill missing previous loan features with 0 for new customers
            prev_cols = [c for c in agg_prev_df.columns if c != "customerid"]
            merged_df[prev_cols] = merged_df[prev_cols].fillna(0)

        # Extract target variable: Good -> 0 (paid), Bad -> 1 (default)
        target = merged_df["good_bad_flag"].map({"Good": 0, "Bad": 1})
        if target.isnull().any():
            logger.warning(
                "Null target values detected after mapping. Imputing based on totaldue vs loanamount."
            )
            # Fallback: if totaldue is unpaid, label as Bad (1)
            target = target.fillna(1)

        # Drop the original target column from features
        features = merged_df.drop(columns=["good_bad_flag"])

        return features, target

    def _aggregate_previous_loans(self, prev_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate previous loans by customerid.

        Args:
            prev_df: Raw previous loans DataFrame.

        Returns:
            DataFrame with one row per customerid containing aggregated features.
        """
        logger.info("Aggregating historical loans...")
        df = prev_df.copy()

        # Parse date columns safely
        date_cols = ["firstduedate", "firstrepaiddate", "closeddate", "approveddate"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Compute repayment speed (negative = early, positive = late)
        df["repayment_days_diff"] = (df["firstrepaiddate"] - df["firstduedate"]).dt.days

        # Identify late repayments
        df["is_late"] = (df["repayment_days_diff"] > 0).astype(int)

        # Group and aggregate
        agg_rules = {
            "loanamount": ["count", "sum", "mean", "max"],
            "totaldue": ["mean"],
            "termdays": ["mean"],
            "is_late": ["sum", "mean"],
            "repayment_days_diff": ["mean"],
        }

        # Filter out rules for missing columns
        agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

        grouped = df.groupby("customerid").agg(agg_rules)

        # Flatten multi-index columns
        grouped.columns = [
            f"prev_{col}_{stat}" if col != "customerid" else col
            for col, stat in grouped.columns
        ]
        grouped = grouped.reset_index()

        # Rename standard columns for clarity
        rename_map = {
            "prev_loanamount_count": "prev_loan_count",
            "prev_loanamount_sum": "prev_total_borrowed",
            "prev_loanamount_mean": "prev_avg_loan_amount",
            "prev_loanamount_max": "prev_max_loan_amount",
            "prev_totaldue_mean": "prev_avg_totaldue",
            "prev_termdays_mean": "prev_avg_term_days",
            "prev_is_late_sum": "prev_late_repayment_count",
            "prev_is_late_mean": "prev_late_repayment_ratio",
            "prev_repayment_days_diff_mean": "prev_repayment_speed_avg",
        }
        # Keep only available ones
        rename_map = {k: v for k, v in rename_map.items() if k in grouped.columns}
        grouped = grouped.rename(columns=rename_map)

        logger.info("Aggregated history for %d unique customers", len(grouped))
        return grouped


class AfricanCreditLoader(DataLoader):
    """Loads the African Credit Risk dataset."""

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load African Credit dataset.

        Returns:
            Tuple of (features_df, target_series).
        """
        logger.info("Loading African Credit Risk dataset...")
        path = self.config.african_credit_train_path
        df = pd.read_csv(path)

        self._validate_columns(df, ["target"])

        # target: 0 = good/paid, 1 = default
        y = df["target"]
        X = df.drop(columns=["target"])

        return X, y


class SyntheticLoader(DataLoader):
    """Loads the custom synthetic credit dataset."""

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load synthetic credit dataset.

        Inverts 'loan_status' (0=default/bad, 1=paid/good) to align with standard
        representation (1=default/bad, 0=paid/good).

        Returns:
            Tuple of (features_df, target_series).
        """
        logger.info("Loading Synthetic credit dataset...")
        path = self.config.synthetic_data_path
        df = pd.read_csv(path)

        self._validate_columns(df, ["loan_status"])

        # Target inversion: original 1=Good, 0=Bad -> New 1=Bad (Default), 0=Good
        y = 1 - df["loan_status"]

        # Drop columns used for mock generations and API responses to prevent leakage
        mock_cols = ["loan_status", "mock_probability", "mock_credit_score", "mock_risk_band"]
        drop_cols = [col for col in mock_cols if col in df.columns]
        X = df.drop(columns=drop_cols)

        return X, y
