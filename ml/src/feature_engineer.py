"""FeatureEngineer class for the credit risk ML pipeline.

Engineers domain-specific credit scoring features (e.g. age, loan ratios,
repayment reliability, and loan escalation flags) and drops raw timestamps.
"""

import logging

import pandas as pd

from ml.src.config import PipelineConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Derives predictive credit risk features from raw and cleaned columns.

    Designed to handle different datasets by checking column availability
    before applying transformations.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise with pipeline config.

        Args:
            config: Pipeline configuration settings.
        """
        self.config: PipelineConfig = config

    def engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        """Orchestrate all feature engineering steps.

        Args:
            X: Input DataFrame after cleaning.

        Returns:
            DataFrame with newly engineered features.
        """
        df = X.copy()

        # Step 1: Create age feature
        df = self._create_age_feature(df)

        # Step 2: Create loan-to-fee ratio
        df = self._create_loan_ratios(df)

        # Step 3: Create repeat customer flag
        df = self._create_repeat_flag(df)

        # Step 4: Create loan processing delay feature
        df = self._create_processing_time(df)

        # Step 5: Create repayment and history escalation features
        df = self._create_repayment_features(df)

        # Step 6: Drop raw timestamp and non-predictive sequence columns
        df = self._drop_raw_dates(df)

        return df

    def _create_age_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate applicant age at application time in years.

        Age = (approveddate - birthdate) in days / 365.25

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with 'age' feature added.
        """
        if "approveddate" in df.columns and "birthdate" in df.columns:
            logger.info("Engineering feature: age")
            # Calculate difference in years
            diff_days = (df["approveddate"] - df["birthdate"]).dt.days
            df["age"] = diff_days / 365.25
            
            # Fill missing age with dataset median or reasonable default (e.g. 35)
            median_age = df["age"].median()
            if pd.isna(median_age):
                median_age = 35.0
            df["age"] = df["age"].fillna(median_age)
        return df

    def _create_loan_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the ratio of interest/fee to the principal loan amount.

        loan_fee_ratio = (totaldue - loanamount) / loanamount

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with 'loan_fee_ratio' feature.
        """
        if "totaldue" in df.columns and "loanamount" in df.columns:
            logger.info("Engineering feature: loan_fee_ratio")
            df["loan_fee_ratio"] = (df["totaldue"] - df["loanamount"]) / df["loanamount"]
            # Handle possible zero division
            df["loan_fee_ratio"] = df["loan_fee_ratio"].fillna(0.0)
        return df

    def _create_repeat_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary indicator for repeat borrowers.

        is_repeat_customer = 1 if loannumber > 1 else 0

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with 'is_repeat_customer' feature.
        """
        if "loannumber" in df.columns:
            logger.info("Engineering feature: is_repeat_customer")
            df["is_repeat_customer"] = (df["loannumber"] > 1).astype(int)
        return df

    def _create_processing_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate application processing duration.

        days_since_creation = approveddate - creationdate (in days)

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with 'days_since_creation' feature.
        """
        if "approveddate" in df.columns and "creationdate" in df.columns:
            logger.info("Engineering feature: days_since_creation")
            df["days_since_creation"] = (df["approveddate"] - df["creationdate"]).dt.days
            df["days_since_creation"] = df["days_since_creation"].fillna(0.0)
        return df

    def _create_repayment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive repayment risk features based on previous loan history.

        Ratios:
            - prev_repayment_reliability = 1 - prev_late_repayment_ratio
            - loan_amount_vs_prev_avg = loanamount / prev_avg_loan_amount
            - loan_escalation_flag = 1 if loanamount > 1.5 * prev_avg_loan_amount else 0

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with repayment features.
        """
        # Repayment reliability
        if "prev_late_repayment_ratio" in df.columns:
            logger.info("Engineering feature: prev_repayment_reliability")
            df["prev_repayment_reliability"] = 1.0 - df["prev_late_repayment_ratio"]
            df["prev_repayment_reliability"] = df["prev_repayment_reliability"].fillna(1.0)

        # Loan escalation features
        if "loanamount" in df.columns and "prev_avg_loan_amount" in df.columns:
            logger.info("Engineering features: loan_amount_vs_prev_avg & loan_escalation_flag")
            # Divide safely, if no previous loans (prev_avg_loan_amount == 0), set ratio to 1.0 (no escalation)
            df["loan_amount_vs_prev_avg"] = df["loanamount"] / df["prev_avg_loan_amount"]
            df["loan_amount_vs_prev_avg"] = df["loan_amount_vs_prev_avg"].replace([float("inf"), -float("inf")], 1.0)
            df["loan_amount_vs_prev_avg"] = df["loan_amount_vs_prev_avg"].fillna(1.0)

            # Flag if the requested loan is significantly larger (>50%) than their previous average
            df["loan_escalation_flag"] = (df["loan_amount_vs_prev_avg"] > 1.5).astype(int)

        return df

    def _drop_raw_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop datetime objects and raw loan sequence numbers.

        These columns cannot be fed directly into standard models.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with raw date and sequence columns removed.
        """
        # Find any columns that are datetime or dates
        cols_to_drop = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                cols_to_drop.append(col)

        # Drop other specific non-predictive features that were replaced
        non_predictive = ["loannumber", "referredby"]
        for col in non_predictive:
            if col in df.columns:
                cols_to_drop.append(col)

        if cols_to_drop:
            logger.info("Dropping raw datetime and sequence columns: %s", cols_to_drop)
            df = df.drop(columns=cols_to_drop)

        return df
