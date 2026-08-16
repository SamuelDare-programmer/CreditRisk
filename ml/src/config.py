"""Configuration module for the credit risk ML pipeline.

Defines the PipelineConfig dataclass holding all file paths, feature listings,
reproducibility settings, and hyper-parameters.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    """Configuration settings for the credit risk preprocessing and training pipeline.

    Provides default paths and settings for dataset ingestion, cleaning,
    transformation, and training stages.
    """

    # --- Dataset paths (relative to project root) ---
    superlender_demographics_path: str = "ml/data/loan_default_datasets/traindemographics.csv"
    superlender_performance_path: str = "ml/data/loan_default_datasets/trainperf.csv"
    superlender_prevloans_path: str = "ml/data/loan_default_datasets/trainprevloans.csv"
    african_credit_train_path: str = "ml/data/african_credit_risk_datasets/Train.csv"
    synthetic_data_path: str = "ml/data/synthetic_ng_credit_data.csv"

    # --- Serialisation paths ---
    model_save_path: str = "ml/artefacts/model.pkl"
    preprocessor_save_path: str = "ml/artefacts/preprocessor.pkl"
    explainer_save_path: str = "ml/artefacts/explainer.pkl"

    # --- Reproducibility and split parameters ---
    random_state: int = 42
    test_size: float = 0.2
    smote_k_neighbors: int = 5
    cv_folds: int = 5

    # --- Feature column listings (empty by default to trigger auto-detection) ---
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    binary_features: List[str] = field(default_factory=list)

    # --- Sparse columns to drop during data cleaning ---
    # Drops sparse variables which are mostly missing (1%-13% fill rate)
    columns_to_drop: List[str] = field(
        default_factory=lambda: [
            "bank_branch_clients",
            "level_of_education_clients",
            "referredby",
        ]
    )

    # --- Identifiers to drop during cleaning ---
    identifiers: List[str] = field(
        default_factory=lambda: [
            "customerid",
            "systemloanid",
            "ID",
            "customer_id",
            "tbl_loan_id",
            "lender_id",
        ]
    )
