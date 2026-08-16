"""Unit tests for the credit risk preprocessing and model training pipeline.

Tests data cleaning, feature engineering, preprocessing, and model training
components using mock DataFrames.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.src.config import PipelineConfig
from ml.src.data_cleaner import DataCleaner
from ml.src.feature_engineer import FeatureEngineer
from ml.src.preprocessor import Preprocessor
from ml.src.model_trainer import ModelTrainer


@pytest.fixture
def sample_config() -> PipelineConfig:
    """Fixture to provide a clean PipelineConfig instance."""
    return PipelineConfig()


@pytest.fixture
def mock_raw_df() -> pd.DataFrame:
    """Fixture to provide a mock raw DataFrame with missing values and outliers."""
    return pd.DataFrame(
        {
            "customerid": ["id_1", "id_2", "id_3", "id_4", "id_5"],
            "systemloanid": [101, 102, 103, 104, 105],
            "loannumber": [1, 2, 5, 1, 3],
            "loanamount": [10000.0, 15000.0, 20000.0, 100000.0, 12000.0],  # 100000 is an outlier
            "totaldue": [11500.0, 17250.0, 23000.0, 115000.0, 13800.0],
            "termdays": [30, 30, 60, 30, 15],
            "bank_account_type": ["Savings", "Other", None, "Savings", "Savings"],
            "bank_name_clients": ["GT Bank", "Access Bank", "GT Bank", "UBA", "Fidelity"],
            "employment_status_clients": ["Permanent", None, "Self-Employed", "Permanent", "Permanent"],
            "bank_branch_clients": [None, None, "Ikeja", None, None],  # Sparse column
            "level_of_education_clients": [None, "Graduate", None, None, None],  # Sparse column
            "referredby": [None, None, None, "ref_123", None],  # Sparse column
            "longitude_gps": [3.4, 3.5, 3.4, 3.6, 3.4],
            "latitude_gps": [6.5, 6.4, 6.5, 6.6, 6.5],
            "approveddate": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
            "creationdate": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-04", "2023-01-04"]),
            "birthdate": pd.to_datetime(["1990-01-01", "1985-05-12", "1995-09-30", "1980-04-15", "1988-11-22"]),
            "prev_late_repayment_ratio": [0.0, 0.5, 0.0, 1.0, 0.25],
            "prev_avg_loan_amount": [0.0, 10000.0, 15000.0, 8000.0, 12000.0],
        }
    )


def test_data_cleaner(sample_config, mock_raw_df):
    """Test DataCleaner columns removal, imputation, and outlier capping."""
    cleaner = DataCleaner(sample_config)
    y = pd.Series([0, 1, 0, 1, 0])
    X_clean, y_clean = cleaner.clean(mock_raw_df, y)

    # 1. Assert sparse columns dropped
    for col in sample_config.columns_to_drop:
        assert col not in X_clean.columns

    # 2. Assert identifier columns dropped
    for col in sample_config.identifiers:
        assert col not in X_clean.columns

    # 3. Assert no null values remain
    assert X_clean.isnull().sum().sum() == 0

    # 4. Assert categorical values mapped to lowercase
    assert X_clean["bank_account_type"].iloc[2] == "unknown"
    assert X_clean["employment_status_clients"].iloc[1] == "unknown"


def test_feature_engineer(sample_config, mock_raw_df):
    """Test FeatureEngineer derived variables logic."""
    cleaner = DataCleaner(sample_config)
    y = pd.Series([0, 1, 0, 1, 0])
    X_clean, _ = cleaner.clean(mock_raw_df, y)

    engineer = FeatureEngineer(sample_config)
    X_featured = engineer.engineer(X_clean)

    # Assert new features exist
    assert "age" in X_featured.columns
    assert "loan_fee_ratio" in X_featured.columns
    assert "is_repeat_customer" in X_featured.columns
    assert "days_since_creation" in X_featured.columns
    assert "prev_repayment_reliability" in X_featured.columns
    assert "loan_amount_vs_prev_avg" in X_featured.columns
    assert "loan_escalation_flag" in X_featured.columns

    # Assert raw datetimes are dropped
    for col in X_featured.columns:
        assert not pd.api.types.is_datetime64_any_dtype(X_featured[col])

    # Assert correctness of calculations
    assert np.allclose(X_featured["loan_fee_ratio"], 0.15)
    assert X_featured["is_repeat_customer"].iloc[0] == 0
    assert X_featured["is_repeat_customer"].iloc[2] == 1


def test_preprocessor(sample_config, mock_raw_df):
    """Test Preprocessor transformation and serialization."""
    cleaner = DataCleaner(sample_config)
    y = pd.Series([0, 1, 0, 1, 0])
    X_clean, _ = cleaner.clean(mock_raw_df, y)

    engineer = FeatureEngineer(sample_config)
    X_featured = engineer.engineer(X_clean)

    # Use a mock config with empty lists to test type auto-detection
    sample_config.smote_k_neighbors = 1
    preprocessor = Preprocessor(sample_config)
    X_trans, y_res = preprocessor.fit_transform(X_featured, y)

    # Assert shape has changed (One-hot encoding increases features)
    assert X_trans.shape[1] > X_featured.shape[1]
    
    # Assert SMOTE has balanced the classes
    unique, counts = np.unique(y_res, return_counts=True)
    class_dist = dict(zip(unique, counts))
    assert class_dist[0] == class_dist[1]

    # Test test set transform (no fit, no SMOTE)
    X_test_trans = preprocessor.transform(X_featured)
    assert X_test_trans.shape[0] == X_featured.shape[0]
    assert X_test_trans.shape[1] == X_trans.shape[1]

    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pkl_path = os.path.join(tmpdir, "preprocessor.pkl")
        preprocessor.save(temp_pkl_path)
        assert os.path.exists(temp_pkl_path)

        loaded_preprocessor = Preprocessor.load(temp_pkl_path, sample_config)
        assert loaded_preprocessor._is_fitted
        assert len(loaded_preprocessor.numeric_features) == len(preprocessor.numeric_features)


def test_model_trainer(sample_config):
    """Test ModelTrainer training and metrics generation."""
    trainer = ModelTrainer(sample_config)
    
    # Generate mock preprocessed training data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.choice([0, 1], size=100)
    
    X_test = np.random.randn(20, 10)
    y_test = np.random.choice([0, 1], size=20)

    # Train all models
    cv_results = trainer.train_all(X_train, y_train)
    assert len(cv_results) == 4
    for model_name in ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]:
        assert model_name in cv_results
        assert "mean" in cv_results[model_name]

    # Evaluate all models
    eval_results = trainer.evaluate_all(X_test, y_test)
    assert len(eval_results) == 4
    for model_name in eval_results:
        assert "auc_roc" in eval_results[model_name]
        assert "f1" in eval_results[model_name]
        assert "confusion_matrix" in eval_results[model_name]

    # Get best model
    best_name, best_model = trainer.get_best_model()
    assert best_name in trainer.models
    assert isinstance(best_model, LogisticRegression) or hasattr(best_model, "fit")
