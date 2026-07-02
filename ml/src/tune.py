"""Hyperparameter tuning module for the credit risk ML pipeline.

Uses GridSearchCV with imblearn pipelines (applying SMOTE inside folds) to find
the optimal parameters for Logistic Regression, XGBoost, and LightGBM.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ml.src.config import PipelineConfig
from ml.src.data_cleaner import DataCleaner
from ml.src.data_loader import SuperLenderLoader
from ml.src.feature_engineer import FeatureEngineer
from ml.src.preprocessor import Preprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def tune_hyperparameters() -> None:
    """Orchestrate hyperparameter tuning for Logistic Regression, XGBoost, and LightGBM."""
    config = PipelineConfig()
    
    # 1. Load data
    loader = SuperLenderLoader(config)
    X, y = loader.load()
    
    # 2. Clean data
    cleaner = DataCleaner(config)
    X, y = cleaner.clean(X, y)
    
    # 3. Engineer features
    engineer = FeatureEngineer(config)
    X = engineer.engineer(X)
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    
    # 5. Transform features (ColumnTransformer only, no SMOTE here as SMOTE goes inside grid CV)
    preprocessor = Preprocessor(config)
    X_train_proc, _ = preprocessor.fit_transform(X_train, y_train, use_smote=False)
    
    logger.info("Transformed features shape: %s", X_train_proc.shape)
    
    # Setup CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
    
    # Define grids
    grids = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=config.random_state),
            "params": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__solver": ["lbfgs", "liblinear"]
            }
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=config.random_state, verbose=-1),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__num_leaves": [15, 31, 63],
                "model__max_depth": [3, 6, -1]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=config.random_state, use_label_encoder=False, eval_metric="logloss"),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 6, 8]
            }
        }
    }
    
    best_results = {}
    
    for name, grid_info in grids.items():
        logger.info("Tuning %s...", name)
        
        # Build imblearn pipeline to prevent leakage
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=config.random_state, k_neighbors=config.smote_k_neighbors)),
            ("model", grid_info["model"])
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid_info["params"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_proc, y_train.to_numpy())
        
        best_auc = grid_search.best_score_
        best_params = grid_search.best_params_
        
        # Clean parameter names (remove 'model__' prefix)
        cleaned_params = {k.replace("model__", ""): v for k, v in best_params.items()}
        
        logger.info("%s — Best CV AUC-ROC: %.4f", name, best_auc)
        logger.info("%s — Best Parameters: %s", name, cleaned_params)
        
        best_results[name] = {
            "auc_roc": best_auc,
            "params": cleaned_params
        }

    logger.info("--- Parameter Tuning Summary ---")
    for name, res in best_results.items():
        logger.info("Model: %s | Best CV AUC-ROC: %.4f | Params: %s", name, res["auc_roc"], res["params"])


if __name__ == "__main__":
    tune_hyperparameters()
