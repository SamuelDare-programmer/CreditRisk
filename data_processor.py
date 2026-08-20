"""
Data processing and feature engineering pipeline.

This module defines the DataProcessor class, which encapsulates all
preprocessing steps for the credit risk model. It uses scikit-learn
pipelines to ensure transformations are applied consistently.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class DataProcessor:
    """
    Encapsulates the data preprocessing pipeline for the credit risk model.

    This class is responsible for:
    1.  Handling missing values (imputation).
    2.  Engineering new features (e.g., debt-to-income ratio).
    3.  Encoding categorical features.
    4.  Scaling numerical features.

    The output is a scikit-learn pipeline object that can be fitted to training
    data and used to transform new data.
    """

    def __init__(self, numerical_features: list, categorical_features: list) -> None:
        """
        Initialise the DataProcessor with feature lists.

        Args:
            numerical_features (list): List of names for numerical columns.
            categorical_features (list): List of names for categorical columns.
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self._pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        """
        Creates the full scikit-learn preprocessing pipeline.

        The pipeline includes separate transformers for numerical and categorical
        features, which are then combined.

        Returns:
            Pipeline: The scikit-learn preprocessing pipeline.
        """
        # Pipeline for numerical features:
        # 1. Impute missing values with the median (robust to outliers).
        # 2. Scale features using RobustScaler (also robust to outliers).
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        # Pipeline for categorical features:
        # 1. Impute missing values with the most frequent value.
        # 2. One-hot encode the features, ignoring unknown categories at predict time.
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Use ColumnTransformer to apply different transformers to different columns.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'  # Keep other columns (if any)
        )

        # The final pipeline object
        # Note: We are not adding feature engineering as a step here, as it's
        # easier to perform it on the DataFrame directly before the pipeline.
        return preprocessor

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the pipeline to the data and transforms it.

        Args:
            df (pd.DataFrame): The training data.

        Returns:
            pd.DataFrame: The transformed and preprocessed data.
        """
        # First, perform DataFrame-native feature engineering
        df_engineered = self.engineer_features(df.copy())

        # Now, fit and transform using the scikit-learn pipeline
        processed_data = self._pipeline.fit_transform(df_engineered)

        # Get feature names after one-hot encoding
        ohe_feature_names = self._pipeline.named_transformers_['cat'] \
            .named_steps['onehot'].get_feature_names_out(self.categorical_features)

        all_feature_names = self.numerical_features + list(ohe_feature_names)

        return pd.DataFrame(processed_data, columns=all_feature_names, index=df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new data using the already-fitted pipeline.

        Args:
            df (pd.DataFrame): The new data to transform (e.g., test set).

        Returns:
            pd.DataFrame: The transformed data.
        """
        df_engineered = self.engineer_features(df.copy())
        processed_data = self._pipeline.transform(df_engineered)
        ohe_feature_names = self._pipeline.named_transformers_['cat'] \
            .named_steps['onehot'].get_feature_names_out(self.categorical_features)
        all_feature_names = self.numerical_features + list(ohe_feature_names)

        return pd.DataFrame(processed_data, columns=all_feature_names, index=df.index)

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Engineers new features from the raw data."""
        # Avoid division by zero
        df['debt_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
        return df

    def get_pipeline(self) -> Pipeline:
        """Returns the fitted pipeline object for serialization."""
        return self._pipeline
