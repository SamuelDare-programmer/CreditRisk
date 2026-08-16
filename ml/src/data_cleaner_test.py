import numpy as np
import pandas as pd
from typing import List

from ml.src.config import PipelineConfig

class DataCleaner:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def clean(self, X: pd.DataFrame, y: pd.Series):
        initial_shape = X.shape

        X_clean = X.copy()
        y_clean = y.copy()

        #drop identifiers
        #drop sparse columns
        # convert datetimes
        # fill missing values
        # treat outliers
        # standardize categorical columns

        # drop sparse columns
        X_clean = self._drop_sparse_columns(X_clean)

        X_clean = self._drop_identifiers_columns(X_clean)

        X_clean = self._parse_datetime(X_clean)

        X_clean = self._handle_missing_values(X_clean)

        X_clean = self._handle_outliers(X_clean)

        X_clean = self._standardize_categorical_columns(X_clean)
    
    def _drop_sparse_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._config.columns_to_drop:
            df = df.drop(col)
        return df
    
    def _drop_identifiers_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [col for col in self._config.identifiers]

        df = df.drop(columns=cols)
        return df
    
    def _parse_datetime(self, df: pd.DataFrame):

        date_cols = ["date", 'birth', 'creation', "approved", 'time']
        df = df.copy()
        for col in df.columns.to_list():
            if any(kw in col.lower() for kw in date_cols):
                if not pd.api.types.is_datetime64_any_dtype:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def _handle_missing_values(self, df:pd.DataFrame):
        df = df.copy()

        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            null = df[col].isnull().sum()
            if null > 0:
                median_value = df[col].median()
                if pd.isna(median_value):
                    median_value = 0
                df[col] = df[col].fillna(median_value)

        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                median_value = "Unknown"
                df[col] = df[col].fillna(median_value)

        return df
    

    def _handle_outliers(self, df:pd.DataFrame):
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quartile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                ordinary_max = df[col].max()
                ordinary_min = df[col].min()

                upperbound = q3 - 1.5 * iqr
                lowerbound = q1 + 1.5 * iqr

                df[col] = np.clip(df[col], lowerbound, upperbound)

        return df

    def _standardize_categorical_columns(self, df: pd.DataFrame):
        cat_cols = df.select_dtypes(include=['object']).columns

        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()

        return df