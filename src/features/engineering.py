"""
Feature Engineering Module
Advanced feature transformations for NYC Taxi data
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxiFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for NYC Taxi features.
    Handles scaling and encoding in a sklearn-compatible way.
    """
    
    def __init__(
        self,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None,
        scale_numerical: bool = True
    ):
        self.numerical_features = numerical_features or [
            'trip_distance',
            'passenger_count', 
            'pickup_hour',
            'pickup_dayofweek',
            'trip_duration_minutes'
        ]
        self.categorical_features = categorical_features or [
            'PULocationID',
            'DOLocationID'
        ]
        self.scale_numerical = scale_numerical
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer on training data."""
        logger.info("Fitting feature transformer...")
        
        # Fit scaler on numerical features
        if self.scale_numerical:
            num_cols = [c for c in self.numerical_features if c in X.columns]
            if num_cols:
                self.scaler.fit(X[num_cols])
        
        # Fit label encoders on categorical features
        for col in self.categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Store feature names
        self.feature_names_ = list(X.columns)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        X = X.copy()
        
        # Scale numerical features
        if self.scale_numerical:
            num_cols = [c for c in self.numerical_features if c in X.columns]
            if num_cols:
                X[num_cols] = self.scaler.transform(X[num_cols])
        
        # Encode categorical features
        for col, le in self.label_encoders.items():
            if col in X.columns:
                # Handle unseen categories
                X[col] = X[col].astype(str)
                X[col] = X[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def save(self, filepath: Path):
        """Save transformer to disk."""
        joblib.dump(self, filepath)
        logger.info(f"Saved transformer to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'TaxiFeatureTransformer':
        """Load transformer from disk."""
        transformer = joblib.load(filepath)
        logger.info(f"Loaded transformer from {filepath}")
        return transformer


def create_polynomial_features(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2
) -> pd.DataFrame:
    """Create polynomial features for specified columns."""
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for d in range(2, degree + 1):
                df[f"{col}_pow{d}"] = df[col] ** d
    
    return df


def create_interaction_features(
    df: pd.DataFrame,
    feature_pairs: List[Tuple[str, str]]
) -> pd.DataFrame:
    """Create interaction features between pairs of columns."""
    df = df.copy()
    
    for col1, col2 in feature_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    
    return df


def create_cyclic_features(
    df: pd.DataFrame,
    column: str,
    max_value: int
) -> pd.DataFrame:
    """
    Create cyclic (sin/cos) features for periodic variables.
    Useful for hour of day, day of week, month, etc.
    """
    df = df.copy()
    
    if column in df.columns:
        df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / max_value)
        df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / max_value)
    
    return df


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Sample data
    data = {
        'trip_distance': [2.5, 3.0, 1.5],
        'passenger_count': [1, 2, 1],
        'pickup_hour': [8, 14, 22],
        'pickup_dayofweek': [0, 3, 5],
        'PULocationID': [100, 150, 100],
        'DOLocationID': [200, 250, 200],
        'trip_duration_minutes': [15, 20, 10]
    }
    df = pd.DataFrame(data)
    
    # Transform
    transformer = TaxiFeatureTransformer()
    df_transformed = transformer.fit_transform(df)
    print(df_transformed)
