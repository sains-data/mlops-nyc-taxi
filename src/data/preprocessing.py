"""
Data Preprocessing Module
Cleans and prepares NYC Taxi data for modeling
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PROCESSED_DIR = Path("data/processed")

# Column definitions for Yellow Taxi data
REQUIRED_COLUMNS = [
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime', 
    'passenger_count',
    'trip_distance',
    'PULocationID',
    'DOLocationID',
    'payment_type',
    'fare_amount',
    'tip_amount',
    'total_amount'
]


def clean_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NYC Taxi data by removing invalid records.
    
    Args:
        df: Raw taxi data
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Starting data cleaning. Initial rows: {len(df):,}")
    
    # Make a copy
    df = df.copy()
    
    # Select only required columns (if they exist)
    available_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
    df = df[available_cols]
    
    # Remove rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Removed {initial_rows - len(df):,} rows with missing values")
    
    # Filter valid fare amounts (between $2.5 and $500)
    initial_rows = len(df)
    df = df[(df['fare_amount'] >= 2.5) & (df['fare_amount'] <= 500)]
    logger.info(f"Removed {initial_rows - len(df):,} rows with invalid fare_amount")
    
    # Filter valid trip distances (between 0.1 and 100 miles)
    initial_rows = len(df)
    df = df[(df['trip_distance'] >= 0.1) & (df['trip_distance'] <= 100)]
    logger.info(f"Removed {initial_rows - len(df):,} rows with invalid trip_distance")
    
    # Filter valid passenger counts (between 1 and 6)
    initial_rows = len(df)
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    logger.info(f"Removed {initial_rows - len(df):,} rows with invalid passenger_count")
    
    # Filter valid total amounts
    initial_rows = len(df)
    df = df[(df['total_amount'] >= 2.5) & (df['total_amount'] <= 1000)]
    logger.info(f"Removed {initial_rows - len(df):,} rows with invalid total_amount")
    
    logger.info(f"Data cleaning complete. Final rows: {len(df):,}")
    
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from raw data.
    
    Args:
        df: Cleaned taxi data
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering...")
    
    df = df.copy()
    
    # Convert datetime columns
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    # Time-based features
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_year'] = df['pickup_datetime'].dt.year
    
    # Is weekend
    df['is_weekend'] = (df['pickup_dayofweek'] >= 5).astype(int)
    
    # Time of day category
    df['time_of_day'] = pd.cut(
        df['pickup_hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True
    )
    
    # Trip duration in minutes
    df['trip_duration_minutes'] = (
        df['dropoff_datetime'] - df['pickup_datetime']
    ).dt.total_seconds() / 60
    
    # Filter valid trip duration (between 1 and 180 minutes)
    df = df[(df['trip_duration_minutes'] >= 1) & (df['trip_duration_minutes'] <= 180)]
    
    # Speed (mph)
    df['avg_speed_mph'] = df['trip_distance'] / (df['trip_duration_minutes'] / 60)
    
    # Filter valid speed (between 1 and 60 mph)
    df = df[(df['avg_speed_mph'] >= 1) & (df['avg_speed_mph'] <= 60)]
    
    # Drop original datetime columns
    df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 
                          'pickup_datetime', 'dropoff_datetime'])
    
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    
    return df.reset_index(drop=True)


def prepare_features_target(
    df: pd.DataFrame,
    target_column: str = 'fare_amount'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and target (y) for modeling.
    
    Args:
        df: DataFrame with engineered features
        target_column: Name of target column
        
    Returns:
        Tuple of (X, y)
    """
    # Feature columns
    feature_columns = [
        'trip_distance',
        'passenger_count',
        'PULocationID',
        'DOLocationID',
        'pickup_hour',
        'pickup_dayofweek',
        'pickup_month',
        'is_weekend',
        'trip_duration_minutes'
    ]
    
    # Filter available columns
    available_features = [col for col in feature_columns if col in df.columns]
    
    X = df[available_features]
    y = df[target_column]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: Optional[Path] = None
) -> None:
    """Save processed datasets to parquet files."""
    if output_dir is None:
        output_dir = PROCESSED_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine X and y for each split
    train_df = X_train.copy()
    train_df['fare_amount'] = y_train.values
    
    val_df = X_val.copy()
    val_df['fare_amount'] = y_val.values
    
    test_df = X_test.copy()
    test_df['fare_amount'] = y_test.values
    
    # Save to parquet
    train_df.to_parquet(output_dir / 'train.parquet', index=False)
    val_df.to_parquet(output_dir / 'val.parquet', index=False)
    test_df.to_parquet(output_dir / 'test.parquet', index=False)
    
    logger.info(f"Saved processed data to {output_dir}")


if __name__ == "__main__":
    # Example usage
    from ingestion import download_nyc_taxi_data, load_parquet_data
    
    # Load data
    files = download_nyc_taxi_data(year=2024, months=[1])
    df = load_parquet_data(files[0])
    
    # Clean and engineer features
    df_clean = clean_taxi_data(df)
    df_features = engineer_features(df_clean)
    
    # Prepare for modeling
    X, y = prepare_features_target(df_features)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Save
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
