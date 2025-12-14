"""
Model Prediction Module
Load trained models and make predictions
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import mlflow
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FarePredictor:
    """
    Predictor class for NYC Taxi Fare prediction.
    Handles model loading and inference.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: str = "http://localhost:5000"
    ):
        """
        Initialize predictor with model.
        
        Args:
            model_path: Path to saved model file (joblib)
            mlflow_run_id: MLFlow run ID to load model from
            mlflow_tracking_uri: MLFlow tracking server URI
        """
        self.model = None
        self.model_path = model_path
        self.mlflow_run_id = mlflow_run_id
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
        if model_path:
            self.load_from_path(model_path)
        elif mlflow_run_id:
            self.load_from_mlflow(mlflow_run_id)
    
    def load_from_path(self, model_path: Path) -> None:
        """Load model from local path."""
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    
    def load_from_mlflow(self, run_id: str) -> None:
        """Load model from MLFlow."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        model_uri = f"runs:/{run_id}/model"
        self.model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from MLFlow run: {run_id}")
    
    def predict(self, features: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Make predictions on input features.
        
        Args:
            features: Input features as DataFrame, dict, or list of dicts
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_from_path or load_from_mlflow first.")
        
        # Convert input to DataFrame
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif isinstance(features, list):
            features = pd.DataFrame(features)
        
        predictions = self.model.predict(features)
        
        return predictions
    
    def predict_single(
        self,
        trip_distance: float,
        passenger_count: int,
        PULocationID: int,
        DOLocationID: int,
        pickup_hour: int,
        pickup_dayofweek: int,
        pickup_month: int = 1,
        is_weekend: int = 0,
        trip_duration_minutes: float = 15.0
    ) -> float:
        """
        Make prediction for a single trip.
        
        Returns:
            Predicted fare amount
        """
        features = {
            'trip_distance': trip_distance,
            'passenger_count': passenger_count,
            'PULocationID': PULocationID,
            'DOLocationID': DOLocationID,
            'pickup_hour': pickup_hour,
            'pickup_dayofweek': pickup_dayofweek,
            'pickup_month': pickup_month,
            'is_weekend': is_weekend,
            'trip_duration_minutes': trip_duration_minutes
        }
        
        prediction = self.predict(features)[0]
        
        return float(prediction)


def batch_predict(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    feature_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Make batch predictions on a dataset.
    
    Args:
        model_path: Path to trained model
        data_path: Path to input data (parquet)
        output_path: Path to save predictions
        feature_columns: Columns to use as features
        
    Returns:
        DataFrame with predictions
    """
    # Load model
    predictor = FarePredictor(model_path=model_path)
    
    # Load data
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows from {data_path}")
    
    # Select features
    if feature_columns:
        X = df[feature_columns]
    else:
        # Default features
        feature_columns = [
            'trip_distance', 'passenger_count', 'PULocationID', 'DOLocationID',
            'pickup_hour', 'pickup_dayofweek', 'pickup_month', 'is_weekend',
            'trip_duration_minutes'
        ]
        available_cols = [c for c in feature_columns if c in df.columns]
        X = df[available_cols]
    
    # Predict
    predictions = predictor.predict(X)
    
    # Add predictions to dataframe
    result_df = df.copy()
    result_df['predicted_fare'] = predictions
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    
    return result_df


if __name__ == "__main__":
    # Example usage
    predictor = FarePredictor(model_path=Path("models/best_model.joblib"))
    
    # Single prediction
    fare = predictor.predict_single(
        trip_distance=2.5,
        passenger_count=1,
        PULocationID=100,
        DOLocationID=200,
        pickup_hour=14,
        pickup_dayofweek=2
    )
    print(f"Predicted fare: ${fare:.2f}")
