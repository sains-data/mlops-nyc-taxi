"""
Tests for NYC Taxi MLOps Project
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataIngestion:
    """Tests for data ingestion module."""
    
    def test_imports(self):
        """Test that data ingestion module can be imported."""
        from src.data.ingestion import download_nyc_taxi_data, load_parquet_data
        assert callable(download_nyc_taxi_data)
        assert callable(load_parquet_data)


class TestDataPreprocessing:
    """Tests for data preprocessing module."""
    
    def test_imports(self):
        """Test that preprocessing module can be imported."""
        from src.data.preprocessing import clean_taxi_data, engineer_features
        assert callable(clean_taxi_data)
        assert callable(engineer_features)
    
    def test_clean_taxi_data(self):
        """Test data cleaning function."""
        from src.data.preprocessing import clean_taxi_data
        
        # Create sample data
        data = {
            'tpep_pickup_datetime': pd.date_range('2024-01-01', periods=10, freq='h'),
            'tpep_dropoff_datetime': pd.date_range('2024-01-01 00:15:00', periods=10, freq='h'),
            'passenger_count': [1, 2, 1, 3, 1, 2, 1, 1, 2, 1],
            'trip_distance': [2.5, 3.0, 1.5, 5.0, 2.0, 4.0, 1.0, 3.5, 2.5, 1.5],
            'PULocationID': [100, 150, 100, 200, 100, 150, 100, 200, 100, 150],
            'DOLocationID': [200, 250, 200, 300, 200, 250, 200, 300, 200, 250],
            'payment_type': [1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
            'fare_amount': [10.0, 15.0, 8.0, 25.0, 12.0, 20.0, 6.0, 18.0, 14.0, 7.0],
            'tip_amount': [2.0, 3.0, 0.0, 5.0, 2.5, 0.0, 1.0, 4.0, 0.0, 1.5],
            'total_amount': [15.0, 22.0, 10.0, 35.0, 18.0, 24.0, 9.0, 26.0, 17.0, 11.0]
        }
        df = pd.DataFrame(data)
        
        # Clean
        df_clean = clean_taxi_data(df)
        
        # Assertions
        assert len(df_clean) <= len(df)
        assert 'fare_amount' in df_clean.columns


class TestModels:
    """Tests for model training module."""
    
    def test_imports(self):
        """Test that model module can be imported."""
        from src.models.train import get_model, calculate_metrics, MODELS
        assert callable(get_model)
        assert callable(calculate_metrics)
        assert len(MODELS) > 0
    
    def test_get_model(self):
        """Test model factory function."""
        from src.models.train import get_model
        
        model = get_model('linear_regression')
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        from src.models.train import calculate_metrics
        
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 33, 38, 52])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0


class TestAPI:
    """Tests for API module."""
    
    def test_imports(self):
        """Test that API module can be imported."""
        from src.serving.api import app, TripFeatures, PredictionResponse
        assert app is not None
    
    def test_trip_features_model(self):
        """Test Pydantic model validation."""
        from src.serving.api import TripFeatures
        
        # Valid data
        trip = TripFeatures(
            trip_distance=2.5,
            passenger_count=1,
            PULocationID=100,
            DOLocationID=200,
            pickup_hour=14,
            pickup_dayofweek=2
        )
        
        assert trip.trip_distance == 2.5
        assert trip.passenger_count == 1


class TestFeatureEngineering:
    """Tests for feature engineering module."""
    
    def test_imports(self):
        """Test that feature engineering module can be imported."""
        from src.features.engineering import TaxiFeatureTransformer
        assert TaxiFeatureTransformer is not None
    
    def test_transformer(self):
        """Test feature transformer."""
        from src.features.engineering import TaxiFeatureTransformer
        
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
        transformer = TaxiFeatureTransformer(scale_numerical=False)
        df_transformed = transformer.fit_transform(df)
        
        assert df_transformed.shape == df.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
