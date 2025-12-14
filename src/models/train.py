"""
Model Training Module
Train and compare multiple ML models with MLFlow tracking
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model registry
MODELS = {
    'linear_regression': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'random_forest': RandomForestRegressor,
    'gradient_boosting': GradientBoostingRegressor,
}

# Default hyperparameters
DEFAULT_PARAMS = {
    'linear_regression': {},
    'ridge': {'alpha': 1.0},
    'lasso': {'alpha': 1.0},
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'n_jobs': -1,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42
    },
}


def get_model(model_name: str, params: Optional[Dict] = None):
    """Get a model instance by name."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    model_class = MODELS[model_name]
    model_params = params or DEFAULT_PARAMS.get(model_name, {})
    
    return model_class(**model_params)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None,
    experiment_name: str = "nyc-taxi-fare-prediction",
    tracking_uri: str = None
) -> Tuple[Any, Dict[str, float], str]:
    """
    Train a model with MLFlow tracking.
    
    Args:
        model_name: Name of model to train
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        params: Model hyperparameters
        experiment_name: MLFlow experiment name
        tracking_uri: MLFlow tracking server URI
        
    Returns:
        Tuple of (trained model, metrics dict, run_id)
    """
    # Setup MLFlow - use centralized config for consistent tracking URI
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        from src.config.mlflow_config import get_tracking_uri
        mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        logger.info(f"Starting MLFlow run: {run_id}")
        
        # Get model
        model = get_model(model_name, params)
        
        # Log parameters
        model_params = params or DEFAULT_PARAMS.get(model_name, {})
        mlflow.log_params(model_params)
        mlflow.log_param("model_name", model_name)
        
        # Train
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train.values, y_train_pred)
        val_metrics = calculate_metrics(y_val.values, y_val_pred)
        
        # Log metrics
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value)
        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"Training complete. Val RMSE: {val_metrics['rmse']:.4f}")
        
        return model, val_metrics, run_id


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    models: Optional[list] = None,
    experiment_name: str = "nyc-taxi-fare-prediction",
    tracking_uri: str = None
) -> pd.DataFrame:
    """
    Train all models and compare results.
    
    Returns:
        DataFrame with model comparison results
    """
    if models is None:
        models = list(MODELS.keys())
    
    results = []
    
    for model_name in models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*50}")
        
        try:
            model, metrics, run_id = train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                experiment_name=experiment_name,
                tracking_uri=tracking_uri
            )
            
            results.append({
                'model': model_name,
                'run_id': run_id,
                **metrics
            })
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    
    logger.info("\n" + "="*50)
    logger.info("MODEL COMPARISON")
    logger.info("="*50)
    logger.info(f"\n{results_df.to_string()}")
    
    return results_df


def save_model(model: Any, filepath: Path) -> None:
    """Save model to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Saved model to {filepath}")


def load_model(filepath: Path) -> Any:
    """Load model from disk."""
    model = joblib.load(filepath)
    logger.info(f"Loaded model from {filepath}")
    return model


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    
    # Create sample data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y, name='target')
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train single model
    model, metrics, run_id = train_model(
        model_name='random_forest',
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    
    print(f"Metrics: {metrics}")
