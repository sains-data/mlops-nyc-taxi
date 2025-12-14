"""
Hyperparameter Tuning Script
CLI tool for automated hyperparameter optimization using Optuna

Usage:
    # Tune Random Forest
    python src/scripts/hyperparam_tuning.py --model random_forest --trials 50
    
    # Tune with specific metric
    python src/scripts/hyperparam_tuning.py --model gradient_boosting --trials 100 --metric mae
    
    # Tune and save best config
    python src/scripts/hyperparam_tuning.py --model random_forest --trials 50 --save-config
    
    # Tune and auto-register best model
    python src/scripts/hyperparam_tuning.py --model random_forest --trials 50 --register
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml
import logging
import optuna
from typing import Dict, Any
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.train import train_model, get_model, calculate_metrics
from src.models.registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(data_dir: Path):
    """Load training and validation data."""
    logger.info(f"Loading data from: {data_dir}")
    
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    logger.info(f"Training samples: {len(train_df):,}")
    logger.info(f"Validation samples: {len(val_df):,}")
    
    X_train = train_df.drop('fare_amount', axis=1)
    y_train = train_df['fare_amount']
    X_val = val_df.drop('fare_amount', axis=1)
    y_val = val_df['fare_amount']
    
    return X_train, y_train, X_val, y_val


def get_search_space(model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define hyperparameter search space for each model.
    
    Args:
        model_name: Name of the model
        trial: Optuna trial object
        
    Returns:
        Dictionary of hyperparameters
    """
    if model_name == 'random_forest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'n_jobs': -1,
            'random_state': 42
        }
    
    elif model_name == 'gradient_boosting':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
    
    elif model_name == 'ridge':
        return {
            'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
            'random_state': 42
        }
    
    elif model_name == 'lasso':
        return {
            'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
            'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
            'random_state': 42
        }
    
    elif model_name == 'linear_regression':
        # Linear regression has no hyperparameters to tune
        return {}
    
    else:
        raise ValueError(f"Hyperparameter tuning not implemented for: {model_name}")


def objective(
    trial: optuna.Trial,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = 'rmse'
) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        model_name: Model to tune
        X_train, y_train: Training data
        X_val, y_val: Validation data
        metric: Metric to optimize (rmse, mae, r2)
        
    Returns:
        Metric value to minimize (or negative for maximization)
    """
    # Get hyperparameters for this trial
    params = get_search_space(model_name, trial)
    
    # Train model with these params
    model = get_model(model_name, params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    metrics = calculate_metrics(y_val.values, y_pred)
    
    # Return metric to optimize
    if metric == 'r2':
        # R2 should be maximized, so return negative
        return -metrics['r2']
    else:
        # RMSE and MAE should be minimized
        return metrics[metric]


def tune_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    metric: str = 'rmse',
    study_name: str = None
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning using Optuna.
    
    Args:
        model_name: Model to tune
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of trials
        metric: Metric to optimize
        study_name: Optuna study name
        
    Returns:
        Dictionary with best params and metrics
    """
    logger.info("=" * 70)
    logger.info(f"HYPERPARAMETER TUNING: {model_name}")
    logger.info("=" * 70)
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Optimizing: {metric.upper()}")
    logger.info("=" * 70)
    
    # Create study
    direction = 'maximize' if metric == 'r2' else 'minimize'
    study = optuna.create_study(
        direction=direction,
        study_name=study_name or f"tune_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, model_name, X_train, y_train, X_val, y_val, metric
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    
    # Train final model with best params
    logger.info("\n" + "=" * 70)
    logger.info("BEST HYPERPARAMETERS FOUND")
    logger.info("=" * 70)
    for param, value in best_params.items():
        logger.info(f"{param:20s}: {value}")
    
    # Evaluate best model
    logger.info("\nTraining final model with best parameters...")
    model = get_model(model_name, best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = calculate_metrics(y_val.values, y_pred)
    
    logger.info("\n" + "=" * 70)
    logger.info("BEST MODEL METRICS")
    logger.info("=" * 70)
    logger.info(f"{'RMSE':15s}: ${metrics['rmse']:.4f}")
    logger.info(f"{'MAE':15s}: ${metrics['mae']:.4f}")
    logger.info(f"{'R²':15s}: {metrics['r2']:.4f}")
    logger.info(f"{'MAPE':15s}: {metrics['mape']:.2f}%")
    logger.info("=" * 70)
    
    return {
        'best_params': best_params,
        'best_value': abs(best_value) if metric == 'r2' else best_value,
        'metrics': metrics,
        'model': model,
        'study': study
    }


def save_config(model_name: str, params: Dict, metrics: Dict, config_dir: Path):
    """Save best hyperparameters to YAML config file."""
    config_path = config_dir / f"best_params_{model_name}.yaml"
    
    config = {
        'model': model_name,
        'best_params': params,
        'validation_metrics': {
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
            'r2': float(metrics['r2']),
            'mape': float(metrics['mape'])
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"\n✅ Best config saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning - Automated optimization with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune Random Forest with 50 trials
  python src/scripts/hyperparam_tuning.py --model random_forest --trials 50
  
  # Tune Gradient Boosting optimizing MAE
  python src/scripts/hyperparam_tuning.py --model gradient_boosting --trials 100 --metric mae
  
  # Tune and save best config
  python src/scripts/hyperparam_tuning.py --model random_forest --trials 50 --save-config
  
  # Tune and auto-register best model as GREEN candidate
  python src/scripts/hyperparam_tuning.py --model random_forest --trials 30 --register --stage Staging
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['linear_regression', 'ridge', 'lasso', 'random_forest', 'gradient_boosting'],
        help='Model to tune'
    )
    
    # Tuning parameters
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of Optuna trials (default: 50)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='rmse',
        choices=['rmse', 'mae', 'r2'],
        help='Metric to optimize (default: rmse)'
    )
    
    # Data paths
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data (default: data/processed)'
    )
    
    # Output options
    parser.add_argument(
        '--save-config',
        action='store_true',
        help='Save best hyperparameters to config/best_params_<model>.yaml'
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Directory to save config files (default: config)'
    )
    
    # MLflow settings
    parser.add_argument(
        '--experiment',
        type=str,
        default='nyc-taxi-hyperparam-tuning',
        help='MLflow experiment name (default: nyc-taxi-hyperparam-tuning)'
    )
    
    parser.add_argument(
        '--mlflow-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: None = use ./mlruns)'
    )
    
    # Model Registry settings
    parser.add_argument(
        '--register',
        action='store_true',
        help='Register best model to MLflow Model Registry'
    )
    
    parser.add_argument(
        '--registry-name',
        type=str,
        default='nyc-taxi-fare',
        help='Model name in registry (default: nyc-taxi-fare)'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        default='Staging',
        choices=['None', 'Staging', 'Production', 'Archived'],
        help='Model stage in registry (default: Staging)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        X_train, y_train, X_val, y_val = load_data(Path(args.data_dir))
        
        # Run tuning
        results = tune_hyperparameters(
            model_name=args.model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=args.trials,
            metric=args.metric,
            study_name=f"{args.model}_tuning"
        )
        
        # Save config if requested
        if args.save_config:
            config_dir = Path(args.config_dir)
            config_dir.mkdir(exist_ok=True)
            save_config(
                args.model,
                results['best_params'],
                results['metrics'],
                config_dir
            )
        
        # Train final model with MLflow tracking if registering
        if args.register:
            logger.info("\n" + "=" * 70)
            logger.info("TRAINING FINAL MODEL WITH MLFLOW TRACKING")
            logger.info("=" * 70)
            
            model, metrics, run_id = train_model(
                model_name=args.model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                params=results['best_params'],
                experiment_name=args.experiment,
                tracking_uri=args.mlflow_uri
            )
            
            logger.info(f"\nMLflow Run ID: {run_id}")
            logger.info("\nRegistering model to MLflow Model Registry...")
            
            registry = ModelRegistry(model_name=args.registry_name)
            
            tags = {
                'algorithm': args.model,
                'tuned': 'true',
                'trials': str(args.trials),
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2']
            }
            
            version = registry.register_model(
                run_id=run_id,
                stage=args.stage,
                tags=tags,
                description=f"Tuned {args.model} ({args.trials} trials), MAE: {metrics['mae']:.4f}"
            )
            
            logger.info(f"✅ Model registered as version: {version}")
            logger.info(f"✅ Stage: {args.stage}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
