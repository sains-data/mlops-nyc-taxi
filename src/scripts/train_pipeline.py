"""
Training Pipeline Script
CLI tool for training models (replacement for notebooks/03_modeling.ipynb)

Usage:
    # Train single model
    python src/scripts/train_pipeline.py --model random_forest
    
    # Train with specific data
    python src/scripts/train_pipeline.py --model xgboost --data-dir data/processed
    
    # Train and register to MLflow Registry
    python src/scripts/train_pipeline.py --model random_forest --register --stage Staging
    
    # Compare all models
    python src/scripts/train_pipeline.py --compare-all
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.train import train_model, train_all_models
from src.models.registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_dir: Path):
    """
    Load training and validation data.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    logger.info(f"Loading data from: {data_dir}")
    
    # Load training data
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    train_df = pd.read_parquet(train_path)
    logger.info(f"Loaded training data: {train_df.shape}")
    
    # Load validation data
    val_path = data_dir / "val.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    val_df = pd.read_parquet(val_path)
    logger.info(f"Loaded validation data: {val_df.shape}")
    
    # Separate features and target
    X_train = train_df.drop('fare_amount', axis=1)
    y_train = train_df['fare_amount']
    
    X_val = val_df.drop('fare_amount', axis=1)
    y_val = val_df['fare_amount']
    
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")
    
    return X_train, y_train, X_val, y_val


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML file."""
    if config_path and config_path.exists():
        logger.info(f"Loading config from: {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def train_single_model(args):
    """
    Train a single model.
    
    Args:
        args: Command line arguments
    """
    logger.info("=" * 60)
    logger.info(f"TRAINING MODEL: {args.model}")
    logger.info("=" * 60)
    
    # Load data
    X_train, y_train, X_val, y_val = load_data(Path(args.data_dir))
    
    # Load config if provided
    config = load_config(args.config) if args.config else {}
    params = config.get('params', {})
    
    # Train model
    logger.info(f"Training {args.model} model...")
    model, metrics, run_id = train_model(
        model_name=args.model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        experiment_name=args.experiment,
        tracking_uri=args.mlflow_uri,
        **params
    )
    
    # Display metrics
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)
    for metric, value in metrics.items():
        logger.info(f"{metric.upper():15s}: {value:.4f}")
    logger.info(f"{'MLflow Run ID':15s}: {run_id}")
    logger.info("=" * 60)
    
    # Register to Model Registry if requested
    if args.register:
        logger.info("\nRegistering model to MLflow Model Registry...")
        registry = ModelRegistry(model_name=args.registry_name)
        
        tags = {
            'algorithm': args.model,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2']
        }
        
        if args.blue_green:
            tags['blue_green'] = args.blue_green
        
        version = registry.register_model(
            run_id=run_id,
            stage=args.stage,
            tags=tags,
            description=f"Model: {args.model}, MAE: {metrics['mae']:.4f}"
        )
        
        logger.info(f"✅ Model registered as version: {version}")
        logger.info(f"✅ Stage: {args.stage}")
    
    return model, metrics, run_id


def compare_models(args):
    """
    Train and compare all available models.
    
    Args:
        args: Command line arguments
    """
    logger.info("=" * 60)
    logger.info("COMPARING ALL MODELS")
    logger.info("=" * 60)
    
    # Load data
    X_train, y_train, X_val, y_val = load_data(Path(args.data_dir))
    
    # Train all models
    logger.info("Training 5 models...")
    results = train_all_models(
        X_train, y_train, X_val, y_val,
        experiment_name=args.experiment,
        tracking_uri=args.mlflow_uri
    )
    
    # Display comparison
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 60)
    
    # Print table
    print(results.to_string(index=False))
    
    # Best model
    best_model = results.iloc[0]
    logger.info("\n" + "=" * 60)
    logger.info(f"🏆 BEST MODEL: {best_model['model']}")
    logger.info(f"   MAE:  ${best_model['mae']:.2f}")
    logger.info(f"   RMSE: ${best_model['rmse']:.2f}")
    logger.info(f"   R²:   {best_model['r2']:.4f}")
    logger.info("=" * 60)
    
    # Save comparison to CSV
    output_path = Path("models/model_comparison_script.csv")
    results.to_csv(output_path, index=False)
    logger.info(f"\n✅ Comparison saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Training Pipeline - Train ML models via CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Random Forest (MLflow auto uses ./mlruns)
  python src/scripts/train_pipeline.py --model random_forest
  
  # Train XGBoost and register to MLflow
  python src/scripts/train_pipeline.py --model gradient_boosting --register --stage Staging
  
  # Compare all models
  python src/scripts/train_pipeline.py --compare-all
  
  # Train with custom MLflow server
  python src/scripts/train_pipeline.py --model random_forest --mlflow-uri http://localhost:5000
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Train and compare all available models'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['linear_regression', 'ridge', 'lasso', 'random_forest', 'gradient_boosting'],
        help='Model to train (default: random_forest)'
    )
    
    # Data paths
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data (default: data/processed)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to training config YAML file'
    )
    
    # MLflow settings
    parser.add_argument(
        '--experiment',
        type=str,
        default='nyc-taxi-fare',
        help='MLflow experiment name (default: nyc-taxi-fare)'
    )
    
    parser.add_argument(
        '--mlflow-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: None = auto use ./mlruns local directory)'
    )
    
    # Model Registry settings
    parser.add_argument(
        '--register',
        action='store_true',
        help='Register model to MLflow Model Registry'
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
    
    parser.add_argument(
        '--blue-green',
        type=str,
        choices=['blue', 'green'],
        help='Tag model as blue (production) or green (candidate)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare_all:
            # Compare all models
            results = compare_models(args)
        else:
            # Train single model
            model, metrics, run_id = train_single_model(args)
        
        logger.info("\n✅ Training pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
