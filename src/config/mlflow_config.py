"""
Centralized MLflow Configuration

All MLflow tracking URIs and settings should be imported from here
to ensure consistency across all components.
"""

from pathlib import Path
import os

# Get project root (4 levels up from this file: src/config/mlflow_config.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ==================== MLflow Configuration ====================

# SQLite database path for local development
MLFLOW_DB_PATH = PROJECT_ROOT / "mlruns" / "mlflow.db"

# Default tracking URI - SQLite for local development
# Override with environment variable MLFLOW_TRACKING_URI if set
DEFAULT_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH}"
)

# Artifact location (for model files)
MLFLOW_ARTIFACTS_PATH = PROJECT_ROOT / "mlruns"

# ==================== Experiment Names ====================

EXPERIMENT_TRAINING = "nyc-taxi-fare"
EXPERIMENT_TUNING = "nyc-taxi-fare-tuning"
EXPERIMENT_PRODUCTION = "production-models"

# ==================== Registry Names ====================

MODEL_REGISTRY_NAME = "nyc-taxi-fare"

# ==================== Model Paths ====================

PRODUCTION_MODEL_PATH = PROJECT_ROOT / "models" / "production_model.joblib"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def get_tracking_uri() -> str:
    """
    Get the MLflow tracking URI.
    
    Priority:
    1. MLFLOW_TRACKING_URI environment variable
    2. Default SQLite database in mlruns/mlflow.db
    
    Returns:
        str: MLflow tracking URI
    """
    return DEFAULT_TRACKING_URI


def setup_mlflow():
    """
    Setup MLflow with the correct tracking URI.
    Call this at the start of any script that uses MLflow.
    """
    import mlflow
    mlflow.set_tracking_uri(get_tracking_uri())
    return get_tracking_uri()


# ==================== Convenience Info ====================

if __name__ == "__main__":
    print("MLflow Configuration")
    print("=" * 50)
    print(f"Project Root:     {PROJECT_ROOT}")
    print(f"Tracking URI:     {DEFAULT_TRACKING_URI}")
    print(f"DB Path:          {MLFLOW_DB_PATH}")
    print(f"Artifacts Path:   {MLFLOW_ARTIFACTS_PATH}")
    print(f"Production Model: {PRODUCTION_MODEL_PATH}")
