"""
Model Evaluation Module
Comprehensive evaluation metrics and analysis
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'median_ae': np.median(np.abs(y_true - y_pred)),
        'max_error': np.max(np.abs(y_true - y_pred)),
        'explained_variance': 1 - np.var(y_true - y_pred) / np.var(y_true)
    }


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "test"
) -> Dict[str, float]:
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: Trained model with predict method
        X: Features
        y: Target values
        dataset_name: Name of dataset (for logging)
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    metrics = calculate_regression_metrics(y.values, y_pred)
    
    logger.info(f"\n{dataset_name.upper()} SET METRICS:")
    logger.info(f"  RMSE: ${metrics['rmse']:.2f}")
    logger.info(f"  MAE:  ${metrics['mae']:.2f}")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    
    return metrics


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compare multiple models on test data.
    
    Args:
        models: Dictionary of {model_name: trained_model}
        X_test: Test features
        y_test: Test target
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, dataset_name=name)
        results.append({'model': name, **metrics})
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    
    return results_df


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create scatter plot of predictions vs actual values.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Fare ($)', fontsize=12)
    ax.set_ylabel('Predicted Fare ($)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create residual analysis plots.
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Fare ($)')
    axes[0].set_ylabel('Residual ($)')
    axes[0].set_title('Residuals vs Predicted')
    
    # Residual Distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Residual ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    
    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot feature importance for tree-based models.
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return None
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(
        range(len(indices)),
        importance[indices],
        align='center'
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig


def generate_evaluation_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path
) -> Dict:
    """
    Generate comprehensive evaluation report with metrics and plots.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with all metrics and plot paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    metrics = calculate_regression_metrics(y_test.values, y_pred)
    
    # Generate plots
    plot_paths = {}
    
    # Predictions vs Actual
    pred_path = output_dir / f"{model_name}_predictions_vs_actual.png"
    plot_predictions_vs_actual(y_test.values, y_pred, save_path=pred_path)
    plot_paths['predictions_vs_actual'] = str(pred_path)
    
    # Residuals
    resid_path = output_dir / f"{model_name}_residuals.png"
    plot_residuals(y_test.values, y_pred, save_path=resid_path)
    plot_paths['residuals'] = str(resid_path)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        fi_path = output_dir / f"{model_name}_feature_importance.png"
        plot_feature_importance(
            model, 
            list(X_test.columns),
            save_path=fi_path
        )
        plot_paths['feature_importance'] = str(fi_path)
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / f"{model_name}_metrics.csv", index=False)
    
    logger.info(f"\nEvaluation report saved to {output_dir}")
    
    return {
        'metrics': metrics,
        'plots': plot_paths,
        'model_name': model_name
    }


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print(f"\nMetrics: {metrics}")
