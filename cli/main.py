"""
CLI Module for NYC Taxi MLOps Project
Command-line interface using Typer
"""

import logging
from pathlib import Path
from typing import Optional, List
import sys

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup
app = typer.Typer(
    name="nyc-taxi-mlops",
    help="NYC Taxi Fare Prediction - MLOps CLI",
    add_completion=False
)
console = Console()

# Sub-commands
data_app = typer.Typer(help="Data operations")
train_app = typer.Typer(help="Model training")
serve_app = typer.Typer(help="Model serving")
monitor_app = typer.Typer(help="Monitoring")
registry_app = typer.Typer(help="Model Registry operations")
model_app = typer.Typer(help="Model testing and comparison")

app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(serve_app, name="serve")
app.add_typer(monitor_app, name="monitor")
app.add_typer(registry_app, name="registry")
app.add_typer(model_app, name="model")


# ========== DATA COMMANDS ==========

@data_app.command("download")
def download_data(
    year: int = typer.Option(2024, "--year", "-y", help="Year of data"),
    months: str = typer.Option("1,2,3", "--months", "-m", help="Comma-separated months"),
    output_dir: Path = typer.Option(Path("data/raw"), "--output", "-o", help="Output directory")
):
    """Download NYC Taxi data from TLC website."""
    from src.data.ingestion import download_nyc_taxi_data
    
    month_list = [int(m) for m in months.split(",")]
    
    console.print(f"[bold blue]Downloading NYC Taxi data for {year}, months: {month_list}[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Downloading...", total=None)
        files = download_nyc_taxi_data(year=year, months=month_list, data_dir=output_dir)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓ Downloaded {len(files)} files to {output_dir}[/]")


@data_app.command("preprocess")
def preprocess_data(
    input_dir: Path = typer.Option(Path("data/raw"), "--input", "-i", help="Input directory"),
    output_dir: Path = typer.Option(Path("data/processed"), "--output", "-o", help="Output directory"),
    sample_size: Optional[int] = typer.Option(None, "--sample", "-s", help="Sample size (for testing)")
):
    """Preprocess raw data for training."""
    from src.data.ingestion import load_multiple_parquet
    from src.data.preprocessing import clean_taxi_data, engineer_features, prepare_features_target, split_data, save_processed_data
    
    console.print("[bold blue]Preprocessing data...[/]")
    
    # Load all parquet files
    parquet_files = list(input_dir.glob("*.parquet"))
    if not parquet_files:
        console.print(f"[red]No parquet files found in {input_dir}[/]")
        raise typer.Exit(1)
    
    df = load_multiple_parquet(parquet_files)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        console.print(f"[yellow]Sampled {len(df):,} rows[/]")
    
    # Process
    df_clean = clean_taxi_data(df)
    df_features = engineer_features(df_clean)
    X, y = prepare_features_target(df_features)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Save
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir)
    
    console.print(f"[green]✓ Processed data saved to {output_dir}[/]")


# ========== TRAINING COMMANDS ==========

@train_app.command("quick")
def train_model(
    model_name: str = typer.Option("random_forest", "--model", "-m", help="Model to train"),
    data_dir: Path = typer.Option(Path("data/processed"), "--data", "-d", help="Data directory"),
    experiment_name: str = typer.Option("nyc-taxi-fare", "--experiment", "-e", help="MLFlow experiment name"),
    tracking_uri: str = typer.Option("http://localhost:5000", "--tracking-uri", help="MLFlow tracking URI")
):
    """Quick training for exploration (no registry)."""
    import pandas as pd
    from src.models.train import train_model as run_training
    
    console.print(f"[bold blue]Training {model_name}...[/]")
    
    # Load data
    X_train = pd.read_parquet(data_dir / "train.parquet")
    y_train = X_train.pop("fare_amount")
    X_val = pd.read_parquet(data_dir / "val.parquet")
    y_val = X_val.pop("fare_amount")
    
    # Train
    model, metrics, run_id = run_training(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    )
    
    # Display results
    table = Table(title=f"Training Results - {model_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for metric, value in metrics.items():
        table.add_row(metric.upper(), f"{value:.4f}")
    
    console.print(table)
    console.print(f"[green]✓ Run ID: {run_id}[/]")


@train_app.command("compare-algos")
def compare_models(
    data_dir: Path = typer.Option(Path("data/processed"), "--data", "-d", help="Data directory"),
    experiment_name: str = typer.Option("nyc-taxi-fare", "--experiment", "-e", help="MLFlow experiment name")
):
    """Train and compare all available algorithms."""
    import pandas as pd
    from src.models.train import train_all_models
    
    console.print("[bold blue]Training and comparing all models...[/]")
    
    # Load data
    X_train = pd.read_parquet(data_dir / "train.parquet")
    y_train = X_train.pop("fare_amount")
    X_val = pd.read_parquet(data_dir / "val.parquet")
    y_val = X_val.pop("fare_amount")
    
    # Train all
    results = train_all_models(X_train, y_train, X_val, y_val, experiment_name=experiment_name)
    
    # Display results
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("RMSE", style="yellow")
    table.add_column("MAE", style="yellow")
    table.add_column("R²", style="green")
    
    for _, row in results.iterrows():
        table.add_row(
            row['model'],
            f"${row['rmse']:.2f}",
            f"${row['mae']:.2f}",
            f"{row['r2']:.4f}"
        )
    
    console.print(table)
    console.print(f"[green]✓ Best model: {results.iloc[0]['model']}[/]")


@train_app.command("pipeline")
def train_pipeline(
    model: str = typer.Option("random_forest", "--model", "-m", help="Model algorithm"),
    data_dir: Path = typer.Option(Path("data/processed"), "--data", "-d", help="Data directory"),
    experiment: str = typer.Option("nyc-taxi-fare", "--experiment", "-e", help="MLflow experiment"),
    register: bool = typer.Option(False, "--register", help="Register to Model Registry"),
    stage: str = typer.Option("Staging", "--stage", help="Registry stage (Production/Staging)"),
    blue_green: Optional[str] = typer.Option(None, "--blue-green", help="Tag as 'blue' or 'green'")
):
    """Production training pipeline with registry integration."""
    import subprocess
    
    console.print(f"[bold blue]Running training pipeline: {model}[/]")
    
    cmd = [
        "python", "src/scripts/train_pipeline.py",
        "--model", model,
        "--data-dir", str(data_dir),
        "--experiment", experiment
    ]
    
    if register:
        cmd.extend(["--register", "--stage", stage])
        if blue_green:
            cmd.extend(["--blue-green", blue_green])
    
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode == 0:
        console.print(f"[green]✓ Training pipeline completed[/]")
    else:
        console.print(f"[red]✗ Training pipeline failed[/]")
        raise typer.Exit(1)


@train_app.command("tune")
def tune_hyperparameters(
    model: str = typer.Option("random_forest", "--model", "-m", help="Model algorithm"),
    trials: int = typer.Option(50, "--trials", "-t", help="Number of Optuna trials"),
    data_dir: Path = typer.Option(Path("data/processed"), "--data", "-d", help="Data directory"),
    metric: str = typer.Option("rmse", "--metric", help="Optimization metric (rmse/mae/r2)"),
    save_config: bool = typer.Option(False, "--save-config", help="Save best params to YAML"),
    register: bool = typer.Option(False, "--register", help="Train and register best model"),
    stage: str = typer.Option("Staging", "--stage", help="Registry stage if registering")
):
    """Hyperparameter tuning with Optuna."""
    import subprocess
    
    console.print(f"[bold blue]Tuning {model} with {trials} trials...[/]")
    
    cmd = [
        "python", "src/scripts/hyperparam_tuning.py",
        "--model", model,
        "--trials", str(trials),
        "--data-dir", str(data_dir),
        "--metric", metric
    ]
    
    if save_config:
        cmd.append("--save-config")
    if register:
        cmd.extend(["--register", "--stage", stage])
    
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode == 0:
        console.print(f"[green]✓ Hyperparameter tuning completed[/]")
    else:
        console.print(f"[red]✗ Tuning failed[/]")
        raise typer.Exit(1)


# ========== REGISTRY COMMANDS ==========

@registry_app.command("runs")
def list_runs(
    experiment: str = typer.Option("nyc-taxi-fare", "--experiment", "-e", help="Experiment name")
):
    """List available MLflow runs (to find run_id for registration)."""
    import mlflow
    from src.config.mlflow_config import get_tracking_uri
    
    # Use centralized tracking URI for consistency
    mlflow.set_tracking_uri(get_tracking_uri())
    
    console.print(f"[bold blue]Runs in experiment: {experiment}[/]")
    
    try:
        runs = mlflow.search_runs(experiment_names=[experiment])
        
        if runs.empty:
            console.print(f"[yellow]No runs found in experiment '{experiment}'[/]")
            return
        
        table = Table(title=f"MLflow Runs - {experiment}")
        table.add_column("Run ID", style="cyan")
        table.add_column("Model", style="yellow")
        table.add_column("RMSE", style="green")
        table.add_column("MAE", style="green")
        table.add_column("R²", style="green")
        
        for _, row in runs.sort_values('metrics.val_rmse').iterrows():
            rmse = row.get('metrics.val_rmse', 'N/A')
            mae = row.get('metrics.val_mae', 'N/A')
            r2 = row.get('metrics.val_r2', 'N/A')
            model = row.get('params.model_name', 'N/A')
            
            try:
                rmse_str = f"${float(rmse):.4f}" if rmse != 'N/A' else 'N/A'
            except (ValueError, TypeError):
                rmse_str = 'N/A'
            
            try:
                mae_str = f"${float(mae):.4f}" if mae != 'N/A' else 'N/A'
            except (ValueError, TypeError):
                mae_str = 'N/A'
                
            try:
                r2_str = f"{float(r2):.4f}" if r2 != 'N/A' else 'N/A'
            except (ValueError, TypeError):
                r2_str = 'N/A'
            
            table.add_row(row['run_id'][:16] + "...", model, rmse_str, mae_str, r2_str)
        
        console.print(table)
        console.print(f"\n[cyan]Tip: Use full run_id with 'registry register <run_id>'[/]")
        
    except Exception as e:
        console.print(f"[red]Error listing runs: {e}[/]")
        raise typer.Exit(1)


@registry_app.command("list")
def list_versions(
    model_name: str = typer.Option("nyc-taxi-fare", "--model", "-m", help="Model name")
):
    """List all model versions in registry."""
    from src.models.registry import ModelRegistry
    
    console.print(f"[bold blue]Model versions for: {model_name}[/]")
    
    try:
        registry = ModelRegistry(model_name)
        versions = registry.list_all_versions()
    except Exception as e:
        console.print(f"[red]Error initializing registry: {e}[/]")
        raise typer.Exit(1)
    
    if not versions:
        console.print(f"[yellow]No versions found for {model_name}[/]")
        console.print(f"[cyan]Tip: Register a model first using 'train pipeline --register'[/]")
        return
    
    table = Table(title=f"Model Registry: {model_name}")
    table.add_column("Version", style="cyan")
    table.add_column("Stage", style="yellow")
    table.add_column("Algorithm", style="magenta")
    table.add_column("MAE", style="green")
    table.add_column("R²", style="green")
    table.add_column("Blue/Green", style="blue")
    
    for v in sorted(versions, key=lambda x: x['version'], reverse=True):
        mae = v['tags'].get('mae', v['tags'].get('val_mae', 'N/A'))
        r2 = v['tags'].get('r2', v['tags'].get('val_r2', 'N/A'))
        
        # Format metrics with error handling
        try:
            mae_str = f"${float(mae):.4f}" if mae != 'N/A' else '$N/A'
        except (ValueError, TypeError):
            mae_str = '$N/A'
        
        try:
            r2_str = f"{float(r2):.4f}" if r2 != 'N/A' else 'N/A'
        except (ValueError, TypeError):
            r2_str = 'N/A'
        
        # Determine Blue/Green status
        blue_green = '-'
        if v['stage'] == 'Production':
            blue_green = 'BLUE'
        elif v['stage'] == 'Staging':
            blue_green = 'GREEN'

        table.add_row(
            str(v['version']),
            v['stage'],
            v['tags'].get('algorithm', 'N/A'),
            mae_str,
            r2_str,
            blue_green
        )
    
    console.print(table)


@registry_app.command("status")
def deployment_status(
    model_name: str = typer.Option("nyc-taxi-fare", "--model", "-m", help="Model name")
):
    """Show current BLUE/GREEN deployment status."""
    from src.models.registry import ModelRegistry
    
    try:
        registry = ModelRegistry(model_name)
    except Exception as e:
        console.print(f"[red]Error initializing registry: {e}[/]")
        raise typer.Exit(1)
    
    try:
        prod = registry.get_production_model()
        staging = registry.get_staging_model()
    except Exception as e:
        console.print(f"[red]Error retrieving models: {e}[/]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Deployment Status: {model_name}[/]\n")
    
    if prod:
        mae = prod.tags.get('mae', prod.tags.get('val_mae', 'N/A'))
        r2 = prod.tags.get('r2', prod.tags.get('val_r2', 'N/A'))
        
        try:
            mae_str = f"${float(mae):.4f}" if mae != 'N/A' else '$N/A'
        except (ValueError, TypeError):
            mae_str = '$N/A'
        
        try:
            r2_str = f"{float(r2):.4f}" if r2 != 'N/A' else 'N/A'
        except (ValueError, TypeError):
            r2_str = 'N/A'
        
        console.print(f"[bold blue]🔵 BLUE (Production):[/]")
        console.print(f"  Version: v{prod.version}")
        console.print(f"  Algorithm: {prod.tags.get('algorithm', 'N/A')}")
        console.print(f"  MAE: {mae_str}")
        console.print(f"  R²: {r2_str}\n")
    else:
        console.print("[yellow]🔵 BLUE: No production model[/]\n")
    
    if staging:
        mae = staging.tags.get('mae', staging.tags.get('val_mae', 'N/A'))
        r2 = staging.tags.get('r2', staging.tags.get('val_r2', 'N/A'))
        
        try:
            mae_str = f"${float(mae):.4f}" if mae != 'N/A' else '$N/A'
        except (ValueError, TypeError):
            mae_str = '$N/A'
        
        try:
            r2_str = f"{float(r2):.4f}" if r2 != 'N/A' else 'N/A'
        except (ValueError, TypeError):
            r2_str = 'N/A'
        
        console.print(f"[bold green]🟢 GREEN (Staging):[/]")
        console.print(f"  Version: v{staging.version}")
        console.print(f"  Algorithm: {staging.tags.get('algorithm', 'N/A')}")
        console.print(f"  MAE: {mae_str}")
        console.print(f"  R²: {r2_str}")
    else:
        console.print("[yellow]🟢 GREEN: No staging model[/]")


@registry_app.command("promote")
def promote_model(
    version: int = typer.Argument(..., help="Version number to promote"),
    model_name: str = typer.Option("nyc-taxi-fare", "--model", "-m", help="Model name"),
    archive_current: bool = typer.Option(True, "--archive/--no-archive", help="Archive current production model")
):
    """Promote model to production (GREEN → BLUE)."""
    from src.models.registry import ModelRegistry
    
    console.print(f"[bold blue]Promoting v{version} to production...[/]")
    
    registry = ModelRegistry(model_name)
    
    # Show current production
    prod = registry.get_production_model()
    if prod:
        console.print(f"[yellow]Current production: v{prod.version}[/]")
    
    # Confirm
    if not typer.confirm(f"Promote v{version} to BLUE (Production)?"):
        console.print("[red]Cancelled[/]")
        raise typer.Exit(0)
    
    # Promote
    registry.promote_to_production(version=str(version), archive_current=archive_current)
    
    console.print(f"[green]✓ v{version} promoted to production[/]")
    if archive_current:
        console.print(f"[yellow]Previous production archived[/]")


@registry_app.command("rollback")
def rollback_model(
    version: int = typer.Argument(..., help="Version number to rollback to"),
    model_name: str = typer.Option("nyc-taxi-fare", "--model", "-m", help="Model name")
):
    """Rollback production to a previous version."""
    from src.models.registry import ModelRegistry
    
    console.print(f"[bold yellow]Rolling back to v{version}...[/]")
    
    registry = ModelRegistry(model_name)
    
    # Confirm
    if not typer.confirm(f"⚠️  Rollback production to v{version}?"):
        console.print("[red]Cancelled[/]")
        raise typer.Exit(0)
    
    # Rollback
    registry.rollback(version=str(version))
    
    console.print(f"[green]✓ Rolled back to v{version}[/]")


@registry_app.command("register")
def register_from_run(
    run_id: str = typer.Argument(..., help="MLflow run ID to register"),
    model_name: str = typer.Option("nyc-taxi-fare", "--model", "-m", help="Model name in registry"),
    stage: str = typer.Option("Staging", "--stage", "-s", help="Stage (Staging/Production)"),
    algorithm: str = typer.Option(None, "--algorithm", "-a", help="Algorithm name tag")
):
    """Register a model from an existing MLflow run (no retraining needed)."""
    from src.models.registry import ModelRegistry
    import mlflow
    from src.config.mlflow_config import get_tracking_uri
    
    console.print(f"[bold blue]Registering run {run_id[:8]}... to {model_name}[/]")
    
    # Connect to MLflow using centralized config
    mlflow.set_tracking_uri(get_tracking_uri())
    
    # Get run metrics
    try:
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params
    except Exception as e:
        console.print(f"[red]Error getting run info: {e}[/]")
        raise typer.Exit(1)
    
    registry = ModelRegistry(model_name)
    
    # Build tags with metrics
    tags = {
        'mae': metrics.get('val_mae', 'N/A'),
        'rmse': metrics.get('val_rmse', 'N/A'),
        'r2': metrics.get('val_r2', 'N/A'),
    }
    if algorithm:
        tags['algorithm'] = algorithm
    elif params.get('model_name'):
        tags['algorithm'] = params.get('model_name')
    
    try:
        version = registry.register_model(
            run_id=run_id,
            stage=stage,
            tags=tags,
            description=f"Registered from run: {run_id[:8]}..."
        )
        
        console.print(f"[green]✓ Registered as version: {version}[/]")
        console.print(f"[cyan]Stage: {stage}[/]")
        console.print(f"[cyan]MAE: ${tags['mae']:.4f}, RMSE: ${tags['rmse']:.4f}, R²: {tags['r2']:.4f}[/]")
        
        if stage == "Production":
            console.print(f"[yellow]Tip: Model exported to production_model.joblib[/]")
            
    except Exception as e:
        console.print(f"[red]✗ Registration failed: {e}[/]")
        raise typer.Exit(1)


# ========== MODEL TESTING COMMANDS ==========

@model_app.command("test")
def test_model(
    version: int = typer.Argument(..., help="Model version to test"),
    model_name: str = typer.Option("nyc-taxi-fare", "--model", "-m", help="Model name"),
    data_dir: Path = typer.Option(Path("data/processed"), "--data", "-d", help="Data directory"),
    dataset: str = typer.Option("val", "--dataset", help="Dataset to test on (val/test)")
):
    """Test a specific model version."""
    from src.models.registry import ModelRegistry
    import mlflow
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    console.print(f"[bold blue]Testing v{version} on {dataset} set...[/]")
    
    # Load model
    registry = ModelRegistry(model_name)
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    
    # Load data
    df = pd.read_parquet(data_dir / f"{dataset}.parquet")
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']
    
    # Predict
    y_pred = model.predict(X)
    
    # Metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    # Display
    table = Table(title=f"Model v{version} - {dataset.upper()} Set")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("MAE", f"${mae:.4f}")
    table.add_row("RMSE", f"${rmse:.4f}")
    table.add_row("R²", f"{r2:.4f}")
    table.add_row("Samples", f"{len(y):,}")
    
    console.print(table)


@model_app.command("compare")
def compare_versions(
    version1: int = typer.Argument(..., help="First version"),
    version2: int = typer.Argument(..., help="Second version"),
    model_name: str = typer.Option("nyc-taxi-fare", "--model", "-m", help="Model name"),
    data_dir: Path = typer.Option(Path("data/processed"), "--data", "-d", help="Data directory"),
    dataset: str = typer.Option("val", "--dataset", help="Dataset to compare on (val/test)")
):
    """Compare two model versions side-by-side."""
    import mlflow
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    console.print(f"[bold blue]Comparing v{version1} vs v{version2}...[/]")
    
    # Load models
    model1 = mlflow.sklearn.load_model(f"models:/{model_name}/{version1}")
    model2 = mlflow.sklearn.load_model(f"models:/{model_name}/{version2}")
    
    # Load data
    df = pd.read_parquet(data_dir / f"{dataset}.parquet")
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']
    
    # Predict
    y_pred1 = model1.predict(X)
    y_pred2 = model2.predict(X)
    
    # Metrics
    mae1 = mean_absolute_error(y, y_pred1)
    mae2 = mean_absolute_error(y, y_pred2)
    rmse1 = np.sqrt(mean_squared_error(y, y_pred1))
    rmse2 = np.sqrt(mean_squared_error(y, y_pred2))
    r2_1 = r2_score(y, y_pred1)
    r2_2 = r2_score(y, y_pred2)
    
    # Display
    table = Table(title=f"Version Comparison - {dataset.upper()} Set")
    table.add_column("Metric", style="cyan")
    table.add_column(f"v{version1}", style="yellow")
    table.add_column(f"v{version2}", style="green")
    table.add_column("Diff", style="magenta")
    
    table.add_row(
        "MAE",
        f"${mae1:.4f}",
        f"${mae2:.4f}",
        f"{((mae1-mae2)/mae1)*100:.2f}%"
    )
    table.add_row(
        "RMSE",
        f"${rmse1:.4f}",
        f"${rmse2:.4f}",
        f"{((rmse1-rmse2)/rmse1)*100:.2f}%"
    )
    table.add_row(
        "R²",
        f"{r2_1:.4f}",
        f"{r2_2:.4f}",
        f"{((r2_2-r2_1)/r2_1)*100:.2f}%"
    )
    
    console.print(table)
    
    # Winner
    if mae2 < mae1:
        console.print(f"[green]✓ v{version2} is {((mae1-mae2)/mae1)*100:.2f}% better[/]")
    else:
        console.print(f"[yellow]v{version1} is still better[/]")


# ========== SERVING COMMANDS ===========

@serve_app.command("start")
def start_server(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host address"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
    model_path: Path = typer.Option(Path("models/production_model.joblib"), "--model", "-m", help="Model path")
):
    """Start the FastAPI prediction server."""
    import os
    os.environ["MODEL_PATH"] = str(model_path)
    
    console.print(f"[bold blue]Starting API server at {host}:{port}[/]")
    console.print(f"[yellow]Model: {model_path}[/]")
    console.print(f"[green]Swagger UI: http://{host}:{port}/docs[/]")
    
    from src.serving.api import start_server
    start_server(host=host, port=port)


@serve_app.command("mlflow")
def start_mlflow_ui(
    port: int = typer.Option(5000, "--port", "-p", help="Port number for MLflow UI")
):
    """Start MLflow UI with correct tracking URI."""
    import subprocess
    from src.config.mlflow_config import get_tracking_uri
    
    tracking_uri = get_tracking_uri()
    console.print(f"[bold blue]Starting MLflow UI at port {port}[/]")
    console.print(f"[cyan]Tracking URI: {tracking_uri}[/]")
    console.print(f"[green]Open: http://localhost:{port}[/]")
    
    subprocess.run([
        "mlflow", "ui",
        "--port", str(port),
        "--backend-store-uri", tracking_uri
    ], cwd=Path.cwd())


# ========== MONITORING COMMANDS ==========

@monitor_app.command("drift")
def check_drift(
    reference_data: Path = typer.Option(..., "--reference", "-r", help="Reference data path"),
    current_data: Path = typer.Option(..., "--current", "-c", help="Current data path"),
    output_dir: Path = typer.Option(Path("reports"), "--output", "-o", help="Output directory"),
    sample_size: int = typer.Option(10000, "--sample", "-s", help="Sample size for large datasets")
):
    """Generate data drift report using Evidently."""
    import pandas as pd
    
    try:
        # Evidently 0.7+ uses Report class directly from main module
        from evidently import Report
        from evidently.metrics import DriftedColumnsCount
    except ImportError as e:
        console.print(f"[red]Error importing Evidently: {e}[/]")
        console.print("[yellow]Run: pip install evidently[/]")
        raise typer.Exit(1)
    
    console.print("[bold blue]Generating data drift report...[/]")
    
    # Load data  
    console.print(f"[cyan]Loading reference data: {reference_data}[/]")
    ref_df = pd.read_parquet(reference_data)
    console.print(f"[cyan]Loading current data: {current_data}[/]")
    cur_df = pd.read_parquet(current_data)
    
    console.print(f"[yellow]Reference shape: {ref_df.shape}, Current shape: {cur_df.shape}[/]")
    
    # Sample for large datasets
    if len(ref_df) > sample_size:
        console.print(f"[yellow]Sampling reference data to {sample_size} rows...[/]")
        ref_df = ref_df.sample(n=sample_size, random_state=42)
    if len(cur_df) > sample_size:
        console.print(f"[yellow]Sampling current data to {sample_size} rows...[/]")
        cur_df = cur_df.sample(n=sample_size, random_state=42)
    
    console.print(f"[cyan]Sampled shapes - Ref: {ref_df.shape}, Cur: {cur_df.shape}[/]")
    
    # Generate report
    console.print("[cyan]Generating drift report...[/]")
    report = Report(metrics=[
        DriftedColumnsCount(),
    ])
    report.run(reference_data=ref_df, current_data=cur_df)
    
    # Display results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[green]✓ Drift analysis complete![/]")
    console.print(f"[cyan]Drift metrics calculated for {len(ref_df.columns)} columns[/]")
    console.print("[yellow]Note: For full HTML reports, use the Dashboard (Tab 3 - Monitoring)[/]")
    console.print("[cyan]Or run: streamlit run src/mlops_dashboard.py[/]")


# ========== MAIN ==========

@app.command("version")
def version():
    """Show version information."""
    console.print("[bold]NYC Taxi MLOps Project[/]")
    console.print("Version: 1.0.0")


@app.callback()
def main():
    """
    NYC Taxi Fare Prediction - MLOps CLI
    
    A complete MLOps pipeline for predicting taxi fares in New York City.
    """
    pass


if __name__ == "__main__":
    app()
