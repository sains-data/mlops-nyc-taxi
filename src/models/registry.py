"""
MLflow Model Registry Helper
Handles model versioning, promotion, and lifecycle management
"""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, List
import logging
import joblib
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manage model versions and lifecycle stages in MLflow Model Registry.
    
    Stages:
    - None: Just registered
    - Staging: Testing phase (GREEN model)
    - Production: Currently serving (BLUE model)
    - Archived: Old versions (backup for rollback)
    """
    
    def __init__(self, model_name: str = "nyc-taxi-fare", tracking_uri: Optional[str] = None):
        """
        Initialize Model Registry client.
        
        Args:
            model_name: Name of the model in registry
            tracking_uri: MLflow tracking URI (default: from centralized config)
        """
        # Set tracking URI - use centralized config for consistency
        if tracking_uri is None:
            from src.config.mlflow_config import get_tracking_uri
            tracking_uri = get_tracking_uri()
        
        mlflow.set_tracking_uri(tracking_uri)
        
        try:
            self.client = MlflowClient(tracking_uri=tracking_uri)
            self.model_name = model_name
            logger.info(f"Initialized ModelRegistry for: {model_name} (tracking_uri={tracking_uri})")
        except Exception as e:
            logger.error(f"Failed to initialize ModelRegistry: {e}")
            raise
    
    def register_model(
        self, 
        run_id: str, 
        stage: str = 'Staging',
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register model from MLflow run to Model Registry.
        
        Args:
            run_id: MLflow run ID
            stage: Target stage (None, Staging, Production, Archived)
            tags: Additional tags for the model version
            description: Model description
            
        Returns:
            Model version number
            
        Example:
            >>> registry = ModelRegistry()
            >>> version = registry.register_model(
            ...     run_id='abc123',
            ...     stage='Staging',
            ...     tags={'algorithm': 'random_forest', 'blue_green': 'green'}
            ... )
            >>> print(f"Registered as version: {version}")
        """
        try:
            # Register model from run
            model_uri = f"runs:/{run_id}/model"
            
            logger.info(f"Registering model from run: {run_id}")
            mv = mlflow.register_model(model_uri, self.model_name)
            
            logger.info(f"Model registered as version: {mv.version}")
            
            # Transition to specified stage
            if stage:
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=mv.version,
                    stage=stage
                )
                logger.info(f"Transitioned version {mv.version} to stage: {stage}")
            
            # Add tags
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=self.model_name,
                        version=mv.version,
                        key=key,
                        value=str(value)
                    )
                logger.info(f"Added tags: {tags}")
            
            # Add description
            if description:
                self.client.update_model_version(
                    name=self.model_name,
                    version=mv.version,
                    description=description
                )
            
            return mv.version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def get_model_by_stage(self, stage: str = "Production"):
        """
        Get latest model version by stage.
        
        Args:
            stage: Model stage (Staging, Production, Archived)
            
        Returns:
            ModelVersion object or None
            
        Example:
            >>> registry = ModelRegistry()
            >>> prod_model = registry.get_model_by_stage("Production")
            >>> print(f"Production model: v{prod_model.version}")
        """
        try:
            versions = self.client.get_latest_versions(
                self.model_name,
                stages=[stage]
            )
            
            if versions:
                logger.info(f"Found {stage} model: version {versions[0].version}")
                return versions[0]
            else:
                logger.warning(f"No model found in {stage} stage")
                return None
                
        except Exception as e:
            logger.error(f"Error getting model by stage: {e}")
            return None
    
    def get_production_model(self):
        """Get current production model (BLUE)."""
        return self.get_model_by_stage("Production")
    
    def get_staging_model(self):
        """Get staging model (GREEN)."""
        return self.get_model_by_stage("Staging")
    
    def export_to_production_file(self, version: str, output_path: Optional[Path] = None) -> Path:
        """
        Export a model version to production_model.joblib file.
        
        Args:
            version: Model version to export
            output_path: Custom output path (default: models/production_model.joblib)
            
        Returns:
            Path to exported model file
        """
        try:
            # Get project root
            project_root = Path(__file__).parent.parent.parent
            
            if output_path is None:
                output_path = project_root / "models" / "production_model.joblib"
            
            # Get model version info
            mv = self.client.get_model_version(name=self.model_name, version=version)
            run = self.client.get_run(mv.run_id)
            
            # Load model from MLflow
            model_uri = f"models:/{self.model_name}/{version}"
            logger.info(f"Loading model from: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get feature names from run params or use default
            feature_names = [
                'trip_distance', 'passenger_count', 'trip_duration_minutes',
                'avg_speed_mph', 'pickup_hour', 'pickup_dayofweek', 'pickup_month',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'PULocationID', 'DOLocationID', 'VendorID',
                'is_weekend', 'is_rush_hour', 'same_location', 'has_tolls'
            ]
            
            # Try to get algorithm name from tags
            algorithm = mv.tags.get('algorithm', run.data.params.get('model_type', 'random_forest'))
            
            # Get metrics from MLflow run
            run_metrics = run.data.metrics
            metrics = {
                'mae': run_metrics.get('val_mae', run_metrics.get('mae', None)),
                'rmse': run_metrics.get('val_rmse', run_metrics.get('rmse', None)),
                'mse': run_metrics.get('val_mse', run_metrics.get('mse', None)),  # loss
                'mape': run_metrics.get('val_mape', run_metrics.get('mape', None)),
                'r2': run_metrics.get('val_r2', run_metrics.get('r2', None)),
            }
            # Calculate MSE from RMSE if not available
            if metrics['mse'] is None and metrics['rmse'] is not None:
                metrics['mse'] = metrics['rmse'] ** 2
            
            # Create model package (same format as api.py expects)
            model_package = {
                'model': model,
                'model_name': algorithm,
                'model_type': type(model).__name__,
                'version': f"1.0.{version}",
                'features': feature_names,
                'metrics': metrics,  # Added metrics!
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'mlflow_version': version,
                'mlflow_run_id': mv.run_id
            }
            
            # Save to file
            joblib.dump(model_package, output_path)
            logger.info(f"✅ Exported model v{version} to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise
    
    def promote_to_production(self, version: str, archive_current: bool = True, export_model: bool = True):
        """
        Promote a model version to Production stage (GREEN → BLUE).
        
        Args:
            version: Version number to promote
            archive_current: Whether to archive current production model
            export_model: Whether to export model to production_model.joblib (default: True)
            
        Example:
            >>> registry = ModelRegistry()
            >>> # Promote staging (GREEN) to production (BLUE)
            >>> registry.promote_to_production(version="2", archive_current=True)
        """
        try:
            # Archive current production if requested
            if archive_current:
                prod_model = self.get_production_model()
                if prod_model:
                    logger.info(f"Archiving current production model: v{prod_model.version}")
                    self.client.transition_model_version_stage(
                        name=self.model_name,
                        version=prod_model.version,
                        stage="Archived"
                    )
            
            # Promote new version to production
            logger.info(f"Promoting version {version} to Production")
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Production"
            )
            
            logger.info(f"✅ Successfully promoted v{version} to Production!")
            
            # Export model to production file
            if export_model:
                logger.info(f"Exporting model to production_model.joblib...")
                self.export_to_production_file(version)
            
        except Exception as e:
            logger.error(f"Error promoting to production: {e}")
            raise
    
    def rollback(self, version: str):
        """
        Rollback to a specific version (usually from Archived).
        
        Args:
            version: Version number to rollback to
            
        Example:
            >>> registry = ModelRegistry()
            >>> # Rollback to previous version
            >>> registry.rollback(version="1")
        """
        logger.info(f"Rolling back to version {version}")
        self.promote_to_production(version, archive_current=True)
        logger.info(f"✅ Rolled back to version {version}")
    
    def list_all_versions(self) -> List[Dict]:
        """
        List all registered model versions.
        
        Returns:
            List of dicts with version info
            
        Example:
            >>> registry = ModelRegistry()
            >>> versions = registry.list_all_versions()
            >>> for v in versions:
            ...     print(f"v{v['version']}: {v['stage']}")
        """
        try:
            model = self.client.get_registered_model(self.model_name)
            versions = model.latest_versions
            
            logger.info(f"Found {len(versions)} model versions")
            
            # Convert to dict for easier access
            result = []
            for v in versions:
                result.append({
                    'version': v.version,
                    'stage': v.current_stage,
                    'run_id': v.run_id,
                    'tags': v.tags if hasattr(v, 'tags') else {},
                    'description': v.description if hasattr(v, 'description') else ''
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing versions: {e}")
            return []
    
    def get_model_info(self, version: Optional[str] = None, stage: Optional[str] = None):
        """
        Get detailed info about a model version.
        
        Args:
            version: Specific version number
            stage: Or get by stage
            
        Returns:
            Dictionary with model information
        """
        try:
            if version:
                mv = self.client.get_model_version(
                    name=self.model_name,
                    version=version
                )
            elif stage:
                mv = self.get_model_by_stage(stage)
            else:
                raise ValueError("Provide either version or stage")
            
            if not mv:
                return None
            
            # Get run info for metrics
            run = self.client.get_run(mv.run_id)
            
            info = {
                'version': mv.version,
                'stage': mv.current_stage,
                'run_id': mv.run_id,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': dict(mv.tags),
                'description': mv.description,
                'creation_timestamp': mv.creation_timestamp,
                'status': mv.status
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry()
    
    # List all versions
    print("\n=== All Model Versions ===")
    versions = registry.list_all_versions()
    for v in versions:
        print(f"Version {v.version}: Stage={v.current_stage}")
    
    # Get production model info
    print("\n=== Production Model ===")
    prod_model = registry.get_production_model()
    if prod_model:
        info = registry.get_model_info(version=prod_model.version)
        if info:
            print(f"Version: {info['version']}")
            print(f"Metrics: {info['metrics']}")
