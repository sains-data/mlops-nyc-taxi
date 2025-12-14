"""
Model Serving API
FastAPI application for NYC Taxi Fare prediction
"""

import logging
import os
import datetime
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="MLOps Project - Predict taxi fares in New York City",
    version="1.0.0"
)

# ==========================================
# Pydantic Models
# ==========================================

class TripFeatures(BaseModel):
    """Input features for a single trip prediction."""
    trip_distance: float = Field(..., ge=0.1, le=100, description="Trip distance in miles")
    passenger_count: int = Field(..., ge=1, le=6, description="Number of passengers")
    PULocationID: int = Field(..., ge=1, description="Pickup location ID")
    DOLocationID: int = Field(..., ge=1, description="Dropoff location ID")
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup (0-23)")
    pickup_dayofweek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    pickup_month: int = Field(1, ge=1, le=12, description="Month (1-12)")
    is_weekend: int = Field(0, ge=0, le=1, description="Is weekend (0 or 1)")
    trip_duration_minutes: float = Field(15.0, ge=1, le=180, description="Trip duration in minutes")

    class Config:
        json_schema_extra = {
            "example": {
                "trip_distance": 2.5,
                "passenger_count": 1,
                "PULocationID": 161,
                "DOLocationID": 237,
                "pickup_hour": 14,
                "pickup_dayofweek": 2,
                "pickup_month": 6,
                "is_weekend": 0,
                "trip_duration_minutes": 15.0
            }
        }


class BatchTripFeatures(BaseModel):
    """Input features for batch prediction."""
    trips: List[TripFeatures]


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    predicted_fare: float
    currency: str = "USD"
    model_name: str
    model_version: str
    timestamp: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[float]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool

# ==========================================
# Global Variables & Configuration
# ==========================================

current_dir = Path(__file__).parent
static_dir = current_dir / "static"
static_dir.mkdir(exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

MONITORING_FILE = current_dir / "prediction_logs.json"
REFERENCE_STATS_FILE = current_dir / "reference_stats.json"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/production_model.joblib"))

model = None

# ==========================================
# Helper Functions
# ==========================================

def load_model():
    """Load model at startup."""
    global model
    
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    if MODEL_PATH.exists():
        loaded = joblib.load(MODEL_PATH)
        # Check if loaded object is a dict (new format) or model (old format)
        if isinstance(loaded, dict) and "model" in loaded:
            model = loaded
            logger.info(f"Model package loaded from {MODEL_PATH} (v{model.get('version', 'unknown')})")
        else:
            # Wrap legacy model in dict structure
            # Dynamically detect model type from class name
            model_type_name = type(loaded).__name__
            model = {
                "model": loaded,
                "model_name": model_type_name,
                "version": "1.0.0",
                "model_type": model_type_name,
                "features": getattr(loaded, 'feature_names_in_', []).tolist() if hasattr(loaded, 'feature_names_in_') else []
            }
            logger.info(f"Legacy model ({model_type_name}) loaded from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}")

def log_prediction(inputs: dict, prediction: float):
    """Log prediction for monitoring."""
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "inputs": inputs,
            "prediction": prediction
        }
        
        logs = []
        if MONITORING_FILE.exists():
            try:
                with open(MONITORING_FILE, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(log_entry)
        logs = logs[-1000:] # Keep last 1000
        
        with open(MONITORING_FILE, 'w') as f:
            json.dump(logs, f)
            
    except Exception as e:
        logger.error(f"Logging failed: {e}")

# ==========================================
# Endpoints
# ==========================================

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/", tags=["General"])
async def root():
    return FileResponse(static_dir / "index.html")

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.get("/model/info", tags=["General"])
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model.get("model_name", "nyc-taxi-fare"),
        "model_type": model.get("model_type", "unknown"),
        "version": model.get("version", "unknown"),
        "num_features": len(model.get("features", [])),
        "features": model.get("features", [])
    }

@app.get("/monitoring/drift", tags=["Monitoring"])
async def get_drift_metrics():
    """Get drift metrics."""
    # Load Reference
    ref_stats = {}
    if REFERENCE_STATS_FILE.exists():
        with open(REFERENCE_STATS_FILE, 'r') as f:
            ref_stats = json.load(f)
    
    # Load Current Logs
    current_stats = {}
    logs = []
    if MONITORING_FILE.exists():
        with open(MONITORING_FILE, 'r') as f:
            logs = json.load(f)
            
    if logs:
        # Calculate simple means for current window (last 100)
        recent_logs = logs[-100:]
        df_curr = pd.DataFrame([l['inputs'] for l in recent_logs])
        predictions = [l['prediction'] for l in recent_logs]
        
        # Distance drift (input feature)
        if 'trip_distance' in df_curr.columns:
            current_stats['trip_distance'] = {
                "mean": float(df_curr['trip_distance'].mean()),
                "count": len(df_curr)
            }
        
        # Fare/Target drift (model output)
        current_stats['fare_amount'] = {
            "mean": float(np.mean(predictions)),
            "count": len(predictions)
        }
    else:
        current_stats = {k: {"mean": 0, "count": 0} for k in ref_stats.keys()}

    return {
        "reference": ref_stats,
        "current": current_stats,
        "total_predictions": len(logs),
        "model_info": {
            "name": model.get("model_name", "unknown") if model else "Not Loaded",
            "version": model.get("version", "unknown") if model else "N/A"
        }
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fare(trip: TripFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to DataFrame
    df = pd.DataFrame([trip.model_dump()])
    
    # Feature Engineering
    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['pickup_dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['pickup_dayofweek'] / 7)
    
    df['VendorID'] = 2 
    df['avg_speed_mph'] = df['trip_distance'] / (df['trip_duration_minutes'] / 60)
    df.loc[df['trip_duration_minutes'] <= 0, 'avg_speed_mph'] = 12.0
    df['avg_speed_mph'] = df['avg_speed_mph'].clip(1, 60)
    
    df['has_tolls'] = 0
    df['is_rush_hour'] = df['pickup_hour'].apply(lambda x: 1 if 16 <= x <= 19 else 0)
    df['same_location'] = (df['PULocationID'] == df['DOLocationID']).astype(int)
    
    # Column Reordering
    model_obj = model["model"]
    if hasattr(model_obj, "feature_names_in_"):
        required_features = model_obj.feature_names_in_
        for col in required_features:
            if col not in df.columns:
                df[col] = 0
        df = df[required_features]
    
    # Prediction
    try:
        prediction = model_obj.predict(df)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    prediction = max(0, prediction)
    
    # Logging
    log_prediction(trip.model_dump(), float(prediction))
    
    return PredictionResponse(
        predicted_fare=round(prediction, 2),
        currency="USD",
        model_name=model.get("model_name", "nyc-taxi-fare"),
        model_version=model.get("version", "unknown"),
        timestamp=datetime.datetime.now().isoformat()
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchTripFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = pd.DataFrame([trip.model_dump() for trip in batch.trips])
    try:
        predictions = model["model"].predict(df)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    predictions = np.maximum(predictions, 0)
    
    return BatchPredictionResponse(
        predictions=[round(p, 2) for p in predictions],
        count=len(predictions)
    )

@app.post("/reload-model", tags=["Admin"])
async def reload_model_endpoint():
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Failed to load model")
    return {
        "message": "Model reloaded successfully",
        "version": model.get("version", "unknown")
    }

def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
