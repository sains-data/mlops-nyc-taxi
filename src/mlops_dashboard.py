"""
MLOps Dashboard - Comprehensive monitoring and deployment interface
Includes: Production, CI/CD, and Monitoring tabs
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

# Page config
st.set_page_config(
    page_title="NYC Taxi MLOps Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
# Support both local development and production deployment
import os

# Use environment variable or default to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Alternative: Auto-detect (try localhost first, fallback to production)
# try:
#     import requests
#     requests.get(f"http://localhost:8000/health", timeout=1)
#     API_URL = "http://localhost:8000"
# except:
#     API_URL = "https://your-app.onrender.com"  # Replace with your production URL

MODEL_PATH = Path(__file__).parent.parent / "models" / "production_model.joblib"
DATA_PATH = Path(__file__).parent.parent / "data" / "processed"
MONITORING_PATH = Path(__file__).parent.parent / "monitoring"
MONITORING_PATH.mkdir(exist_ok=True)

# Prediction log file
PREDICTION_LOG_FILE = MONITORING_PATH / "prediction_logs.json"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def check_api_status():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, response.json()
    except:
        return False, None

def load_model_info():
    """Load model information from joblib file."""
    try:
        model_package = joblib.load(MODEL_PATH)
        return model_package
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def save_prediction_log(input_data, prediction, timestamp):
    """Save prediction to log file."""
    log_entry = {
        "timestamp": timestamp,
        "input": input_data,
        "prediction": prediction
    }
    
    # Load existing logs
    if PREDICTION_LOG_FILE.exists():
        with open(PREDICTION_LOG_FILE, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Append new log
    logs.append(log_entry)
    
    # Keep only last 100 predictions
    logs = logs[-100:]
    
    # Save
    with open(PREDICTION_LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

def load_prediction_logs():
    """Load prediction logs."""
    if PREDICTION_LOG_FILE.exists():
        with open(PREDICTION_LOG_FILE, 'r') as f:
            return json.load(f)
    return []

def simulate_data_drift(data, drift_type='normal'):
    """Simulate different data drift scenarios."""
    df = data.copy()
    
    if drift_type == 'distance_drift':
        # Increase trip distances
        df['trip_distance'] = df['trip_distance'] * 1.5
    elif drift_type == 'time_drift':
        # Shift pickup hours
        df['pickup_hour'] = (df['pickup_hour'] + 3) % 24
    elif drift_type == 'passenger_drift':
        # Change passenger distribution
        df['passenger_count'] = np.random.choice([1, 2, 3, 4], size=len(df), p=[0.3, 0.3, 0.3, 0.1])
    
    return df

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🚕 NYC Taxi MLOps Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/NewYorkCity-CityFlag.svg/1200px-NewYorkCity-CityFlag.svg.png", width=100)
        st.title("Navigation")
        
        # Check API status
        api_status, api_info = check_api_status()
        if api_status:
            st.success("✅ API Connected")
            if api_info and 'model_name' in api_info:
                st.info(f"Model: {api_info['model_name']}")
        else:
            st.error("❌ API Offline")
            st.warning("Start API: `uvicorn src.api:app --reload`")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Production", "🔄 CI/CD Pipeline", "📊 Monitoring"])
    
    # ========================================
    # TAB 1: PRODUCTION
    # ========================================
    with tab1:
        st.header("🚀 Model Serving & Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Trip Information")
            
            # Input fields
            trip_distance = st.slider(
                "Trip Distance (miles)",
                min_value=0.1,
                max_value=30.0,
                value=2.5,
                step=0.1
            )
            
            pickup_hour = st.slider(
                "Pickup Hour",
                min_value=0,
                max_value=23,
                value=14
            )
            
            pickup_dayofweek = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
                index=2
            )
            
            passenger_count = st.slider(
                "Passenger Count",
                min_value=1,
                max_value=6,
                value=1
            )
            
            with st.expander("⚙️ Advanced Options"):
                pickup_month = st.slider("Month", 1, 12, 1)
                pu_location = st.number_input("Pickup Location ID", value=161)
                do_location = st.number_input("Dropoff Location ID", value=237)
                vendor_id = st.selectbox("Vendor ID", [1, 2], index=1)
        
        with col2:
            st.subheader("💰 Prediction Result")
            
            # Trip summary
            is_weekend = pickup_dayofweek >= 5
            is_rush_hour = pickup_hour in [7, 8, 9, 17, 18, 19]
            
            st.write(f"**📅 Day:** {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][pickup_dayofweek]}")
            st.write(f"**🕐 Time:** {pickup_hour}:00")
            st.write(f"**👥 Passengers:** {passenger_count}")
            st.write(f"**📏 Distance:** {trip_distance} miles")
            
            if is_weekend:
                st.info("🌴 Weekend")
            if is_rush_hour:
                st.warning("⚠️ Rush Hour")
            
            st.markdown("---")
            
            # Predict button
            if st.button("🔮 Predict Fare", type="primary", use_container_width=True):
                if not api_status:
                    st.error("❌ API is offline. Please start the API server.")
                else:
                    with st.spinner("Predicting..."):
                        try:
                            # Prepare request
                            payload = {
                                "trip_distance": trip_distance,
                                "pickup_hour": pickup_hour,
                                "pickup_dayofweek": pickup_dayofweek,
                                "passenger_count": passenger_count,
                                "pickup_month": pickup_month,
                                "PULocationID": int(pu_location),
                                "DOLocationID": int(do_location),
                                "VendorID": vendor_id
                            }
                            
                            # Call API
                            response = requests.post(
                                f"{API_URL}/predict",
                                json=payload,
                                timeout=5
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                predicted_fare = result['predicted_fare']
                                
                                # Display prediction
                                st.markdown(
                                    f'<div class="success-box">'
                                    f'<h2 style="margin:0;">💵 ${predicted_fare:.2f}</h2>'
                                    f'<p style="margin:0;">Predicted Fare</p>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                                
                                # Additional info
                                st.caption(f"Model: {result['model_name']} v{result['model_version']}")
                                st.caption(f"Timestamp: {result['timestamp']}")
                                
                                # Save log
                                save_prediction_log(payload, predicted_fare, result['timestamp'])
                                
                                # Confidence message
                                if predicted_fare < 10:
                                    st.info("💡 Short trip - typical for nearby destinations")
                                elif predicted_fare < 30:
                                    st.info("💡 Medium trip - common for cross-borough travel")
                                else:
                                    st.info("💡 Long trip - possibly to/from airports")
                            else:
                                st.error(f"Prediction failed: {response.text}")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        # Recent predictions
        st.markdown("---")
        st.subheader("📜 Recent Predictions")
        logs = load_prediction_logs()
        
        if logs:
            recent = logs[-10:][::-1]  # Last 10, reversed
            
            df_logs = pd.DataFrame([
                {
                    "Timestamp": log['timestamp'],
                    "Distance": f"{log['input']['trip_distance']} mi",
                    "Hour": log['input']['pickup_hour'],
                    "Passengers": log['input']['passenger_count'],
                    "Predicted Fare": f"${log['prediction']:.2f}"
                }
                for log in recent
            ])
            
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("No predictions yet. Make your first prediction above!")
    
    # ========================================
    # TAB 2: CI/CD PIPELINE
    # ========================================
    with tab2:
        st.header("🔄 CI/CD Pipeline Visualization")
        
        st.markdown("""
        This project uses **GitHub Actions** for Continuous Integration and Continuous Deployment.
        Every push to the repository triggers automated testing, linting, and deployment.
        """)
        
        # Pipeline stages
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>1️⃣ TEST</h3>
                <p>• Run pytest<br>
                • Code coverage<br>
                • Unit tests</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>2️⃣ LINT</h3>
                <p>• flake8<br>
                • black formatter<br>
                • isort</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>3️⃣ BUILD</h3>
                <p>• Docker images<br>
                • API container<br>
                • Training container</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>4️⃣ DEPLOY</h3>
                <p>• Push to registry<br>
                • Deploy to server<br>
                • Health check</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Pipeline flow diagram
        st.subheader("📊 Pipeline Flow")
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Code Push", "Test", "Lint", "Build", "Deploy", "Production"],
                color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#17becf"]
            ),
            link=dict(
                source=[0, 1, 2, 3, 4],
                target=[1, 2, 3, 4, 5],
                value=[10, 10, 10, 10, 10]
            )
        )])
        
        fig.update_layout(
            title="CI/CD Pipeline Flow",
            font_size=12,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Workflow file
        st.markdown("---")
        st.subheader("⚙️ GitHub Actions Workflow")
        
        workflow_path = Path(__file__).parent.parent / ".github" / "workflows" / "ci-cd.yml"
        
        if workflow_path.exists():
            st.info("✅ Workflow file found: `.github/workflows/ci-cd.yml`")
            
            with st.expander("📄 View Workflow Configuration"):
                with open(workflow_path, 'r') as f:
                    st.code(f.read(), language='yaml')
        else:
            st.warning("⚠️ Workflow file not found")
        
        # Instructions
        st.markdown("---")
        st.subheader("🚀 How to Trigger CI/CD")
        
        st.code("""
# 1. Make changes to your code
git add .
git commit -m "Update model or code"

# 2. Push to GitHub (triggers CI/CD automatically)
git push origin main

# 3. Check pipeline status
# Go to: GitHub Repository → Actions tab
        """, language='bash')
        
        st.info("""
        💡 **Tip:** The pipeline runs automatically on every push to `main` or `develop` branches.
        You can view the pipeline status and logs in the GitHub Actions tab of your repository.
        """)
    
    # ========================================
    # TAB 3: MONITORING
    # ========================================
    with tab3:
        st.header("📊 Model Monitoring & Data Drift Detection")
        
        # Load training data for reference
        try:
            train_data_path = DATA_PATH / "train.parquet"
            
            if train_data_path.exists():
                reference_data = pd.read_parquet(train_data_path).sample(n=min(1000, len(pd.read_parquet(train_data_path))))
                
                st.success("✅ Reference data loaded")
                
                # Drift simulator
                st.subheader("🧪 Data Drift Simulator")
                st.markdown("Simulate different data drift scenarios to see how the monitoring system detects changes.")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("✅ Normal Data", use_container_width=True):
                        st.session_state['drift_type'] = 'normal'
                        st.session_state['current_data'] = reference_data
                
                with col2:
                    if st.button("📏 Distance Drift", use_container_width=True):
                        st.session_state['drift_type'] = 'distance_drift'
                        st.session_state['current_data'] = simulate_data_drift(reference_data, 'distance_drift')
                
                with col3:
                    if st.button("🕐 Time Drift", use_container_width=True):
                        st.session_state['drift_type'] = 'time_drift'
                        st.session_state['current_data'] = simulate_data_drift(reference_data, 'time_drift')
                
                # Initialize session state
                if 'drift_type' not in st.session_state:
                    st.session_state['drift_type'] = 'normal'
                    st.session_state['current_data'] = reference_data
                
                current_data = st.session_state['current_data']
                drift_type = st.session_state['drift_type']
                
                # Display drift status
                st.markdown("---")
                
                if drift_type == 'normal':
                    st.markdown('<div class="success-box"><h3>✅ No Drift Detected</h3><p>Data distribution is normal</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box"><h3>⚠️ Drift Detected!</h3><p>Data distribution has changed significantly</p></div>', unsafe_allow_html=True)
                
                # Feature comparison
                st.markdown("---")
                st.subheader("📈 Feature Distribution Comparison")
                
                feature_to_compare = st.selectbox(
                    "Select feature to analyze",
                    ['trip_distance', 'pickup_hour', 'passenger_count', 'pickup_dayofweek']
                )
                
                # Create comparison plot
                fig = go.Figure()
                
                # Reference data
                fig.add_trace(go.Histogram(
                    x=reference_data[feature_to_compare],
                    name="Training Data",
                    opacity=0.7,
                    marker_color='blue'
                ))
                
                # Current data
                fig.add_trace(go.Histogram(
                    x=current_data[feature_to_compare],
                    name="Current Data",
                    opacity=0.7,
                    marker_color='red' if drift_type != 'normal' else 'green'
                ))
                
                fig.update_layout(
                    title=f"Distribution Comparison: {feature_to_compare}",
                    xaxis_title=feature_to_compare,
                    yaxis_title="Frequency",
                    barmode='overlay',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Reference Mean",
                        f"{reference_data[feature_to_compare].mean():.2f}",
                        delta=None
                    )
                
                with col2:
                    current_mean = current_data[feature_to_compare].mean()
                    reference_mean = reference_data[feature_to_compare].mean()
                    delta = current_mean - reference_mean
                    
                    st.metric(
                        "Current Mean",
                        f"{current_mean:.2f}",
                        delta=f"{delta:.2f}",
                        delta_color="inverse" if abs(delta) > 0.1 else "normal"
                    )
                
                # Recommendations
                st.markdown("---")
                st.subheader("💡 Recommendations")
                
                if drift_type != 'normal':
                    st.warning("""
                    **Action Required:**
                    - 🔄 Consider retraining the model with recent data
                    - 📊 Investigate the cause of data drift
                    - 🔍 Monitor model performance closely
                    - 📝 Document the drift in your MLOps logs
                    """)
                else:
                    st.success("""
                    **Status: Healthy**
                    - ✅ No action required
                    - 📊 Continue monitoring regularly
                    - 🔍 Set up automated drift detection alerts
                    """)
            
            else:
                st.warning("⚠️ Training data not found. Please ensure processed data exists.")
                st.info(f"Expected path: {train_data_path}")
        
        except Exception as e:
            st.error(f"Error loading monitoring data: {e}")
            st.info("Make sure you have run the preprocessing notebook to generate processed data.")

if __name__ == "__main__":
    main()
