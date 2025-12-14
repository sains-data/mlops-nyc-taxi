"""Streamlit App for NYC Taxi Fare Prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / "models" / "production_model.joblib"
    return joblib.load(model_path)

# Main app
def main():
    st.title("🚕 NYC Taxi Fare Predictor")
    st.markdown("Predict taxi fare using Machine Learning")
    
    # Load model
    try:
        model_package = load_model()
        model = model_package['model']
        features = model_package['features']
        
        st.sidebar.success(f"✅ Model: {model_package['model_name']}")
        st.sidebar.info(f"Version: {model_package['version']}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Sidebar inputs
    st.sidebar.header("🎛️ Trip Details")
    
    trip_distance = st.sidebar.slider(
        "Trip Distance (miles)",
        min_value=0.1,
        max_value=30.0,
        value=2.5,
        step=0.1
    )
    
    pickup_hour = st.sidebar.slider(
        "Pickup Hour",
        min_value=0,
        max_value=23,
        value=14
    )
    
    pickup_dayofweek = st.sidebar.selectbox(
        "Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
        index=2
    )
    
    passenger_count = st.sidebar.slider(
        "Passenger Count",
        min_value=1,
        max_value=6,
        value=1
    )
    
    # Calculate derived features based on input
    # Better estimation based on actual data patterns:
    # - Short trips (<0.5 mi): avg 4 min, 7 mph
    # - Medium trips: avg 12 mph
    # - Minimum trip duration is ~3 minutes
    
    if trip_distance < 0.5:
        avg_speed_mph = 7.0
        trip_duration_minutes = max(3.0, (trip_distance / avg_speed_mph) * 60)
    elif trip_distance < 1.0:
        avg_speed_mph = 9.0
        trip_duration_minutes = max(5.0, (trip_distance / avg_speed_mph) * 60)
    else:
        avg_speed_mph = 12.0
        trip_duration_minutes = (trip_distance / avg_speed_mph) * 60
    
    # Cyclical encoding for hour (0-23 mapped to sin/cos)
    hour_sin = np.sin(2 * np.pi * pickup_hour / 24)
    hour_cos = np.cos(2 * np.pi * pickup_hour / 24)
    
    # Cyclical encoding for day of week (0-6 mapped to sin/cos)
    dow_sin = np.sin(2 * np.pi * pickup_dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * pickup_dayofweek / 7)
    
    # Binary features
    is_weekend = 1 if pickup_dayofweek >= 5 else 0
    is_rush_hour = 1 if pickup_hour in [7, 8, 9, 17, 18, 19] else 0
    
    # Create input DataFrame with ALL features
    input_data = {
        'trip_distance': trip_distance,
        'passenger_count': passenger_count,
        'trip_duration_minutes': trip_duration_minutes,
        'avg_speed_mph': avg_speed_mph,
        'pickup_hour': pickup_hour,
        'pickup_dayofweek': pickup_dayofweek,
        'pickup_month': 1,  # default January
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'PULocationID': 161,  # Manhattan midtown (common)
        'DOLocationID': 237,  # Upper East Side (common)
        'VendorID': 2,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour,
        'same_location': 0,
        'has_tolls': 0
    }
    
    df = pd.DataFrame([input_data])
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Trip Information")
        st.write(f"**Distance:** {trip_distance} miles")
        st.write(f"**Est. Duration:** {trip_duration_minutes:.0f} minutes")
        st.write(f"**Time:** {pickup_hour}:00")
        st.write(f"**Day:** {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][pickup_dayofweek]}")
        st.write(f"**Passengers:** {passenger_count}")
        if is_rush_hour:
            st.warning("⚠️ Rush Hour")
        if is_weekend:
            st.info("🌴 Weekend")
    
    with col2:
        st.subheader("💰 Fare Prediction")
        
        if st.button("🔮 Predict Fare", type="primary"):
            with st.spinner("Calculating..."):
                prediction = model.predict(df[features])[0]
                
                st.metric(
                    label="Predicted Fare",
                    value=f"${prediction:.2f}"
                )
                
                # Confidence message
                if prediction < 10:
                    st.info("💡 Short trip - typical for nearby destinations")
                elif prediction < 30:
                    st.info("💡 Medium trip - common for cross-borough travel")
                else:
                    st.info("💡 Long trip - possibly to/from airports")
    
    # Footer
    st.markdown("---")
    st.markdown("**📊 Model Info**")
    with st.expander("View Model Details"):
        st.json({
            "model_name": model_package['model_name'],
            "model_type": model_package['model_type'],
            "version": model_package['version'],
            "num_features": len(features),
            "created_at": model_package['created_at']
        })

if __name__ == "__main__":
    main()
