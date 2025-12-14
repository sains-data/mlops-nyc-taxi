#!/bin/bash

# Script to run API and Dashboard
# Usage: ./run_demo.sh

echo "🚀 Starting NYC Taxi MLOps Demo..."
echo ""

# Check if model exists
if [ ! -f "models/production_model.joblib" ]; then
    echo "❌ Error: Model file not found!"
    echo "Please run notebook 07_deployment.ipynb first."
    exit 1
fi

echo "✅ Model file found"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $API_PID $DASHBOARD_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start API
echo "📡 Starting FastAPI backend..."
python3 -m uvicorn src.api:app --reload --port 8000 > /tmp/api.log 2>&1 &
API_PID=$!

# Wait for API to start
sleep 3

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is running on http://localhost:8000"
    echo "   Swagger docs: http://localhost:8000/docs"
else
    echo "❌ Failed to start API"
    cat /tmp/api.log
    exit 1
fi

echo ""
echo "🎨 Starting Streamlit dashboard..."
sleep 2
python3 -m streamlit run src/mlops_dashboard.py &
DASHBOARD_PID=$!

echo ""
echo "✅ Dashboard starting on http://localhost:8501"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Demo is ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 Services:"
echo "   • API:       http://localhost:8000"
echo "   • Dashboard: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for both processes
wait
