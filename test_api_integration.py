#!/usr/bin/env python3
"""
Test script to verify Dashboard uses FastAPI (not direct model loading)
"""

import requests
import sys

API_URL = "http://localhost:8000"

def test_api_connection():
    """Test if API is running."""
    print("🧪 Test 1: API Connection")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is running")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   Model name: {data.get('model_name')}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API")
        print("   Make sure API is running:")
        print("   uvicorn src.api:app --reload --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction_via_api():
    """Test prediction through API."""
    print("\n🧪 Test 2: Prediction via API")
    print("=" * 50)
    
    payload = {
        "trip_distance": 2.5,
        "pickup_hour": 14,
        "pickup_dayofweek": 2,
        "passenger_count": 2,
        "pickup_month": 1,
        "PULocationID": 161,
        "DOLocationID": 237,
        "VendorID": 2
    }
    
    print(f"Sending request to: {API_URL}/predict")
    print(f"Payload: {payload}")
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prediction successful!")
            print(f"   Predicted fare: ${result['predicted_fare']:.2f}")
            print(f"   Model: {result['model_name']} v{result['model_version']}")
            print(f"   Timestamp: {result['timestamp']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint."""
    print("\n🧪 Test 3: Model Info Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=2)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model name: {data.get('model_name')}")
            print(f"   Model type: {data.get('model_type')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Features: {data.get('num_features')}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def verify_dashboard_code():
    """Verify dashboard code doesn't load model directly."""
    print("\n🧪 Test 4: Dashboard Code Verification")
    print("=" * 50)
    
    dashboard_file = "src/mlops_dashboard.py"
    
    try:
        with open(dashboard_file, 'r') as f:
            content = f.read()
        
        # Check for direct model loading
        if "model.predict(" in content:
            print(f"❌ Found 'model.predict(' in dashboard code!")
            print("   Dashboard is directly using model (NOT via API)")
            return False
        
        # Check for API usage
        api_calls = [
            "requests.post(",
            'f"{API_URL}/predict"',
            "requests.get("
        ]
        
        found_api = any(call in content for call in api_calls)
        
        if found_api:
            print(f"✅ Dashboard uses API (requests library)")
            print(f"   No direct model.predict() found")
            
            # Count API calls
            predict_calls = content.count('requests.post(')
            health_calls = content.count('requests.get(')
            print(f"   Found {predict_calls} POST requests")
            print(f"   Found {health_calls} GET requests")
            
            return True
        else:
            print(f"⚠️  No API calls found in dashboard")
            return False
            
    except FileNotFoundError:
        print(f"❌ File not found: {dashboard_file}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🔍 VERIFICATION: Dashboard Uses FastAPI")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("API Connection", test_api_connection()))
    
    if results[0][1]:  # Only test prediction if API is running
        results.append(("Prediction via API", test_prediction_via_api()))
        results.append(("Model Info Endpoint", test_model_info()))
    
    results.append(("Dashboard Code Check", verify_dashboard_code()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results if result[1] is not None)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Dashboard correctly uses FastAPI for predictions")
        print("✅ No direct model loading in dashboard code")
        print("=" * 60)
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Check the output above for details")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
