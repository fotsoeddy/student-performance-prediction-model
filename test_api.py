"""
Simple API test script
Run this after starting the API with: uvicorn app:app --reload
"""

import requests
import json

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_root():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("Testing GET /")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✓ Root endpoint working")


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing GET /health")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("✓ Health check working")


def test_predict_success():
    """Test prediction with valid data"""
    print("\n" + "="*60)
    print("Testing POST /predict (Valid Data)")
    print("="*60)
    
    data = {
        "term1_avg": 12.0,
        "term2_avg": 13.0,
        "seq5_score": 14.0,
        "attendance_percentage": 85.0,
        "parental_support": 1,
        "study_hours_per_day": 3.0,
        "sleep_hours": 7.0,
        "class_participation": 3,
        "homework_completion": 80.0,
        "extra_lessons": 1
    }
    
    print(f"Request Data: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in ["Pass", "Fail"]
    print("✓ Prediction successful")


def test_predict_invalid():
    """Test prediction with invalid data"""
    print("\n" + "="*60)
    print("Testing POST /predict (Invalid Data)")
    print("="*60)
    
    # Invalid: term1_avg exceeds maximum
    data = {
        "term1_avg": 25.0,  # Invalid: max is 20
        "term2_avg": 13.0,
        "seq5_score": 14.0,
        "attendance_percentage": 85.0,
        "parental_support": 1,
        "study_hours_per_day": 3.0,
        "sleep_hours": 7.0,
        "class_participation": 3,
        "homework_completion": 80.0,
        "extra_lessons": 1
    }
    
    print(f"Request Data (Invalid): {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 422  # Validation error
    print("✓ Validation working correctly")


def test_info():
    """Test info endpoint"""
    print("\n" + "="*60)
    print("Testing GET /info")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✓ Info endpoint working")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("STUDENT SUCCESS PREDICTION API - TEST SUITE")
    print("="*60)
    print("\nMake sure the API is running:")
    print("  uvicorn app:app --reload")
    print("\nStarting tests...\n")
    
    try:
        test_root()
        test_health()
        test_predict_success()
        test_predict_invalid()
        test_info()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nAPI is working correctly and ready for production.\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running:")
        print("  uvicorn app:app --reload")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    main()
