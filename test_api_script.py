"""
Test script for Wine Quality API
Run this after starting the Flask service to verify it works
"""

import requests
import json

# API endpoint
URL = "http://localhost:9696/predict"

# Example wine data (a good quality wine)
wine_sample_1 = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

# Example 2: High quality wine
wine_sample_2 = {
    "fixed acidity": 8.1,
    "volatile acidity": 0.38,
    "citric acid": 0.28,
    "residual sugar": 2.1,
    "chlorides": 0.066,
    "free sulfur dioxide": 13.0,
    "total sulfur dioxide": 30.0,
    "density": 0.9968,
    "pH": 3.23,
    "sulphates": 0.73,
    "alcohol": 9.7
}

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get("http://localhost:9696/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
def test_prediction(wine_data, sample_name):
    """Test prediction endpoint"""
    print("\n" + "="*60)
    print(f"Testing Prediction: {sample_name}")
    print("="*60)
    
    print(f"\nInput Data:")
    print(json.dumps(wine_data, indent=2))
    
    response = requests.post(URL, json=wine_data)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Quality Score: {result['predictions']['quality_score']}/10")
        print(f"  Classification: {result['predictions']['quality_class']}")
        print(f"  Confidence: {result['predictions']['confidence']}%")
        print(f"\nProbabilities:")
        print(f"  Bad Wine: {result['predictions']['probabilities']['bad_wine']}")
        print(f"  Good Wine: {result['predictions']['probabilities']['good_wine']}")
    else:
        print(f"\nError: {response.json()}")

def test_error_handling():
    """Test error handling"""
    print("\n" + "="*60)
    print("Testing Error Handling (Missing Features)")
    print("="*60)
    
    incomplete_data = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7
        # Missing other required features
    }
    
    response = requests.post(URL, json=incomplete_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("="*60)
    print("WINE QUALITY API TESTING")
    print("="*60)
    print("\nMake sure the Flask server is running:")
    print("  python predict.py")
    print("\nor via Docker:")
    print("  docker run -p 9696:9696 wine-quality-api")
    
    try:
        # Test health endpoint
        test_health()
        
        # Test predictions
        test_prediction(wine_sample_1, "Sample Wine 1")
        test_prediction(wine_sample_2, "Sample Wine 2 (High Quality)")
        
        # Test error handling
        test_error_handling()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the Flask server is running on http://localhost:9696")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")