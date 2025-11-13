"""
Flask Web Service for Wine Quality Prediction
Serves the trained models via REST API
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask('wine-quality-predictor')

# Load models at startup
print("Loading models...")
regressor = joblib.load('wine_quality_regressor.pkl')
classifier = joblib.load('wine_quality_classifier.pkl')
scaler = joblib.load('feature_scaler.pkl')
print("Models loaded successfully!")

# Feature names in correct order
FEATURE_NAMES = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'wine-quality-predictor',
        'version': '1.0'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict wine quality from physicochemical properties
    
    Expected JSON format:
    {
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
    """
    try:
        # Get JSON data from request
        wine_data = request.get_json()
        
        # Validate input
        if not wine_data:
            return jsonify({
                'error': 'No input data provided',
                'status': 'error'
            }), 400
        
        # Check for missing features
        missing_features = [f for f in FEATURE_NAMES if f not in wine_data]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing_features': missing_features,
                'status': 'error'
            }), 400
        
        # Prepare input data in correct order
        input_data = [[wine_data[feature] for feature in FEATURE_NAMES]]
        input_df = pd.DataFrame(input_data, columns=FEATURE_NAMES)
        
        # Make predictions
        # Regression prediction (quality score)
        quality_score = float(regressor.predict(input_df)[0])
        
        # Classification prediction (good/bad wine)
        input_scaled = scaler.transform(input_df)
        quality_class = int(classifier.predict(input_scaled)[0])
        quality_proba = classifier.predict_proba(input_scaled)[0].tolist()
        
        # Prepare response
        response = {
            'status': 'success',
            'predictions': {
                'quality_score': round(quality_score, 2),
                'quality_class': 'Good Wine (≥7)' if quality_class == 1 else 'Average Wine (<7)',
                'quality_class_numeric': quality_class,
                'probabilities': {
                    'bad_wine': round(quality_proba[0], 3),
                    'good_wine': round(quality_proba[1], 3)
                },
                'confidence': round(max(quality_proba) * 100, 1)
            },
            'input': wine_data
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            'error': f'Invalid input values: {str(e)}',
            'status': 'error'
        }), 400
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'error'
        }), 500


@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API documentation"""
    return jsonify({
        'service': 'Wine Quality Prediction API',
        'version': '1.0',
        'endpoints': {
            'GET /': 'API documentation',
            'GET /health': 'Health check',
            'POST /predict': 'Predict wine quality'
        },
        'example_request': {
            'url': '/predict',
            'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'body': {
                'fixed acidity': 7.4,
                'volatile acidity': 0.7,
                'citric acid': 0.0,
                'residual sugar': 1.9,
                'chlorides': 0.076,
                'free sulfur dioxide': 11.0,
                'total sulfur dioxide': 34.0,
                'density': 0.9978,
                'pH': 3.51,
                'sulphates': 0.56,
                'alcohol': 9.4
            }
        },
        'features_required': FEATURE_NAMES
    }), 200


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=9696)