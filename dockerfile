# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and application
COPY wine_quality_regressor.pkl .
COPY wine_quality_classifier.pkl .
COPY feature_scaler.pkl .
COPY predict.py .

# Expose port 9696
EXPOSE 9696

# Run the Flask application
CMD ["python", "predict.py"]
