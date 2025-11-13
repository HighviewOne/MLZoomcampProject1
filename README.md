# 🍷 Wine Quality Prediction - ML Zoomcamp Project

## Problem Description

**Context:** Wine quality assessment is crucial for wineries, distributors, and consumers. Traditional quality evaluation relies on expert sommeliers conducting sensory analysis, which is subjective, time-consuming, and expensive. 

**Problem:** Can we predict wine quality objectively using physicochemical properties measured through laboratory tests?

**Solution:** This project builds a machine learning model that predicts wine quality based on 11 physicochemical properties (acidity, pH, alcohol content, etc.), providing:
1. **Regression Model**: Predicts exact quality score (0-10 scale)
2. **Classification Model**: Classifies wine as "Good" (≥7) or "Average" (<7)

**Business Value:**
- 🏭 **Winemakers**: Optimize production processes by identifying key quality factors
- 🔬 **Quality Control**: Early detection of quality issues before bottling
- 💰 **Pricing**: Data-driven pricing based on predicted quality
- 🛒 **Consumers**: Make informed purchasing decisions

---

## Dataset

**Source:** [UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

**Description:**
- **Samples**: 1,599 red wine samples
- **Features**: 11 physicochemical properties
- **Target**: Quality score (0-10) based on sensory evaluation

**Features:**
1. `fixed acidity` - Tartaric acid content (g/dm³)
2. `volatile acidity` - Acetic acid content (g/dm³)
3. `citric acid` - Freshness factor (g/dm³)
4. `residual sugar` - Sugar after fermentation (g/dm³)
5. `chlorides` - Salt content (g/dm³)
6. `free sulfur dioxide` - Free SO₂ (mg/dm³)
7. `total sulfur dioxide` - Total SO₂ (mg/dm³)
8. `density` - Wine density (g/cm³)
9. `pH` - Acidity level (0-14 scale)
10. `sulphates` - Potassium sulphate (g/dm³)
11. `alcohol` - Alcohol percentage (%)

**Dataset included in repository:** `winequality-red.csv`

---

## Project Structure

```
wine-quality-prediction/
│
├── data/
│   └── winequality-red.csv          # Dataset
│
├── notebook.ipynb                    # Complete EDA and modeling
├── train.py                          # Training script
├── predict.py                        # Flask API service
├── test_api.py                       # API testing script
│
├── wine_quality_regressor.pkl        # Trained regression model
├── wine_quality_classifier.pkl       # Trained classification model
├── feature_scaler.pkl                # Feature scaler
│
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker configuration
├── README.md                         # This file
│
└── app.py                           # (BONUS) Streamlit web app
```

---

## Key Findings from EDA

### Data Overview
- No missing values ✓
- Quality scores range from 3 to 8 (most wines are 5-6)
- Only 13.6% of wines are "good quality" (≥7) - **class imbalance**

### Most Important Features
1. **Alcohol** (+) - Higher alcohol → Better quality
2. **Volatile Acidity** (-) - Too much → Vinegar taste
3. **Sulphates** (+) - Preservative quality
4. **Citric Acid** (+) - Adds freshness

### Correlations
- Strong positive: Alcohol ↔ Quality (ρ = 0.48)
- Strong negative: Volatile Acidity ↔ Quality (ρ = -0.39)
- Density and alcohol are inversely related

---

## Model Performance

### Regression Model (Random Forest)
- **MAE**: 0.52 (out of 10-point scale)
- **RMSE**: 0.65
- **R²**: 0.42

### Classification Model (Random Forest)
- **Accuracy**: 88%
- **Precision**: 0.68
- **Recall**: 0.54
- **F1-Score**: 0.60
- **ROC AUC**: 0.75

**Model Selection:** Random Forest outperformed Linear Regression and Gradient Boosting after hyperparameter tuning with GridSearchCV.

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Option 1: Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/HighviewOne/MLZoomcampProject1.git
cd MLZoomcampProject1
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model** (optional - models already included)
```bash
python train.py
```

5. **Start the Flask API**
```bash
python predict.py
```

The API will be available at `http://localhost:9696`

### Option 2: Docker Deployment

1. **Build Docker image**
```bash
docker build -t wine-quality-api .
```

2. **Run container**
```bash
docker run -it -p 9696:9696 wine-quality-api
```

3. **Verify it's running**
```bash
curl http://localhost:9696/health
```

---

## API Usage

### Endpoints

#### 1. Health Check
```bash
GET http://localhost:9696/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "wine-quality-predictor",
  "version": "1.0"
}
```

#### 2. Predict Wine Quality
```bash
POST http://localhost:9696/predict
Content-Type: application/json
```

**Request Body:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "predictions": {
    "quality_score": 5.23,
    "quality_class": "Average Wine (<7)",
    "quality_class_numeric": 0,
    "probabilities": {
      "bad_wine": 0.847,
      "good_wine": 0.153
    },
    "confidence": 84.7
  },
  "input": {...}
}
```

### Testing the API

**Using curl:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Using Python:**
```python
import requests

wine_data = {
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

response = requests.post('http://localhost:9696/predict', json=wine_data)
print(response.json())
```

**Using test script:**
```bash
python test_api.py
```

---

## Reproducibility

### Training from Scratch

1. **Ensure dataset is present:** `winequality-red.csv`

2. **Run training script:**
```bash
python train.py
```

This will:
- Load and preprocess data
- Train models with GridSearchCV
- Save three files:
  - `wine_quality_regressor.pkl`
  - `wine_quality_classifier.pkl`
  - `feature_scaler.pkl`

3. **Start API:**
```bash
python predict.py
```

### Running the Notebook

Open `notebook.ipynb` in Jupyter:
```bash
jupyter notebook notebook.ipynb
```

The notebook contains:
- Complete EDA with visualizations
- Feature analysis
- Model training and comparison
- Hyperparameter tuning
- Model evaluation

---

## Bonus: Streamlit Web App

An interactive web interface is also provided (not required for ML Zoomcamp, but great for demos!):

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

Features:
- Interactive sliders for all 11 features
- Real-time predictions
- Visual feedback and confidence scores
- Educational tooltips

---

## Deployment (Bonus)

### Local Docker Deployment ✅

**Build and run:**
```bash
docker build -t wine-quality-api .
docker run -it -p 9696:9696 wine-quality-api
```

### Cloud Deployment Options

The service can be deployed to:
- **AWS Elastic Beanstalk**
- **Google Cloud Run**
- **Azure Container Instances**
- **Heroku**

*(Deployment instructions available upon request)*

---

## Project Limitations & Future Work

### Current Limitations
1. **Dataset Size**: Only 1,599 samples - more data could improve generalization
2. **Class Imbalance**: Only 13.6% "good" wines - affects classification performance
3. **Geographic Scope**: Dataset from Portuguese wines - may not generalize globally
4. **Quality Subjectivity**: Based on human ratings which vary

### Future Improvements
1. **Feature Engineering**: Polynomial features, interaction terms
2. **Advanced Models**: XGBoost, Neural Networks
3. **Ensemble Methods**: Stacking multiple models
4. **SHAP Values**: For better model interpretability
5. **Real-time Monitoring**: Track model performance in production
6. **Multi-class Classification**: Predict exact quality levels (3-8)

---

## Tech Stack

- **Language**: Python 3.11
- **ML Framework**: Scikit-learn
- **Web Framework**: Flask
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Docker
- **Version Control**: Git

---

## Model Files

Three pickle files are included:
1. **wine_quality_regressor.pkl** (3.2 MB) - Random Forest Regressor
2. **wine_quality_classifier.pkl** (3.0 MB) - Random Forest Classifier  
3. **feature_scaler.pkl** (1.5 KB) - StandardScaler for features

---

## Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-profile](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

---

## Acknowledgments

- **Dataset**: UCI Machine Learning Repository
- **Course**: ML Zoomcamp by DataTalks.Club
- **Inspiration**: Cortez et al. (2009) - Decision Support Systems

---

## License

This project is for educational purposes as part of ML Zoomcamp.

---

## ML Zoomcamp Project Checklist

- [x] Problem description with context
- [x] Dataset included and documented
- [x] Complete EDA in notebook
- [x] Multiple models trained and compared
- [x] Hyperparameter tuning performed
- [x] `train.py` script for model training
- [x] `predict.py` script with Flask API
- [x] Dependencies listed in `requirements.txt`
- [x] Dockerfile for containerization
- [x] API endpoints tested and documented
- [x] Reproducible from scratch
- [x] README with clear instructions
