"""
Wine Quality Prediction Web App
Deploy with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Title and description
st.title("🍷 Red Wine Quality Prediction")
st.markdown("""
This app predicts the quality of red wine based on physicochemical properties.
Enter the wine's characteristics below to get predictions for both:
- **Regression**: Numeric quality score (0-10)
- **Classification**: Good wine (≥7) or Bad wine (<7)
""")

# Load models
@st.cache_resource
def load_models():
    try:
        regressor = joblib.load('wine_quality_regressor.pkl')
        classifier = joblib.load('wine_quality_classifier.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return regressor, classifier, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the training notebook first to generate the model files.")
        st.stop()

regressor, classifier, scaler = load_models()

# Feature names - MUST match the training data column names exactly
# The CSV uses these exact column names (with quotes removed by pandas)
feature_names = [
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

# Feature descriptions
feature_info = {
    'fixed acidity': 'Tartaric acid content (g/dm³)',
    'volatile acidity': 'Acetic acid content (g/dm³) - high values can lead to vinegar taste',
    'citric acid': 'Adds freshness and flavor (g/dm³)',
    'residual sugar': 'Sugar remaining after fermentation (g/dm³)',
    'chlorides': 'Salt content (g/dm³)',
    'free sulfur dioxide': 'Free form of SO₂ (mg/dm³) - prevents microbial growth',
    'total sulfur dioxide': 'Total SO₂ (mg/dm³)',
    'density': 'Density of wine (g/cm³)',
    'pH': 'Acidity level (0-14 scale)',
    'sulphates': 'Potassium sulphate content (g/dm³)',
    'alcohol': 'Alcohol percentage (%)'
}

# Typical ranges for features
feature_ranges = {
    'fixed acidity': (4.6, 15.9),
    'volatile acidity': (0.12, 1.58),
    'citric acid': (0.0, 1.0),
    'residual sugar': (0.9, 15.5),
    'chlorides': (0.012, 0.611),
    'free sulfur dioxide': (1.0, 72.0),
    'total sulfur dioxide': (6.0, 289.0),
    'density': (0.9901, 1.0037),
    'pH': (2.74, 4.01),
    'sulphates': (0.33, 2.0),
    'alcohol': (8.4, 14.9)
}

# Default values (median values from dataset)
default_values = {
    'fixed acidity': 7.9,
    'volatile acidity': 0.52,
    'citric acid': 0.26,
    'residual sugar': 2.2,
    'chlorides': 0.079,
    'free sulfur dioxide': 14.0,
    'total sulfur dioxide': 38.0,
    'density': 0.9968,
    'pH': 3.31,
    'sulphates': 0.62,
    'alcohol': 10.2
}

st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns(2)

# Collect user inputs
user_inputs = {}

with col1:
    st.subheader("📊 Acidity & Chemical Properties")
    
    user_inputs['fixed acidity'] = st.slider(
        'Fixed Acidity',
        min_value=float(feature_ranges['fixed acidity'][0]),
        max_value=float(feature_ranges['fixed acidity'][1]),
        value=float(default_values['fixed acidity']),
        help=feature_info['fixed acidity']
    )
    
    user_inputs['volatile acidity'] = st.slider(
        'Volatile Acidity',
        min_value=float(feature_ranges['volatile acidity'][0]),
        max_value=float(feature_ranges['volatile acidity'][1]),
        value=float(default_values['volatile acidity']),
        help=feature_info['volatile acidity']
    )
    
    user_inputs['citric acid'] = st.slider(
        'Citric Acid',
        min_value=float(feature_ranges['citric acid'][0]),
        max_value=float(feature_ranges['citric acid'][1]),
        value=float(default_values['citric acid']),
        help=feature_info['citric acid']
    )
    
    user_inputs['pH'] = st.slider(
        'pH',
        min_value=float(feature_ranges['pH'][0]),
        max_value=float(feature_ranges['pH'][1]),
        value=float(default_values['pH']),
        help=feature_info['pH']
    )
    
    st.subheader("🧪 Sulfur Compounds")
    
    user_inputs['free sulfur dioxide'] = st.slider(
        'Free Sulfur Dioxide',
        min_value=float(feature_ranges['free sulfur dioxide'][0]),
        max_value=float(feature_ranges['free sulfur dioxide'][1]),
        value=float(default_values['free sulfur dioxide']),
        help=feature_info['free sulfur dioxide']
    )
    
    user_inputs['total sulfur dioxide'] = st.slider(
        'Total Sulfur Dioxide',
        min_value=float(feature_ranges['total sulfur dioxide'][0]),
        max_value=float(feature_ranges['total sulfur dioxide'][1]),
        value=float(default_values['total sulfur dioxide']),
        help=feature_info['total sulfur dioxide']
    )
    
    user_inputs['sulphates'] = st.slider(
        'Sulphates',
        min_value=float(feature_ranges['sulphates'][0]),
        max_value=float(feature_ranges['sulphates'][1]),
        value=float(default_values['sulphates']),
        help=feature_info['sulphates']
    )

with col2:
    st.subheader("🍬 Sugar & Salt Content")
    
    user_inputs['residual sugar'] = st.slider(
        'Residual Sugar',
        min_value=float(feature_ranges['residual sugar'][0]),
        max_value=float(feature_ranges['residual sugar'][1]),
        value=float(default_values['residual sugar']),
        help=feature_info['residual sugar']
    )
    
    user_inputs['chlorides'] = st.slider(
        'Chlorides',
        min_value=float(feature_ranges['chlorides'][0]),
        max_value=float(feature_ranges['chlorides'][1]),
        value=float(default_values['chlorides']),
        help=feature_info['chlorides']
    )
    
    st.subheader("🍾 Physical Properties")
    
    user_inputs['density'] = st.slider(
        'Density',
        min_value=float(feature_ranges['density'][0]),
        max_value=float(feature_ranges['density'][1]),
        value=float(default_values['density']),
        help=feature_info['density'],
        format="%.4f"
    )
    
    user_inputs['alcohol'] = st.slider(
        'Alcohol',
        min_value=float(feature_ranges['alcohol'][0]),
        max_value=float(feature_ranges['alcohol'][1]),
        value=float(default_values['alcohol']),
        help=feature_info['alcohol']
    )

st.markdown("---")

# Predict button
if st.button("🔮 Predict Wine Quality", type="primary", use_container_width=True):
    # Prepare input data - CRITICAL: Must be in the exact order as training data
    input_data = {
        'fixed acidity': [user_inputs['fixed acidity']],
        'volatile acidity': [user_inputs['volatile acidity']],
        'citric acid': [user_inputs['citric acid']],
        'residual sugar': [user_inputs['residual sugar']],
        'chlorides': [user_inputs['chlorides']],
        'free sulfur dioxide': [user_inputs['free sulfur dioxide']],
        'total sulfur dioxide': [user_inputs['total sulfur dioxide']],
        'density': [user_inputs['density']],
        'pH': [user_inputs['pH']],
        'sulphates': [user_inputs['sulphates']],
        'alcohol': [user_inputs['alcohol']]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Verify column order matches training
    input_df = input_df[feature_names]
    
    # Make predictions
    # Regression prediction
    quality_pred = regressor.predict(input_df)[0]
    
    # Classification prediction (needs scaling)
    input_scaled = scaler.transform(input_df)
    class_pred = classifier.predict(input_scaled)[0]
    class_proba = classifier.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    # Create three columns for results
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.metric(
            label="Quality Score (Regression)",
            value=f"{quality_pred:.2f}",
            delta=f"{quality_pred - 5.5:.2f} from average"
        )
    
    with res_col2:
        quality_label = "Good Wine 🎉" if class_pred == 1 else "Average Wine 📊"
        st.metric(
            label="Quality Classification",
            value=quality_label
        )
    
    with res_col3:
        confidence = max(class_proba) * 100
        st.metric(
            label="Confidence",
            value=f"{confidence:.1f}%"
        )
    
    # Interpretation
    st.markdown("---")
    st.subheader("📝 Interpretation")
    
    if quality_pred >= 7:
        st.success(f"""
        **Excellent Wine!** 🌟
        
        This wine is predicted to have a quality score of **{quality_pred:.2f}/10**, which is considered high quality.
        Wines with scores ≥7 are typically well-balanced with excellent characteristics.
        """)
    elif quality_pred >= 5.5:
        st.info(f"""
        **Good Wine** ✓
        
        This wine is predicted to have a quality score of **{quality_pred:.2f}/10**, which is above average.
        Most wines in the dataset fall in this range (5-6).
        """)
    else:
        st.warning(f"""
        **Below Average Wine** ⚠️
        
        This wine is predicted to have a quality score of **{quality_pred:.2f}/10**, which is below average.
        Consider adjusting the chemical properties for better quality.
        """)
    
    # Probability breakdown
    st.markdown("---")
    st.subheader("📊 Classification Probabilities")
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.metric("Bad Wine (Quality < 7)", f"{class_proba[0]*100:.1f}%")
    
    with prob_col2:
        st.metric("Good Wine (Quality ≥ 7)", f"{class_proba[1]*100:.1f}%")
    
    # Feature importance insights
    st.markdown("---")
    st.subheader("💡 Key Factors")
    st.markdown("""
    Based on the trained model, the most important features for wine quality are:
    1. **Alcohol** - Higher alcohol content generally correlates with better quality
    2. **Volatile Acidity** - Lower values are preferred (too much causes vinegar taste)
    3. **Sulphates** - Contributes to wine's antimicrobial properties
    4. **Citric Acid** - Adds freshness and flavor complexity
    """)

# Sidebar with additional info
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**Wine Quality Prediction System**

This machine learning model was trained on 1,599 red wine samples to predict quality based on physicochemical tests.

**Models Used:**
- Regression: Random Forest Regressor
- Classification: Random Forest Classifier

**Quality Scale:**
- 0-4: Poor
- 5-6: Average (most common)
- 7-10: Good to Excellent
""")

st.sidebar.header("📚 Dataset Info")
st.sidebar.markdown("""
**Features:** 11 physicochemical properties

**Source:** Wine Quality Dataset (UCI Machine Learning Repository)

**Target:** Quality score (0-10) based on sensory data
""")

st.sidebar.header("🔧 Usage Tips")
st.sidebar.markdown("""
1. Adjust sliders to input wine characteristics
2. Click "Predict Wine Quality" button
3. View both regression score and classification
4. Experiment with different values to see how they affect quality
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ using Streamlit | Wine Quality Prediction Project</p>
</div>
""", unsafe_allow_html=True)