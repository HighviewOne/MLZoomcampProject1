"""
Wine Quality Model Training Script
Trains the final models and saves them as pickle files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("WINE QUALITY MODEL TRAINING")
print("="*80)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('winequality-red.csv', sep=';')
print(f"   Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Create binary target for classification
df['quality_binary'] = (df['quality'] >= 7).astype(int)
print(f"   Good wines (quality ≥7): {df['quality_binary'].sum()} ({df['quality_binary'].mean()*100:.1f}%)")

# Separate features and targets
X = df.drop(['quality', 'quality_binary'], axis=1)
y_regression = df['quality']
y_classification = df['quality_binary']

print(f"\n2. Features: {list(X.columns)}")

# Train/test split
print("\n3. Splitting data (80/20)...")
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

X_train_clf, X_test_clf, y_clf_train, y_clf_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Standardize features (for classification model)
print("\n4. Standardizing features...")
scaler = StandardScaler()
scaler.fit(X_train_clf)
print("   ✓ Scaler fitted")

# Train Regression Model
print("\n5. Training Random Forest Regressor...")
print("   Hyperparameter tuning with GridSearchCV...")

rf_reg_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf_reg_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_reg_params,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

rf_reg_grid.fit(X_train, y_reg_train)

print(f"   Best parameters: {rf_reg_grid.best_params_}")
print(f"   Best CV MAE: {-rf_reg_grid.best_score_:.4f}")

best_regressor = rf_reg_grid.best_estimator_

# Evaluate on test set
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred_reg = best_regressor.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
r2 = r2_score(y_reg_test, y_pred_reg)

print(f"\n   Test Set Performance:")
print(f"   MAE:  {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   R²:   {r2:.4f}")

# Train Classification Model
print("\n6. Training Random Forest Classifier...")
print("   Hyperparameter tuning with GridSearchCV...")

rf_clf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf_clf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_clf_params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

rf_clf_grid.fit(X_train_clf, y_clf_train)

print(f"   Best parameters: {rf_clf_grid.best_params_}")
print(f"   Best CV F1: {rf_clf_grid.best_score_:.4f}")

best_classifier = rf_clf_grid.best_estimator_

# Evaluate on test set
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X_test_clf_scaled = scaler.transform(X_test_clf)
y_pred_clf = best_classifier.predict(X_test_clf)
y_pred_proba = best_classifier.predict_proba(X_test_clf)[:, 1]

accuracy = accuracy_score(y_clf_test, y_pred_clf)
precision = precision_score(y_clf_test, y_pred_clf)
recall = recall_score(y_clf_test, y_pred_clf)
f1 = f1_score(y_clf_test, y_pred_clf)
roc_auc = roc_auc_score(y_clf_test, y_pred_proba)

print(f"\n   Test Set Performance:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC AUC:   {roc_auc:.4f}")

# Save models
print("\n7. Saving models...")
joblib.dump(best_regressor, 'wine_quality_regressor.pkl')
print("   ✓ Saved: wine_quality_regressor.pkl")

joblib.dump(best_classifier, 'wine_quality_classifier.pkl')
print("   ✓ Saved: wine_quality_classifier.pkl")

joblib.dump(scaler, 'feature_scaler.pkl')
print("   ✓ Saved: feature_scaler.pkl")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nModel files ready for deployment:")
print("  - wine_quality_regressor.pkl")
print("  - wine_quality_classifier.pkl")
print("  - feature_scaler.pkl")
print("\nNext steps:")
print("  1. Test the API: python predict.py")
print("  2. Build Docker: docker build -t wine-quality-api .")
print("  3. Run container: docker run -p 9696:9696 wine-quality-api")
