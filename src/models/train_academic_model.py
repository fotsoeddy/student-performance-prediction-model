# =========================================
# ACADEMIC MODEL TRAINING WITH CALIBRATION
# =========================================
"""
Trains the academic model with probability calibration.
Uses CalibratedClassifierCV to produce well-calibrated probabilities.
Saves metrics to JSON for reporting.
"""

import pandas as pd
import joblib
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# =========================================
# 1. LOAD DATA
# =========================================
print("=" * 60)
print("ACADEMIC MODEL TRAINING WITH CALIBRATION")
print("=" * 60)

DATA_PATH = "data/processed/aligned_student_data.csv"
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found. Run dataset_alignment.py first.")
    exit(1)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape: {df.shape}")

# =========================================
# 2. FEATURES & TARGET
# =========================================
TARGET = "Pass"
FEATURES = [
    "Term1_avg",
    "Term2_avg",
    "Seq5_score",
    "Attendance_percentage",
    "Parental_support"
]

X = df[FEATURES]
y = df[TARGET]

# Add guard: Imputation if any missing values slipped through
if X.isnull().any().any():
    print("  ⚠️ Warning: Missing values found in training data. Imputing...")
    X = X.fillna(X.median())

print(f"\nClass distribution:")
print(y.value_counts(normalize=True))

# =========================================
# 3. TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# 4. TRAIN CALIBRATED MODEL
# =========================================
print("\nTraining calibrated Logistic Regression...")

base_model = LogisticRegression(max_iter=1000, random_state=42)
calibrated_model = CalibratedClassifierCV(
    base_model,
    method="sigmoid",
    cv=5,
    n_jobs=-1
)

calibrated_model.fit(X_train, y_train)

# Evaluate
y_pred = calibrated_model.predict(X_test)
y_prob = calibrated_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "brier_score": float(brier_score_loss(y_test, y_prob))
}

print(f"\nCalibrated Model Performance:")
print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
print(f"  Brier:    {metrics['brier_score']:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================================
# 5. SAVE
# =========================================
os.makedirs("models", exist_ok=True)
joblib.dump(calibrated_model, "models/academic_model.pkl")

# Save metrics for reporting
os.makedirs("reports", exist_ok=True)
with open("reports/academic_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Academic model and metrics saved.")
print("=" * 60)