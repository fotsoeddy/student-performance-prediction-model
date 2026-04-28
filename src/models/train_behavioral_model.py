# =========================================
# BEHAVIORAL MODEL TRAINING WITH CALIBRATION
# =========================================
"""
Trains the behavioral model (XGBoost) with probability calibration.
Uses engineered Engagement Score to reduce feature redundancy.
Saves metrics to JSON for reporting.
"""

import pandas as pd
import joblib
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# =========================================
# 1. LOAD DATA
# =========================================
print("=" * 60)
print("BEHAVIORAL MODEL TRAINING WITH CALIBRATION")
print("=" * 60)

DATA_PATH = "data/processed/aligned_kaggle_data_full.csv"
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found. Run kaggle_dataset_alignment.py first.")
    exit(1)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape: {df.shape}")

# =========================================
# 2. FEATURE ENGINEERING
# =========================================
print("\nCreating Engagement Score...")
scaler = MinMaxScaler()
df[["Study_hours_scaled", "Homework_scaled"]] = scaler.fit_transform(
    df[["Study_hours_per_day", "Homework_completion"]]
)
df["Engagement_score"] = (df["Study_hours_scaled"] + df["Homework_scaled"]) / 2

# Target & Features
TARGET = "Pass"
FEATURES = [
    "Engagement_score",
    "Attendance_percentage",
    "Extra_lessons",
    "Class_participation"
]

X = df[FEATURES]
y = df[TARGET]

# Imputation guard
if X.isnull().any().any():
    X = X.fillna(X.median())

# =========================================
# 3. TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# 4. TRAINING
# =========================================
print("\nTraining calibrated XGBoost...")

base_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

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

# =========================================
# 5. SAVE
# =========================================
os.makedirs("models", exist_ok=True)
joblib.dump(calibrated_model, "models/behavioral_model.pkl")
joblib.dump(scaler, "models/engagement_scaler.pkl")

# Save metrics
os.makedirs("reports", exist_ok=True)
with open("reports/behavioral_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Behavioral model, scaler, and metrics saved.")
print("=" * 60)