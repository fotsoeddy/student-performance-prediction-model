# =========================================
# ACADEMIC MODEL TRAINING WITH CALIBRATION
# =========================================
"""
Trains the academic model with probability calibration.
Uses CalibratedClassifierCV to produce well-calibrated probabilities.
"""

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

# =========================================
# 1. LOAD DATA
# =========================================
print("=" * 60)
print("ACADEMIC MODEL TRAINING WITH CALIBRATION")
print("=" * 60)

df = pd.read_csv("data/processed/aligned_student_data.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

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

print(f"\nClass distribution:")
print(y.value_counts(normalize=True))

# =========================================
# 3. TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# =========================================
# 4. TRAIN BASE MODEL (UNCALIBRATED)
# =========================================
print("\n" + "-" * 60)
print("PHASE 1: Training Base Model (Uncalibrated)")
print("-" * 60)

base_model = LogisticRegression(max_iter=1000, random_state=42)
base_model.fit(X_train, y_train)

# Evaluate uncalibrated model
y_pred_uncal = base_model.predict(X_test)
y_prob_uncal = base_model.predict_proba(X_test)[:, 1]

acc_uncal = accuracy_score(y_test, y_pred_uncal)
auc_uncal = roc_auc_score(y_test, y_prob_uncal)
brier_uncal = brier_score_loss(y_test, y_prob_uncal)

print(f"\nUncalibrated Model Performance:")
print(f"  Accuracy: {acc_uncal:.4f}")
print(f"  ROC AUC: {auc_uncal:.4f}")
print(f"  Brier Score: {brier_uncal:.4f} (lower is better)")
print(f"  Train Accuracy: {base_model.score(X_train, y_train):.4f}")
print(f"  Test Accuracy: {base_model.score(X_test, y_test):.4f}")

# =========================================
# 5. TRAIN CALIBRATED MODEL
# =========================================
print("\n" + "-" * 60)
print("PHASE 2: Training Calibrated Model")
print("-" * 60)

# Create calibrated classifier
calibrated_model = CalibratedClassifierCV(
    base_model,
    method="sigmoid",  # Platt scaling
    cv=5,
    n_jobs=-1
)

print("\nCalibrating model with 5-fold cross-validation...")
calibrated_model.fit(X_train, y_train)

# Evaluate calibrated model
y_pred_cal = calibrated_model.predict(X_test)
y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

acc_cal = accuracy_score(y_test, y_pred_cal)
auc_cal = roc_auc_score(y_test, y_prob_cal)
brier_cal = brier_score_loss(y_test, y_prob_cal)

print(f"\nCalibrated Model Performance:")
print(f"  Accuracy: {acc_cal:.4f}")
print(f"  ROC AUC: {auc_cal:.4f}")
print(f"  Brier Score: {brier_cal:.4f} (lower is better)")

# =========================================
# 6. COMPARISON
# =========================================
print("\n" + "=" * 60)
print("CALIBRATION IMPACT")
print("=" * 60)

print(f"\nAccuracy:     {acc_uncal:.4f} → {acc_cal:.4f} (Δ {acc_cal - acc_uncal:+.4f})")
print(f"ROC AUC:      {auc_uncal:.4f} → {auc_cal:.4f} (Δ {auc_cal - auc_uncal:+.4f})")
print(f"Brier Score:  {brier_uncal:.4f} → {brier_cal:.4f} (Δ {brier_cal - brier_uncal:+.4f})")

# Probability distribution analysis
print(f"\nProbability Distribution (Test Set):")
print(f"  Uncalibrated - Min: {y_prob_uncal.min():.4f}, Max: {y_prob_uncal.max():.4f}, Std: {y_prob_uncal.std():.4f}")
print(f"  Calibrated   - Min: {y_prob_cal.min():.4f}, Max: {y_prob_cal.max():.4f}, Std: {y_prob_cal.std():.4f}")

# =========================================
# 7. SAVE CALIBRATED MODEL
# =========================================
print("\n" + "-" * 60)
print("Saving calibrated model...")
print("-" * 60)

joblib.dump(calibrated_model, "models/academic_model.pkl")

print("\n✅ Calibrated academic model saved as 'academic_model.pkl'")
print("=" * 60)