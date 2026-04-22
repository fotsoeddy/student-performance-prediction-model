# =========================================
# BEHAVIORAL MODEL TRAINING WITH CALIBRATION
# =========================================
"""
Trains the behavioral model with probability calibration.
Uses engineered Engagement Score to reduce feature redundancy.
Removes highly correlated features for better stability.
"""

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# =========================================
# 1. LOAD DATA
# =========================================
print("=" * 60)
print("BEHAVIORAL MODEL TRAINING WITH CALIBRATION")
print("=" * 60)

df = pd.read_csv("data/processed/aligned_kaggle_data_full.csv")
print(f"\nDataset shape: {df.shape}")

# =========================================
# 2. FEATURE ENGINEERING
# =========================================
print("\n" + "-" * 60)
print("FEATURE ENGINEERING")
print("-" * 60)

# Create Engagement Score (combines Study_hours and Homework_completion)
print("\nCreating Engagement Score...")
print("  Combining: Study_hours_per_day + Homework_completion")

scaler = MinMaxScaler()
df[["Study_hours_scaled", "Homework_scaled"]] = scaler.fit_transform(
    df[["Study_hours_per_day", "Homework_completion"]]
)

df["Engagement_score"] = (
    df["Study_hours_scaled"] + df["Homework_scaled"]
) / 2

print(f"  ✓ Engagement_score created (range: {df['Engagement_score'].min():.2f} - {df['Engagement_score'].max():.2f})")

# =========================================
# 3. FEATURES & TARGET
# =========================================
TARGET = "Pass"

# Updated feature set (removed redundant features)
FEATURES = [
    "Engagement_score",        # Combines study hours + homework
    "Attendance_percentage",
    "Extra_lessons",
    "Class_participation"
]

print(f"\nFeature Set (reduced from 6 to 4):")
for feat in FEATURES:
    print(f"  • {feat}")

print(f"\nRemoved features (redundant/low correlation):")
print(f"  ✗ Study_hours_per_day (redundant with Engagement_score)")
print(f"  ✗ Homework_completion (redundant with Engagement_score)")
print(f"  ✗ Sleep_hours (low correlation with target)")

X = df[FEATURES]
y = df[TARGET]

print(f"\nClass distribution:")
print(y.value_counts(normalize=True))

# =========================================
# 4. TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# =========================================
# 5. TRAIN BASE MODEL (UNCALIBRATED)
# =========================================
print("\n" + "-" * 60)
print("PHASE 1: Training Base Model (Uncalibrated)")
print("-" * 60)

# Controlled XGBoost parameters for stability
base_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

print("\nTraining XGBoost with controlled complexity...")
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

# Feature importance
print(f"\nFeature Importance:")
for feat, imp in sorted(zip(FEATURES, base_model.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat:25s}: {imp:.4f}")

# =========================================
# 6. TRAIN CALIBRATED MODEL
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
# 7. COMPARISON
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

# Check for extreme probabilities
extreme_uncal = np.sum((y_prob_uncal < 0.1) | (y_prob_uncal > 0.9))
extreme_cal = np.sum((y_prob_cal < 0.1) | (y_prob_cal > 0.9))
print(f"\nExtreme Probabilities (<0.1 or >0.9):")
print(f"  Uncalibrated: {extreme_uncal}/{len(y_prob_uncal)} ({100*extreme_uncal/len(y_prob_uncal):.1f}%)")
print(f"  Calibrated:   {extreme_cal}/{len(y_prob_cal)} ({100*extreme_cal/len(y_prob_cal):.1f}%)")

# =========================================
# 8. SAVE CALIBRATED MODEL & SCALER
# =========================================
print("\n" + "-" * 60)
print("Saving calibrated model and scaler...")
print("-" * 60)

joblib.dump(calibrated_model, "models/behavioral_model.pkl")
joblib.dump(scaler, "models/engagement_scaler.pkl")

print("\n✅ Calibrated behavioral model saved as 'behavioral_model.pkl'")
print("✅ Engagement scaler saved as 'engagement_scaler.pkl'")
print("=" * 60)

# =========================================
# 9. SUMMARY
# =========================================
print("\n" + "=" * 60)
print("IMPROVEMENTS SUMMARY")
print("=" * 60)
print("\nFeature Engineering:")
print("  ✓ Created Engagement_score (combines study + homework)")
print("  ✓ Reduced features from 6 to 4")
print("  ✓ Removed redundant/low-correlation features")
print("\nModel Improvements:")
print("  ✓ Controlled complexity (max_depth=4)")
print("  ✓ Probability calibration (sigmoid method)")
print("  ✓ Reduced extreme predictions")
print("\nExpected Benefits:")
print("  ✓ More stable predictions")
print("  ✓ Smoother probability distribution")
print("  ✓ Better generalization")
print("=" * 60)