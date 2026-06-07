import pandas as pd
import joblib
import numpy as np
import os

# Paths
DATA_PATH = "data/processed/aligned_kaggle_data_full.csv"
MODEL_PATH = "models/academic_model.pkl"

if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
    print("Missing data or model files.")
    exit(1)

# 1. Load Data
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("PARENTAL SUPPORT ANALYSIS (V2)")
print("=" * 60)

# Distribution
print("\n1. Distribution of Parental_support in Training Data:")
print(df["Parental_support"].value_counts(normalize=True).sort_index())
print("\nMean Pass rate per Support Level:")
print(df.groupby("Parental_support")["Pass"].mean())

# Correlation
print("\n2. Correlation with Target (Pass):")
corr = df[["Parental_support", "Pass"]].corr().iloc[0, 1]
print(f"Correlation: {corr:.4f}")

# 2. Load Model
model = joblib.load(MODEL_PATH)

print("\n3. Academic Model (Logistic Regression) Coefficients:")
features = ["Term1_avg", "Term2_avg", "Seq5_score", "Attendance_percentage", "Parental_support"]

if hasattr(model, "calibrated_classifiers_"):
    coeffs = []
    for clf in model.calibrated_classifiers_:
        # Try both estimator (newer) and base_estimator (older)
        if hasattr(clf, "estimator"):
            base = clf.estimator
        elif hasattr(clf, "base_estimator"):
            base = clf.base_estimator
        else:
            base = None
        
        if base and hasattr(base, "coef_"):
            coeffs.append(base.coef_[0])
    
    if coeffs:
        avg_coeffs = np.mean(coeffs, axis=0)
        # Calculate Absolute Importance
        importance = np.abs(avg_coeffs)
        rel_importance = (importance / np.sum(importance)) * 100
        
        for feat, val, rel in zip(features, avg_coeffs, rel_importance):
            print(f"  {feat:25s}: Coeff={val:+.4f} | Rel. Importance={rel:5.2f}%")
    else:
        print("Could not find base estimators with coefficients.")
else:
    print("Model structure is not a CalibratedClassifierCV.")
