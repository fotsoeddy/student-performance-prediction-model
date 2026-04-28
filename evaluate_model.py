"""
Model Evaluation Script
=======================
Evaluates the retrained student performance prediction models.
Produces metrics (accuracy, precision, recall, F1, ROC-AUC) and saves results.
Focuses on recall for the Fail/High Risk class.
"""

import pandas as pd
import joblib
import numpy as np
import json
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Constants
ACADEMIC_WEIGHT = 0.7
BEHAVIORAL_WEIGHT = 0.3

def load_data():
    """Load the processed datasets."""
    # We use a mix of both datasets for a holistic evaluation
    # For simplicity in this script, we'll evaluate on the aligned_student_data.csv (Academic focus)
    # and aligned_kaggle_data_full.csv (Behavioral focus) then show ensemble performance.
    
    academic_df = pd.read_csv("data/processed/aligned_student_data.csv")
    behavioral_df = pd.read_csv("data/processed/aligned_kaggle_data_full.csv")
    return academic_df, behavioral_df

def evaluate_ensemble():
    """Evaluate the combined ensemble model performance."""
    print("=" * 60)
    print("STUDENT SUCCESS PREDICTION - MODEL EVALUATION")
    print("=" * 60)
    
    # Check if models exist
    if not os.path.exists("models/academic_model.pkl") or not os.path.exists("models/behavioral_model.pkl"):
        print("❌ Error: Models not found. Run train_models.py first.")
        return

    academic_model = joblib.load("models/academic_model.pkl")
    behavioral_model = joblib.load("models/behavioral_model.pkl")
    engagement_scaler = joblib.load("models/engagement_scaler.pkl")
    
    # We'll use the combined student data for testing
    df = pd.read_csv("data/processed/aligned_student_data.csv")
    
    # 1. Prepare features
    academic_features = ["Term1_avg", "Term2_avg", "Seq5_score", "Attendance_percentage", "Parental_support"]
    
    # Compute engagement for behavioral model input
    # In real world evaluation, we'd use a held-out test set.
    # Here we evaluate on the full processed set as a sanity check of the pipeline.
    
    X_acad = df[academic_features]
    y_true = df["Pass"]
    
    # For behavioral, we need Engagement_score
    scaled = engagement_scaler.transform(df[["Study_hours_per_day", "Homework_completion"]])
    engagement = (scaled[:, 0] + scaled[:, 1]) / 2
    
    X_beh = pd.DataFrame({
        "Engagement_score": engagement,
        "Attendance_percentage": df["Attendance_percentage"],
        "Extra_lessons": df["Extra_lessons"],
        "Class_participation": df["Class_participation"]
    })
    
    # 2. Get probabilities
    prob_acad = academic_model.predict_proba(X_acad)[:, 1]
    prob_beh = behavioral_model.predict_proba(X_beh)[:, 1]
    
    # 3. Ensemble
    final_prob = (ACADEMIC_WEIGHT * prob_acad) + (BEHAVIORAL_WEIGHT * prob_beh)
    y_pred = (final_prob >= 0.5).astype(int)
    
    # 4. Compute Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, final_prob)),
    }
    
    # Special focus on Fail recall (Detecting students at risk)
    # y_true=0 is Fail
    fail_recall = float(recall_score(y_true, y_pred, pos_label=0))
    metrics["fail_recall"] = fail_recall
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 5. Output
    print(f"\nOverall Ensemble Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Recall (Fail class): {metrics['fail_recall']:.4f} 🎯 (CRITICAL for Early Warning)")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=["Fail", "Pass"]))
    
    # 6. Save results
    os.makedirs("reports", exist_ok=True)
    with open("reports/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Generate MD report
    with open("reports/model_evaluation.md", "w") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        f.write("## Performance Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("| --- | --- |\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision | {metrics['precision']:.4f} |\n")
        f.write(f"| Recall (Pass) | {metrics['recall']:.4f} |\n")
        f.write(f"| Recall (Fail/Risk) | {metrics['fail_recall']:.4f} |\n")
        f.write(f"| F1-Score | {metrics['f1_score']:.4f} |\n")
        f.write(f"| ROC-AUC | {metrics['roc_auc']:.4f} |\n\n")
        f.write("## Why Fail Recall Matters\n")
        f.write("In an Early Warning System, **Recall for the Fail class** is more important than overall accuracy. ")
        f.write("It measures our ability to correctly identify students who are actually at risk. ")
        f.write("Higher recall means fewer students slip through the cracks without interventions.\n\n")
        f.write("## Confusion Matrix\n")
        f.write("```\n")
        f.write(str(cm))
        f.write("\n```\n")
        
    print("\n✅ Evaluation documents saved to reports/ directory.")

if __name__ == "__main__":
    evaluate_ensemble()
