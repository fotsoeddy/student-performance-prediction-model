"""
Student Success Prediction System
==================================
Combines calibrated academic and behavioral models using weighted ensemble.
Includes risk level classification, rule-based explanations, and batch support.
"""

import joblib
import os
import numpy as np
from typing import Dict, List, Union, Any

# =========================================
# CONSTANTS & CONFIGURATION
# =========================================

ACADEMIC_WEIGHT = 0.7
BEHAVIORAL_WEIGHT = 0.3
PASS_THRESHOLD = 0.5

# Risk Level Thresholds (based on pass probability)
# low_risk: >= 0.70
# medium_risk: 0.45 - 0.69
# high_risk: < 0.45
RISK_LEVELS = {
    "LOW": {"threshold": 0.70, "label": "Low Risk"},
    "MEDIUM": {"threshold": 0.45, "label": "Medium Risk"},
    "HIGH": {"threshold": 0.00, "label": "High Risk"}
}

# Academic feature order
ACADEMIC_FEATURE_ORDER = [
    "term1_avg",
    "term2_avg",
    "seq5_score",
    "attendance_percentage",
    "parental_support"
]

# Raw behavioral inputs
BEHAVIORAL_RAW_FEATURES = [
    "study_hours_per_day",
    "homework_completion"
]

# Validation constraints
VALIDATION_RULES = {
    "term1_avg":             {"min": 0,   "max": 20,  "type": (int, float)},
    "term2_avg":             {"min": 0,   "max": 20,  "type": (int, float)},
    "seq5_score":            {"min": 0,   "max": 20,  "type": (int, float)},
    "attendance_percentage": {"min": 0,   "max": 100, "type": (int, float)},
    "parental_support":      {"min": 0,   "max": 1,   "type": int},
    "study_hours_per_day":   {"min": 0,   "max": 24,  "type": (int, float)},
    "homework_completion":   {"min": 0,   "max": 100, "type": (int, float)},
    "class_participation":   {"min": 0,   "max": 5,   "type": (int, float)},
    "extra_lessons":         {"min": 0,   "max": None,"type": int},
}

# =========================================
# LOAD MODELS
# =========================================

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

try:
    academic_model    = joblib.load(os.path.join(base_dir, "models", "academic_model.pkl"))
    behavioral_model  = joblib.load(os.path.join(base_dir, "models", "behavioral_model.pkl"))
    engagement_scaler = joblib.load(os.path.join(base_dir, "models", "engagement_scaler.pkl"))
except Exception as e:
    print(f"⚠️ Warning: Some models could not be loaded. Ensure training is complete. Error: {e}")
    academic_model = None
    behavioral_model = None
    engagement_scaler = None


# =========================================
# HELPERS
# =========================================

def validate_input(data: Dict[str, Union[int, float]]) -> None:
    """Validate all required fields exist and are within allowed ranges."""
    required = set(ACADEMIC_FEATURE_ORDER + BEHAVIORAL_RAW_FEATURES +
                   ["attendance_percentage", "extra_lessons", "class_participation"])

    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    for field, value in data.items():
        if field not in VALIDATION_RULES:
            continue
        rules = VALIDATION_RULES[field]

        if not isinstance(value, rules["type"]):
            # Allow int to be treated as float for scores
            if isinstance(value, int) and float in rules["type"]:
                pass
            else:
                raise ValueError(f"'{field}' must be {rules['type']}, got {type(value)}")
        
        if value < rules["min"]:
            raise ValueError(f"'{field}' must be >= {rules['min']}, got {value}")
        if rules["max"] is not None and value > rules["max"]:
            raise ValueError(f"'{field}' must be <= {rules['max']}, got {value}")


def classify_risk_level(prob: float) -> str:
    """Classify risk level based on pass probability."""
    if prob >= RISK_LEVELS["LOW"]["threshold"]:
        return RISK_LEVELS["LOW"]["label"]
    elif prob >= RISK_LEVELS["MEDIUM"]["threshold"]:
        return RISK_LEVELS["MEDIUM"]["label"]
    else:
        return RISK_LEVELS["HIGH"]["label"]


def generate_explanations(data: Dict[str, Any], prob: float) -> List[str]:
    """Rule-based explanations for the prediction."""
    reasons = []
    
    # 1. Attendance checks
    if data["attendance_percentage"] < 60:
        reasons.append("Critical: Very low attendance (below 60%)")
    elif data["attendance_percentage"] < 80:
        reasons.append("Low attendance (below 80%)")
    
    # 2. Academic performance
    avg_score = (data["term1_avg"] + data["term2_avg"]) / 2
    if avg_score < 10:
        reasons.append("Weak academic foundation: Average scores below 10/20")
    if data["seq5_score"] < 10:
        reasons.append("Downward trend: Sequence 5 score is failing")

    # 3. Effort/Support
    if data["study_hours_per_day"] < 1.5:
        reasons.append("Insufficient study time (below 1.5h/day)")
    if data["homework_completion"] < 50:
        reasons.append("Poor homework completion (below 50%)")
    if data["parental_support"] == 0:
        reasons.append("Lack of parental support at home")

    # If probability is high but some flags exist
    if prob >= 0.7 and not reasons:
        reasons.append("Consistent performance across all metrics")
    elif not reasons:
        reasons.append("Performances are borderline across multiple factors")
        
    return reasons


def compute_engagement_score(data: Dict[str, Union[int, float]]) -> float:
    """Normalize study hours and homework completion into one score."""
    if engagement_scaler is None:
        return 0.5 # Default middle ground if model not loaded
    scaled = engagement_scaler.transform(
        [[data["study_hours_per_day"], data["homework_completion"]]]
    )
    return float((scaled[0][0] + scaled[0][1]) / 2)


# =========================================
# CORE PREDICTION FUNCTIONS
# =========================================

def predict_student(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict success for a single student.
    """
    if academic_model is None or behavioral_model is None:
        raise RuntimeError("Models not loaded. Please run training pipeline first.")

    # 1. Validate
    validate_input(data)

    # 2. Academic features
    academic_features = [data[f] for f in ACADEMIC_FEATURE_ORDER]

    # 3. Behavioral features
    engagement_score = compute_engagement_score(data)
    behavioral_features = [
        engagement_score,
        data["attendance_percentage"],
        data["extra_lessons"],
        data["class_participation"]
    ]

    # 4. Probabilities
    prob_academic   = academic_model.predict_proba([academic_features])[0][1]
    prob_behavioral = behavioral_model.predict_proba([behavioral_features])[0][1]

    # 5. Ensemble
    final_prob = (ACADEMIC_WEIGHT * prob_academic) + (BEHAVIORAL_WEIGHT * prob_behavioral)

    # 6. Results
    risk_level = classify_risk_level(final_prob)
    prediction = "Pass" if final_prob >= PASS_THRESHOLD else "Fail"
    explanations = generate_explanations(data, final_prob)
    
    # Confidence message
    if final_prob > 0.85 or final_prob < 0.15:
        message = "High confidence prediction"
    elif final_prob > 0.65 or final_prob < 0.35:
        message = "Moderate confidence prediction"
    else:
        message = "Low confidence (borderline case)"

    return {
        "prediction":      prediction,
        "probability":     round(float(final_prob), 3),
        "risk_level":      risk_level,
        "academic_prob":   round(float(prob_academic), 3),
        "behavioral_prob": round(float(prob_behavioral), 3),
        "confidence":      message,
        "explanations":    explanations
    }


def predict_batch(students: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict success for a list of students.
    """
    results = []
    for i, student in enumerate(students):
        try:
            results.append({
                "index": i,
                "status": "success",
                "result": predict_student(student)
            })
        except Exception as e:
            results.append({
                "index": i,
                "status": "error",
                "message": str(e)
            })
    return results


# =========================================
# CLI TEST HOOK
# =========================================

if __name__ == "__main__":
    sample = {
        "term1_avg": 8.5,
        "term2_avg": 9.0,
        "seq5_score": 7.5,
        "attendance_percentage": 55,
        "parental_support": 0,
        "study_hours_per_day": 1.0,
        "homework_completion": 40,
        "class_participation": 1,
        "extra_lessons": 0
    }

    try:
        res = predict_student(sample)
        import json
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Prediction failed: {e}")
