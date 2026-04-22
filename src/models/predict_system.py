"""
Student Success Prediction System
Combines calibrated academic and behavioral models using weighted ensemble.
Uses engagement score to reduce feature redundancy.
"""

import joblib
import os
import numpy as np
from typing import Dict, List, Union

# =========================================
# CONSTANTS
# =========================================

ACADEMIC_WEIGHT = 0.7
BEHAVIORAL_WEIGHT = 0.3
PASS_THRESHOLD = 0.5

# Academic feature order (must match training order)
ACADEMIC_FEATURE_ORDER = [
    "term1_avg",
    "term2_avg",
    "seq5_score",
    "attendance_percentage",
    "parental_support"
]

# Raw behavioral inputs (used to compute engagement_score)
BEHAVIORAL_RAW_FEATURES = [
    "study_hours_per_day",
    "homework_completion"
]

# Validation constraints for all input fields
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

academic_model    = joblib.load(os.path.join(base_dir, "models", "academic_model.pkl"))
behavioral_model  = joblib.load(os.path.join(base_dir, "models", "behavioral_model.pkl"))
engagement_scaler = joblib.load(os.path.join(base_dir, "models", "engagement_scaler.pkl"))


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
            raise ValueError(f"'{field}' must be {rules['type']}, got {type(value)}")
        if value < rules["min"]:
            raise ValueError(f"'{field}' must be >= {rules['min']}, got {value}")
        if rules["max"] is not None and value > rules["max"]:
            raise ValueError(f"'{field}' must be <= {rules['max']}, got {value}")


def extract_features(data: Dict[str, Union[int, float]],
                     feature_order: List[str]) -> List[Union[int, float]]:
    """Return feature values in the exact order the model expects."""
    return [data[f] for f in feature_order]


def compute_engagement_score(data: Dict[str, Union[int, float]]) -> float:
    """
    Combine study_hours_per_day and homework_completion into a single
    normalized engagement score (0–1) using the fitted MinMaxScaler.
    """
    scaled = engagement_scaler.transform(
        [[data["study_hours_per_day"], data["homework_completion"]]]
    )
    return float((scaled[0][0] + scaled[0][1]) / 2)


# =========================================
# PREDICTION FUNCTION
# =========================================

def predict_student(data: Dict[str, Union[int, float]]) -> Dict[str, Union[str, float]]:
    """
    Predict student pass/fail probability.

    Required input fields:
        term1_avg             – Term 1 average score       (0–20)
        term2_avg             – Term 2 average score       (0–20)
        seq5_score            – Sequence 5 score           (0–20)
        attendance_percentage – Attendance percentage      (0–100)
        parental_support      – Parental support           (0 or 1)
        study_hours_per_day   – Avg daily study hours      (0–24)
        homework_completion   – Homework completion %      (0–100)
        class_participation   – Participation level        (0–5)
        extra_lessons         – Number of extra lessons    (≥ 0)

    Returns:
        {
            "prediction":     "Pass" | "Fail",
            "probability":    float,
            "academic_prob":  float,
            "behavioral_prob": float
        }
    """
    # 1. Validate
    validate_input(data)

    # 2. Academic features
    academic_features = extract_features(data, ACADEMIC_FEATURE_ORDER)

    # 3. Behavioral features (engagement score replaces study + homework)
    engagement_score = compute_engagement_score(data)
    behavioral_features = [
        engagement_score,
        data["attendance_percentage"],
        data["extra_lessons"],
        data["class_participation"]
    ]

    # 4. Get calibrated probabilities
    prob_academic   = academic_model.predict_proba([academic_features])[0][1]
    prob_behavioral = behavioral_model.predict_proba([behavioral_features])[0][1]

    # 5. Weighted ensemble
    final_prob = (ACADEMIC_WEIGHT * prob_academic) + (BEHAVIORAL_WEIGHT * prob_behavioral)

    # 6. Decision
    prediction = "Pass" if final_prob >= PASS_THRESHOLD else "Fail"

    return {
        "prediction":      prediction,
        "probability":     round(float(final_prob), 3),
        "academic_prob":   round(float(prob_academic), 3),
        "behavioral_prob": round(float(prob_behavioral), 3)
    }


# =========================================
# QUICK TEST
# =========================================

if __name__ == "__main__":
    sample = {
        "term1_avg": 10,
        "term2_avg": 10,
        "seq5_score": 10,
        "attendance_percentage": 70,
        "parental_support": 0,
        "study_hours_per_day": 2.5,
        "homework_completion": 65,
        "class_participation": 2,
        "extra_lessons": 0
    }

    result = predict_student(sample)
    print(f"Prediction:          {result['prediction']}")
    print(f"Overall Probability: {result['probability']:.1%}")
    print(f"Academic Prob:       {result['academic_prob']:.1%}")
    print(f"Behavioral Prob:     {result['behavioral_prob']:.1%}")
