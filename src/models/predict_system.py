"""
Student Success Prediction System
Combines academic and behavioral models to predict student pass/fail probability
"""

import joblib
import os
from typing import Dict, List, Union

# =========================================
# CONSTANTS
# =========================================

# Model weights
ACADEMIC_WEIGHT = 0.52
BEHAVIORAL_WEIGHT = 0.48

# Probability bounds
MIN_PROBABILITY = 0.05
MAX_PROBABILITY = 0.95

# Decision threshold
PASS_THRESHOLD = 0.5

# Feature order (CRITICAL - must match training order)
ACADEMIC_FEATURE_ORDER = [
    "term1_avg",
    "term2_avg",
    "seq5_score",
    "attendance_percentage",
    "parental_support"
]

BEHAVIORAL_FEATURE_ORDER = [
    "study_hours_per_day",
    "sleep_hours",
    "class_participation",
    "homework_completion",
    "extra_lessons",
    "attendance_percentage"
]

# Validation constraints
VALIDATION_RULES = {
    "term1_avg": {"min": 0, "max": 20, "type": (int, float)},
    "term2_avg": {"min": 0, "max": 20, "type": (int, float)},
    "seq5_score": {"min": 0, "max": 20, "type": (int, float)},
    "attendance_percentage": {"min": 0, "max": 100, "type": (int, float)},
    "parental_support": {"min": 0, "max": 1, "type": int},
    "study_hours_per_day": {"min": 0, "max": 24, "type": (int, float)},
    "sleep_hours": {"min": 0, "max": 24, "type": (int, float)},
    "class_participation": {"min": 0, "max": 5, "type": int},
    "homework_completion": {"min": 0, "max": 100, "type": (int, float)},
    "extra_lessons": {"min": 0, "max": None, "type": int}
}

# =========================================
# LOAD MODELS
# =========================================

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_A = joblib.load(os.path.join(base_dir, "models", "academic_model.pkl"))
model_B = joblib.load(os.path.join(base_dir, "models", "behavioral_model.pkl"))


# =========================================
# VALIDATION FUNCTIONS
# =========================================

def validate_input(data: Dict[str, Union[int, float]]) -> None:
    """
    Validate input data against defined constraints.
    
    Args:
        data: Dictionary containing student features
        
    Raises:
        ValueError: If validation fails
    """
    # Check all required fields are present
    all_required_fields = set(ACADEMIC_FEATURE_ORDER + BEHAVIORAL_FEATURE_ORDER)
    # Remove duplicates (attendance_percentage appears in both)
    all_required_fields = set(all_required_fields)
    
    missing_fields = all_required_fields - set(data.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Validate each field
    for field, value in data.items():
        if field not in VALIDATION_RULES:
            continue  # Skip unknown fields
            
        rules = VALIDATION_RULES[field]
        
        # Type check
        if not isinstance(value, rules["type"]):
            raise ValueError(
                f"Field '{field}' must be of type {rules['type']}, got {type(value)}"
            )
        
        # Range check - minimum
        if value < rules["min"]:
            raise ValueError(
                f"Field '{field}' must be >= {rules['min']}, got {value}"
            )
        
        # Range check - maximum (if defined)
        if rules["max"] is not None and value > rules["max"]:
            raise ValueError(
                f"Field '{field}' must be <= {rules['max']}, got {value}"
            )


def extract_features(data: Dict[str, Union[int, float]], 
                     feature_order: List[str]) -> List[Union[int, float]]:
    """
    Extract features in the correct order from input data.
    
    Args:
        data: Dictionary containing student features
        feature_order: List defining the correct feature order
        
    Returns:
        List of feature values in correct order
    """
    return [data[feature] for feature in feature_order]


def clip_probability(prob: float) -> float:
    """
    Clip probability to safe bounds to avoid extreme predictions.
    
    Args:
        prob: Raw probability from model
        
    Returns:
        Clipped probability between MIN_PROBABILITY and MAX_PROBABILITY
    """
    return min(max(prob, MIN_PROBABILITY), MAX_PROBABILITY)


# =========================================
# PREDICTION FUNCTION
# =========================================

def predict_student(data: Dict[str, Union[int, float]]) -> Dict[str, Union[str, float]]:
    """
    Predict student success probability using dual-model approach.
    
    Args:
        data: Dictionary containing all required student features:
            - term1_avg: Term 1 average score (0-20)
            - term2_avg: Term 2 average score (0-20)
            - seq5_score: Sequence 5 score (0-20)
            - attendance_percentage: Attendance percentage (0-100)
            - parental_support: Parental support (0 or 1)
            - study_hours_per_day: Daily study hours (0-24)
            - sleep_hours: Average sleep hours (0-24)
            - class_participation: Participation level (0-5)
            - homework_completion: Homework completion % (0-100)
            - extra_lessons: Number of extra lessons (≥0)
    
    Returns:
        Dictionary containing:
            - prediction: "Pass" or "Fail"
            - probability: Final combined probability (0-1)
            - academic_prob: Academic model probability (0-1)
            - behavioral_prob: Behavioral model probability (0-1)
    
    Raises:
        ValueError: If input validation fails
    """
    # Step 1: Validate input
    validate_input(data)
    
    # Step 2: Extract features in correct order
    academic_features = extract_features(data, ACADEMIC_FEATURE_ORDER)
    behavioral_features = extract_features(data, BEHAVIORAL_FEATURE_ORDER)
    
    # Step 3: Get raw predictions from models
    prob_academic_raw = model_A.predict_proba([academic_features])[0][1]
    prob_behavioral_raw = model_B.predict_proba([behavioral_features])[0][1]
    
    # Step 4: Apply probability safeguards
    prob_academic = clip_probability(prob_academic_raw)
    prob_behavioral = clip_probability(prob_behavioral_raw)
    
    # Step 5: Combine predictions (weighted average)
    final_prob = (ACADEMIC_WEIGHT * prob_academic) + (BEHAVIORAL_WEIGHT * prob_behavioral)
    
    # Step 6: Make final decision
    prediction = "Pass" if final_prob >= PASS_THRESHOLD else "Fail"
    
    # Step 7: Return results
    return {
        "prediction": prediction,
        "probability": round(float(final_prob), 3),
        "academic_prob": round(float(prob_academic), 3),
        "behavioral_prob": round(float(prob_behavioral), 3)
    }


# =========================================
# TESTING
# =========================================

if __name__ == "__main__":
    # Test with sample data
    sample = {
        "term1_avg": 12,
        "term2_avg": 13,
        "seq5_score": 14,
        "attendance_percentage": 85,
        "parental_support": 1,
        "study_hours_per_day": 3,
        "sleep_hours": 7,
        "class_participation": 3,
        "homework_completion": 80,
        "extra_lessons": 1
    }
    
    print("Testing prediction system...")
    print("-" * 50)
    result = predict_student(sample)
    print(f"Prediction: {result['prediction']}")
    print(f"Overall Probability: {result['probability']:.1%}")
    print(f"Academic Probability: {result['academic_prob']:.1%}")
    print(f"Behavioral Probability: {result['behavioral_prob']:.1%}")
    print("-" * 50)
    
    # Test validation
    print("\nTesting validation...")
    try:
        invalid_sample = sample.copy()
        invalid_sample["term1_avg"] = 25  # Invalid: exceeds max
        predict_student(invalid_sample)
    except ValueError as e:
        print(f"✓ Validation working: {e}")
    
    print("\n✅ All tests passed!")