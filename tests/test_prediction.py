import pytest
from src.models.predict_system import predict_student, predict_batch, validate_input

def test_single_prediction_pass(sample_student):
    """Test that a high-performing student passes."""
    try:
        result = predict_student(sample_student)
        assert result["prediction"] == "Pass"
        assert result["probability"] >= 0.5
        assert "risk_level" in result
        assert "explanations" in result
    except RuntimeError as e:
        pytest.skip(f"Models not loaded: {e}")

def test_single_prediction_fail(failing_student):
    """Test that an at-risk student fails."""
    try:
        result = predict_student(failing_student)
        assert result["prediction"] == "Fail"
        assert result["probability"] < 0.5
        assert result["risk_level"] == "High Risk"
        assert len(result["explanations"]) > 0
    except RuntimeError as e:
        pytest.skip(f"Models not loaded: {e}")

def test_batch_prediction(sample_student, failing_student):
    """Test batch prediction logic."""
    students = [sample_student, failing_student]
    try:
        results = predict_batch(students)
        assert len(results) == 2
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "success"
    except RuntimeError as e:
        pytest.skip(f"Models not loaded: {e}")

def test_invalid_input_range():
    """Test validation for out-of-range values."""
    invalid_data = {
        "term1_avg": 25.0, # Max is 20
        "term2_avg": 10.0,
        "seq5_score": 10.0,
        "attendance_percentage": 70.0,
        "parental_support": 0,
        "study_hours_per_day": 2.5,
        "homework_completion": 65.0,
        "class_participation": 2.5,
        "extra_lessons": 0
    }
    with pytest.raises(ValueError, match="must be <= 20"):
        validate_input(invalid_data)

def test_missing_input():
    """Test validation for missing fields."""
    incomplete_data = {
        "term1_avg": 10.0
    }
    with pytest.raises(ValueError, match="Missing required fields"):
        validate_input(incomplete_data)

def test_risk_level_logic():
    """Manual check for risk level classification ranges."""
    from src.models.predict_system import classify_risk_level
    assert classify_risk_level(0.85) == "Low Risk"
    assert classify_risk_level(0.55) == "Medium Risk"
    assert classify_risk_level(0.30) == "High Risk"
