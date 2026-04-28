import pytest
import os
import json

@pytest.fixture
def sample_student():
    """Valid sample student data."""
    return {
        "term1_avg": 15.0,
        "term2_avg": 14.5,
        "seq5_score": 14.0,
        "attendance_percentage": 90.0,
        "parental_support": 1,
        "study_hours_per_day": 3.0,
        "homework_completion": 85.0,
        "class_participation": 4.5,
        "extra_lessons": 2
    }

@pytest.fixture
def failing_student():
    """At-risk student data."""
    return {
        "term1_avg": 5.0,
        "term2_avg": 6.0,
        "seq5_score": 4.5,
        "attendance_percentage": 40.0,
        "parental_support": 0,
        "study_hours_per_day": 0.5,
        "homework_completion": 20.0,
        "class_participation": 1.0,
        "extra_lessons": 0
    }
