"""
Student Success Prediction API
===============================
FastAPI service for predicting student pass/fail probability.
Supports single and batch predictions with risk levels and explanations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Union, List, Optional
import logging
from datetime import datetime

# Import prediction functions
from src.models.predict_system import predict_student, predict_batch

# =========================================
# LOGGING CONFIGURATION
# =========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================
# FASTAPI APP INITIALIZATION
# =========================================

app = FastAPI(
    title="Student Success Prediction API",
    description="Predicts student pass/fail probability using academic and behavioral data",
    version="1.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# PYDANTIC MODELS (INPUT VALIDATION)
# =========================================

class StudentInput(BaseModel):
    """Input schema for a single student."""
    term1_avg: float = Field(..., ge=0, le=20, description="Term 1 average (0-20)")
    term2_avg: float = Field(..., ge=0, le=20, description="Term 2 average (0-20)")
    seq5_score: float = Field(..., ge=0, le=20, description="Sequence 5 score (0-20)")
    attendance_percentage: float = Field(..., ge=0, le=100, description="Attendance % (0-100)")
    parental_support: int = Field(..., ge=0, le=1, description="Parental support (0 or 1)")
    study_hours_per_day: float = Field(..., ge=0, le=24, description="Daily study hours (0-24)")
    homework_completion: float = Field(..., ge=0, le=100, description="Homework completion % (0-100)")
    class_participation: float = Field(..., ge=0, le=5, description="Participation level (0-5)")
    extra_lessons: int = Field(..., ge=0, description="Number of extra lessons (>=0)")

    class Config:
        json_schema_extra = {
            "example": {
                "term1_avg": 12.5,
                "term2_avg": 11.0,
                "seq5_score": 10.5,
                "attendance_percentage": 85.0,
                "parental_support": 1,
                "study_hours_per_day": 2.0,
                "homework_completion": 90.0,
                "class_participation": 4.0,
                "extra_lessons": 1
            }
        }

class BatchInput(BaseModel):
    """Input schema for batch prediction."""
    students: List[StudentInput]

class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    prediction: str = Field(..., description="Pass or Fail")
    probability: float = Field(..., description="Overall pass probability (0-1)")
    risk_level: str = Field(..., description="Low/Medium/High Risk")
    academic_prob: float = Field(..., description="Academic model probability")
    behavioral_prob: float = Field(..., description="Behavioral model probability")
    confidence: str = Field(..., description="Confidence message")
    explanations: List[str] = Field(..., description="Reasons behind the prediction")

class BatchPredictionItem(BaseModel):
    """Item for batch response."""
    index: int
    status: str
    result: Optional[PredictionResponse] = None
    message: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[BatchPredictionItem]
    count: int
    timestamp: str

class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: str

# =========================================
# API ENDPOINTS
# =========================================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Student Success API (MVP Readiness 80+)",
        "version": "1.1.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(student: StudentInput):
    """Predict success for a single student."""
    try:
        student_data = student.model_dump()
        return predict_student(student_data)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Prediction failed", "detail": str(e)})

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_endpoint(batch: BatchInput):
    """Predict success for multiple students (e.g., an entire class)."""
    try:
        students_list = [s.model_dump() for s in batch.students]
        results = predict_batch(students_list)
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Batch prediction failed", "detail": str(e)})

@app.get("/info", tags=["Info"])
async def api_info():
    """Detailed API and feature information."""
    return {
        "api_name": "Student Success Prediction API",
        "version": "1.1.0",
        "features": {
            "risk_classification": "Low, Medium, High Risk based on probability",
            "explainability": "Rule-based reasons for prediction",
            "batch_support": "Single endpoint for multiple student predictions",
            "input_scale": "0-20 score scale for academic features"
        },
        "academic_thresholds": {
            "pass": "probability >= 0.5",
            "low_risk": "probability >= 0.7",
            "medium_risk": "0.45 - 0.7",
            "high_risk": "below 0.45"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
