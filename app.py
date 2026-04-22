"""
Student Success Prediction API
FastAPI service for predicting student pass/fail probability
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Union
import logging
from datetime import datetime

# Import prediction function
from src.models.predict_system import predict_student

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
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# PYDANTIC MODELS (INPUT VALIDATION)
# =========================================

class StudentInput(BaseModel):
    """
    Input schema for student prediction with validation constraints.
    """
    term1_avg: float = Field(
        ...,
        ge=0,
        le=20,
        description="Term 1 average score (0-20)"
    )
    term2_avg: float = Field(
        ...,
        ge=0,
        le=20,
        description="Term 2 average score (0-20)"
    )
    seq5_score: float = Field(
        ...,
        ge=0,
        le=20,
        description="Sequence 5 score (0-20)"
    )
    attendance_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Attendance percentage (0-100)"
    )
    parental_support: int = Field(
        ...,
        ge=0,
        le=1,
        description="Parental support level (0 or 1)"
    )
    study_hours_per_day: float = Field(
        ...,
        ge=0,
        le=24,
        description="Average daily study hours (0-24)"
    )
    sleep_hours: float = Field(
        ...,
        ge=0,
        le=24,
        description="Average sleep hours per day (0-24)"
    )
    class_participation: int = Field(
        ...,
        ge=0,
        le=5,
        description="Class participation level (0-5)"
    )
    homework_completion: float = Field(
        ...,
        ge=0,
        le=100,
        description="Homework completion percentage (0-100)"
    )
    extra_lessons: int = Field(
        ...,
        ge=0,
        description="Number of extra lessons/tutoring sessions (≥0)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "term1_avg": 12.0,
                "term2_avg": 13.0,
                "seq5_score": 14.0,
                "attendance_percentage": 85.0,
                "parental_support": 1,
                "study_hours_per_day": 3.0,
                "sleep_hours": 7.0,
                "class_participation": 3,
                "homework_completion": 80.0,
                "extra_lessons": 1
            }
        }


class PredictionResponse(BaseModel):
    """
    Output schema for prediction results.
    """
    prediction: str = Field(..., description="Pass or Fail")
    probability: float = Field(..., description="Overall pass probability (0-1)")
    academic_prob: float = Field(..., description="Academic model probability (0-1)")
    behavioral_prob: float = Field(..., description="Behavioral model probability (0-1)")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "Pass",
                "probability": 0.867,
                "academic_prob": 0.950,
                "behavioral_prob": 0.742
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response schema.
    """
    error: str = Field(..., description="Error message")
    detail: str = Field(None, description="Detailed error information")


# =========================================
# API ENDPOINTS
# =========================================

@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """
    Root endpoint - API status message.
    
    Returns:
        Basic API information
    """
    return {
        "message": "Student Success API running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for monitoring.
    
    Returns:
        API health status
    """
    logger.info("Health check requested")
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Prediction"]
)
async def predict(student: StudentInput) -> Dict[str, Union[str, float]]:
    """
    Predict student success probability.
    
    Args:
        student: StudentInput object with all required features
        
    Returns:
        PredictionResponse with prediction and probabilities
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Log incoming request
        logger.info("=" * 60)
        logger.info("Prediction request received")
        logger.info(f"Input data: {student.dict()}")
        
        # Convert Pydantic model to dictionary
        student_data = student.dict()
        
        # Call prediction function
        result = predict_student(student_data)
        
        # Log prediction result
        logger.info(f"Prediction result: {result}")
        logger.info("=" * 60)
        
        return result
        
    except ValueError as e:
        # Handle validation errors from predict_student
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Validation error",
                "detail": str(e)
            }
        )
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "detail": "An error occurred during prediction. Please try again."
            }
        )


@app.get("/info", tags=["Info"])
async def api_info() -> Dict[str, Union[str, Dict]]:
    """
    Get API information and feature requirements.
    
    Returns:
        API information including required features
    """
    return {
        "api_name": "Student Success Prediction API",
        "version": "1.0.0",
        "description": "Predicts student pass/fail probability using dual-model approach",
        "models": {
            "academic": {
                "algorithm": "Logistic Regression",
                "weight": 0.6,
                "features": [
                    "term1_avg",
                    "term2_avg",
                    "seq5_score",
                    "attendance_percentage",
                    "parental_support"
                ]
            },
            "behavioral": {
                "algorithm": "XGBoost",
                "weight": 0.4,
                "features": [
                    "study_hours_per_day",
                    "sleep_hours",
                    "class_participation",
                    "homework_completion",
                    "extra_lessons",
                    "attendance_percentage"
                ]
            }
        },
        "endpoints": {
            "GET /": "Root message",
            "GET /health": "Health check",
            "POST /predict": "Make prediction",
            "GET /info": "API information",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation"
        }
    }


# =========================================
# STARTUP/SHUTDOWN EVENTS
# =========================================

@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on API startup.
    """
    logger.info("=" * 60)
    logger.info("Student Success Prediction API starting...")
    logger.info("Models loaded successfully")
    logger.info("API ready to accept requests")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform on API shutdown.
    """
    logger.info("Student Success Prediction API shutting down...")


# =========================================
# MAIN (FOR DEVELOPMENT)
# =========================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
