# Student Performance Prediction & Early Warning API

## Overview
This repository hosts the ML model and FastAPI service for predicting student performance. It is designed to identify students at risk of failure early in the academic year to allow for timely interventions.

**MVP Readiness Score: 85/100**

## Features
- **Pass/Fail Prediction**: Calibrated probability calculation.
- **Risk Level Classification**: Low, Medium, and High Risk status.
- **Rule-based Explanations**: Specific reasons for each prediction (e.g., low attendance, weak scores).
- **Batch Prediction**: Single request for an entire class or list of students.
- **Dual-Model Ensemble**: Combines Academic (Logistic Regression) and Behavioral (XGBoost) insights.
- **0-20 Grading Scale**: Standardized for primary school contexts (Cameroon).

## Project Structure
```bash
.
├── app.py                # FastAPI Service
├── Dockerfile            # Production deployment
├── requirements.txt      # Dependencies
├── evaluate_model.py     # Evaluation script
├── train_models.py       # Training pipeline runner
├── src/
│   └── models/           # Training scripts and prediction logic
├── models/               # Saved PKL models and scalers
└── reports/              # Model evaluation and improvement reports
```

## Setup & Usage

### Local Setup
1. Create venv: `python -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install: `pip install -r requirements.txt`

### Run API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Batch Prediction Example
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "students": [
         {"term1_avg": 15, "term2_avg": 14, "seq5_score": 14, "attendance_percentage": 95, ...},
         {"term1_avg": 6, "term2_avg": 7, "seq5_score": 5, "attendance_percentage": 40, ...}
       ]
     }'
```

## Model Evaluation
| Metric | Value |
| --- | --- |
| Accuracy | 88.35% |
| Recall (Fail class) | 90.77% |
| ROC-AUC | 0.9554 |

Detailed reports are available in `reports/model_evaluation.md`.

## Retraining the Model
To retrain with fresh data:
1. Update `data/raw/` files.
2. Run alignment: `python dataset_alignment.py` and `python kaggle_dataset_alignment.py`.
3. Run pipeline: `python train_models.py --non-interactive`.

## Future Improvements
- Real-time performance tracking (Sequence by Sequence).
- Advanced SHAP explainability.
- Feature expansion (Family income, Teacher quality).
