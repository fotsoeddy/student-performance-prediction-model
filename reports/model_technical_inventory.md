# Model Technical Inventory

> **Repository:** `fotsoeddy/student-performance-prediction-model`
> **Generated:** 28 April 2026

---

## 1. Complete File Listing

### Root Directory

| File | Purpose |
|---|---|
| `app.py` | FastAPI application — defines API endpoints, Pydantic schemas, CORS configuration |
| `train_models.py` | Training pipeline orchestrator — runs academic and behavioural model training scripts sequentially |
| `evaluate_model.py` | Evaluation script — computes ensemble metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix) and saves reports |
| `dataset_alignment.py` | UCI dataset preprocessor — transforms `student-mat.csv` into `aligned_student_data.csv` |
| `kaggle_dataset_alignment.py` | Kaggle dataset preprocessor — transforms `StudentPerformanceFactors.csv` into `aligned_kaggle_data_full.csv` |
| `requirements.txt` | Python dependencies (12 packages) |
| `Dockerfile` | Docker containerisation — Python 3.12-slim base, installs deps, runs Uvicorn on port 8000 |
| `.env.example` | Example environment variables (PORT, LOG_LEVEL, ENV) |
| `.gitignore` | Git ignore rules (venv, __pycache__, .env, IDE files) |
| `README.md` | Project documentation (setup, usage, model evaluation summary) |
| `technical_audit_report.md` | Comprehensive technical audit (600 lines — datasets, architecture, risks, MVP score) |

### Source Code (`src/`)

| File | Purpose |
|---|---|
| `src/__init__.py` | Package initialiser |
| `src/models/__init__.py` | Models sub-package initialiser |
| `src/models/predict_system.py` | Core prediction logic — validation, ensemble prediction, risk classification, explanations, batch support |
| `src/models/train_academic_model.py` | Academic model training — calibrated Logistic Regression on UCI data |
| `src/models/train_behavioral_model.py` | Behavioural model training — calibrated XGBoost on Kaggle data with engagement score engineering |

### Test Files (`tests/`)

| File | Purpose |
|---|---|
| `tests/conftest.py` | Pytest fixtures — `sample_student` (passing) and `failing_student` (at-risk) test data |
| `tests/test_prediction.py` | 6 test cases — single pass/fail prediction, batch prediction, input validation, risk level logic |

### Model Artifacts (`models/`)

| File | Size | Description |
|---|---|---|
| `models/academic_model.pkl` | 6.3 KB | Serialised calibrated Logistic Regression model |
| `models/behavioral_model.pkl` | 1.6 MB | Serialised calibrated XGBoost model |
| `models/engagement_scaler.pkl` | 1.0 KB | Serialised MinMaxScaler for engagement score computation |

### Datasets (`data/`)

| File | Rows | Columns | Description |
|---|---|---|---|
| `data/raw/student-mat.csv` | 395 | 33 | UCI Student Performance Dataset (semicolon-separated) |
| `data/raw/StudentPerformanceFactors.csv` | 6,607 | 20 | Kaggle Student Performance Factors Dataset |
| `data/processed/aligned_student_data.csv` | 395 | 10 | Preprocessed UCI data for academic model |
| `data/processed/aligned_kaggle_data_full.csv` | 6,607 | 19 | Preprocessed Kaggle data for behavioural model |

### Reports (`reports/`)

| File | Description |
|---|---|
| `reports/model_metrics.json` | Ensemble evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, fail_recall) |
| `reports/academic_model_metrics.json` | Academic model metrics (accuracy, ROC-AUC, Brier score) |
| `reports/behavioral_model_metrics.json` | Behavioural model metrics (accuracy, ROC-AUC, Brier score) |
| `reports/model_evaluation.md` | Generated evaluation report with metrics table and confusion matrix |
| `reports/model_improvement_report.md` | Documents improvements made to reach 85/100 MVP readiness |

---

## 2. Main Functions and Classes

### `app.py`

| Name | Type | Description |
|---|---|---|
| `StudentInput` | Pydantic Model | Input validation schema for single student (9 fields with range constraints) |
| `BatchInput` | Pydantic Model | Input schema wrapping a list of `StudentInput` |
| `PredictionResponse` | Pydantic Model | Output schema (prediction, probability, risk_level, academic_prob, behavioral_prob, confidence, explanations) |
| `BatchPredictionResponse` | Pydantic Model | Output schema for batch predictions (predictions list, count, timestamp) |
| `root()` | Endpoint (GET /) | Returns API name, version, docs link |
| `health_check()` | Endpoint (GET /health) | Returns status and timestamp |
| `predict()` | Endpoint (POST /predict) | Single student prediction |
| `predict_batch_endpoint()` | Endpoint (POST /predict/batch) | Batch prediction for multiple students |
| `api_info()` | Endpoint (GET /info) | Returns API metadata, features, and thresholds |

### `src/models/predict_system.py`

| Name | Type | Description |
|---|---|---|
| `validate_input(data)` | Function | Validates all required fields exist and are within allowed ranges |
| `classify_risk_level(prob)` | Function | Maps probability to Low/Medium/High Risk |
| `generate_explanations(data, prob)` | Function | Produces rule-based explanation list based on input thresholds |
| `compute_engagement_score(data)` | Function | Normalises study hours + homework completion into one score using MinMaxScaler |
| `predict_student(data)` | Function | Core prediction — validates, extracts features, runs both models, ensembles, classifies risk, generates explanations |
| `predict_batch(students)` | Function | Iterates over student list, calls `predict_student` for each, handles errors per-student |

### `src/models/train_academic_model.py`

| Name | Type | Description |
|---|---|---|
| Main script | Script | Loads UCI data, selects 5 features, trains calibrated LogisticRegression, saves model + metrics |

### `src/models/train_behavioral_model.py`

| Name | Type | Description |
|---|---|---|
| Main script | Script | Loads Kaggle data, engineers engagement score, trains calibrated XGBoost, saves model + scaler + metrics |

### `train_models.py`

| Name | Type | Description |
|---|---|---|
| `run_script(script_path, description)` | Function | Executes a Python script via subprocess, handles errors |
| `main()` | Function | Runs academic then behavioural training, prints summary |

### `evaluate_model.py`

| Name | Type | Description |
|---|---|---|
| `load_data()` | Function | Loads both processed CSV datasets |
| `evaluate_ensemble()` | Function | Loads models, computes ensemble predictions, calculates metrics, saves JSON + MD reports |

### `dataset_alignment.py`

| Name | Type | Description |
|---|---|---|
| `pass_label(score)` | Function | Returns 1 if score ≥ 10, else 0 |
| Main script | Script | Transforms UCI CSV → aligned CSV with standardised features |

### `kaggle_dataset_alignment.py`

| Name | Type | Description |
|---|---|---|
| Main script | Script | Transforms Kaggle CSV → aligned CSV with standardised features + extended features |

---

## 3. Input/Output of Important Scripts

| Script | Input | Output |
|---|---|---|
| `dataset_alignment.py` | `data/raw/student-mat.csv` | `data/processed/aligned_student_data.csv` |
| `kaggle_dataset_alignment.py` | `data/raw/StudentPerformanceFactors.csv` | `data/processed/aligned_kaggle_data_full.csv` |
| `train_academic_model.py` | `data/processed/aligned_student_data.csv` | `models/academic_model.pkl`, `reports/academic_model_metrics.json` |
| `train_behavioral_model.py` | `data/processed/aligned_kaggle_data_full.csv` | `models/behavioral_model.pkl`, `models/engagement_scaler.pkl`, `reports/behavioral_model_metrics.json` |
| `train_models.py` | Calls training scripts | All model artifacts |
| `evaluate_model.py` | Trained models + processed data | `reports/model_metrics.json`, `reports/model_evaluation.md` |
| `app.py` (API) | JSON student data via HTTP | JSON prediction response |

---

## 4. API Endpoints

| Method | Endpoint | Request Body | Response |
|---|---|---|---|
| GET | `/` | — | `{ message, version, docs }` |
| GET | `/health` | — | `{ status, timestamp }` |
| POST | `/predict` | `StudentInput` (9 fields) | `PredictionResponse` (prediction, probability, risk_level, academic_prob, behavioral_prob, confidence, explanations) |
| POST | `/predict/batch` | `BatchInput` (list of students) | `BatchPredictionResponse` (predictions list, count, timestamp) |
| GET | `/info` | — | API metadata, features, thresholds |
| GET | `/docs` | — | Swagger UI (auto-generated) |
| GET | `/redoc` | — | ReDoc (auto-generated) |

---

## 5. Tests Available

| Test | File | What It Tests |
|---|---|---|
| `test_single_prediction_pass` | `tests/test_prediction.py` | High-performing student returns "Pass" with probability ≥ 0.5 |
| `test_single_prediction_fail` | `tests/test_prediction.py` | At-risk student returns "Fail" with "High Risk" and explanations |
| `test_batch_prediction` | `tests/test_prediction.py` | Batch of 2 students returns 2 results with "success" status |
| `test_invalid_input_range` | `tests/test_prediction.py` | Out-of-range value (term1_avg=25) raises ValueError |
| `test_missing_input` | `tests/test_prediction.py` | Incomplete data raises "Missing required fields" error |
| `test_risk_level_logic` | `tests/test_prediction.py` | Verifies 0.85→Low Risk, 0.55→Medium Risk, 0.30→High Risk |

**Run tests:** `pytest tests/ -v`

---

## 6. Evaluation Metrics Available

### Ensemble (`reports/model_metrics.json`)

```json
{
    "accuracy": 0.8835,
    "precision": 0.9506,
    "recall": 0.8717,
    "f1_score": 0.9094,
    "roc_auc": 0.9554,
    "fail_recall": 0.9077
}
```

### Academic Model (`reports/academic_model_metrics.json`)

```json
{
    "accuracy": 0.8354,
    "roc_auc": 0.9572,
    "brier_score": 0.0806
}
```

### Behavioural Model (`reports/behavioral_model_metrics.json`)

```json
{
    "accuracy": 0.8329,
    "roc_auc": 0.9221,
    "brier_score": 0.1164
}
```
