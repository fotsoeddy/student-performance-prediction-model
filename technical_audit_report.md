# 🔍 Technical Audit Report — Student Performance Prediction Model

> **Auditor**: Automated ML Codebase Audit  
> **Date**: 2026-04-27  
> **Repository**: `fotsoeddy/student-performance-prediction-model`

---

## Table of Contents

- [A. Executive Summary](#a-executive-summary)
- [B. Current Repository Structure](#b-current-repository-structure)
- [C. Dataset and Feature Analysis](#c-dataset-and-feature-analysis)
- [D. Model Architecture](#d-model-architecture)
- [E. Training and Evaluation Summary](#e-training-and-evaluation-summary)
- [F. API and Integration Summary](#f-api-and-integration-summary)
- [G. Available Features (Feature Inventory)](#g-available-features)
- [H. Missing MVP Features](#h-missing-mvp-features)
- [I. Academic Report Notes](#i-academic-report-notes)
- [J. Risks and Recommendations](#j-risks-and-recommendations)
- [K. Final MVP Readiness Score](#k-final-mvp-readiness-score)

---

## A. Executive Summary

The project is an **AI-based Student Performance Prediction System** that predicts whether a secondary school student will **pass or fail** their final sequence (Sequence 6). It uses a **dual-model weighted ensemble**:

| Component | Algorithm | Weight |
|---|---|---|
| Academic Model | Logistic Regression (Platt-calibrated) | 70 % |
| Behavioral Model | XGBoost (Platt-calibrated) | 30 % |

**Current outputs**: binary Pass/Fail prediction, overall pass probability, individual academic and behavioral probabilities.

### Key Findings

| Area | Status |
|---|---|
| Core prediction pipeline | ✅ Functional |
| FastAPI endpoint | ✅ Working (`POST /predict`) |
| Model training scripts | ✅ Present & runnable |
| Calibrated probabilities | ✅ Implemented |
| Risk level classification | ❌ Missing |
| Prediction history / DB | ❌ Missing |
| Batch prediction | ❌ Missing |
| Explainability (SHAP/LIME) | ❌ Missing |
| Early warning alerts | ❌ Missing |
| Real Cameroonian data | ❌ Missing — uses proxy datasets |
| Model evaluation report artifact | ❌ Not persisted |
| Confusion matrix / Recall analysis | ❌ Not saved |
| Deployment config (Docker, CI/CD) | ⚠️ Dockerfile in README only, not in repo |

> **Overall MVP Readiness Score: 38 / 100** — The core ML pipeline works, but significant gaps remain in risk classification, explainability, batch prediction, and production deployment.

---

## B. Current Repository Structure

```
student-performance-prediction-model/
├── data/
│   ├── raw/
│   │   ├── student-mat.csv              ← UCI Student Performance Dataset (395 rows)
│   │   └── StudentPerformanceFactors.csv ← Kaggle Student Performance Factors (6607 rows)
│   └── processed/
│       ├── aligned_student_data.csv      ← Preprocessed UCI → academic model training
│       └── aligned_kaggle_data_full.csv  ← Preprocessed Kaggle → behavioral model training
├── models/
│   ├── academic_model.pkl               ← Calibrated Logistic Regression
│   ├── behavioral_model.pkl             ← Calibrated XGBoost
│   └── engagement_scaler.pkl            ← MinMaxScaler for engagement score
├── src/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── train_academic_model.py      ← Academic model training + calibration
│       ├── train_behavioral_model.py    ← Behavioral model training + calibration
│       └── predict_system.py            ← Core prediction logic (weighted ensemble)
├── app.py                               ← FastAPI application (4 endpoints)
├── test_api.py                          ← Manual API test script (requests-based)
├── train_models.py                      ← Pipeline runner (calls both training scripts)
├── dataset_alignment.py                 ← UCI raw → aligned CSV preprocessing
├── kaggle_dataset_alignment.py          ← Kaggle raw → aligned CSV preprocessing
├── requirements.txt                     ← 8 dependencies
├── README.md                            ← Project documentation
└── venv/                                ← Local virtual environment
```

### File Purpose Summary

| File | Purpose | Category |
|---|---|---|
| [dataset_alignment.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/dataset_alignment.py) | Preprocesses UCI dataset → [aligned_student_data.csv](file:///home/eddy/projects/defense_project/student-performance-prediction-model/data/processed/aligned_student_data.csv) | Preprocessing |
| [kaggle_dataset_alignment.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/kaggle_dataset_alignment.py) | Preprocesses Kaggle dataset → [aligned_kaggle_data_full.csv](file:///home/eddy/projects/defense_project/student-performance-prediction-model/data/processed/aligned_kaggle_data_full.csv) | Preprocessing |
| [train_academic_model.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/train_academic_model.py) | Trains & calibrates academic model | Training |
| [train_behavioral_model.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/train_behavioral_model.py) | Trains & calibrates behavioral model | Training |
| [predict_system.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/predict_system.py) | Core prediction logic: validation → ensemble → result | Prediction |
| [app.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/app.py) | FastAPI application (endpoints, schemas) | API |
| [test_api.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/test_api.py) | Manual HTTP tests using `requests` | Testing |
| [train_models.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/train_models.py) | Pipeline orchestrator (runs both training scripts) | Training |

### Unused / Unclear Files

| Issue | Detail |
|---|---|
| `Late_count` column | Created in [dataset_alignment.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/dataset_alignment.py) (line 38) but never used by any model |
| `Sleep_hours` column | Created in [kaggle_dataset_alignment.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/kaggle_dataset_alignment.py) (line 55) but explicitly removed during behavioral model training |
| Extended features in Kaggle processed CSV | 9 extended features (`Teacher_quality`, `Peer_influence`, etc.) are saved in [aligned_kaggle_data_full.csv](file:///home/eddy/projects/defense_project/student-performance-prediction-model/data/processed/aligned_kaggle_data_full.csv) but **not used** by any model |
| `Physical_activity` column | Mapped raw value directly, not ordinal-encoded like others — treated as numeric |
| No `Dockerfile` | Referenced in README but not present in repository |
| No `.env` / config file | No configuration management for different environments |

---

## C. Dataset and Feature Analysis

### C.1 Raw Datasets

#### UCI Student Performance Dataset ([student-mat.csv](file:///home/eddy/projects/defense_project/student-performance-prediction-model/data/raw/student-mat.csv))

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Rows | 395 |
| Columns | 33 (semicolon-separated) |
| Subject | Mathematics grades of Portuguese secondary students |
| Target source | `G3` (final grade, 0–20 scale) |
| Missing values | 0 |
| Duplicated rows | 0 |

**Original columns**: `school`, `sex`, [age](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/predict_system.py#91-100), `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `reason`, `guardian`, `traveltime`, `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, [health](file:///home/eddy/projects/defense_project/student-performance-prediction-model/test_api.py#26-39), `absences`, `G1`, `G2`, `G3`

#### Kaggle Student Performance Factors Dataset ([StudentPerformanceFactors.csv](file:///home/eddy/projects/defense_project/student-performance-prediction-model/data/raw/StudentPerformanceFactors.csv))

| Property | Value |
|---|---|
| Source | Kaggle |
| Rows | 6,607 |
| Columns | 20 |
| Subject | Synthetic/survey-based student performance factors |
| Target source | `Exam_Score` (0–100 scale) |
| Missing values | Potentially in `Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home` |
| Duplicated rows | 0 |

**Original columns**: `Hours_Studied`, `Attendance`, `Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Sleep_Hours`, `Previous_Scores`, `Motivation_Level`, `Internet_Access`, `Tutoring_Sessions`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Physical_Activity`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`, `Exam_Score`

### C.2 Processed / Aligned Datasets

#### [aligned_student_data.csv](file:///home/eddy/projects/defense_project/student-performance-prediction-model/data/processed/aligned_student_data.csv) (from UCI — used for **Academic Model**)

| Property | Value |
|---|---|
| Rows | 395 |
| Columns | 11 |
| Target | `Pass` (1 if G3 ≥ 10, else 0) |

**Columns**: `Term1_avg`, `Term2_avg`, `Seq5_score`, `Attendance_percentage`, `Late_count`, `Study_hours_per_day`, `Homework_completion`, `Extra_lessons`, `Class_participation`, `Parental_support`, `Pass`

#### [aligned_kaggle_data_full.csv](file:///home/eddy/projects/defense_project/student-performance-prediction-model/data/processed/aligned_kaggle_data_full.csv) (from Kaggle — used for **Behavioral Model**)

| Property | Value |
|---|---|
| Rows | 6,607 |
| Columns | 19 |
| Target | `Pass` (1 if Exam_Score ≥ 60th percentile) |

**Columns**: `Term1_avg`, `Term2_avg`, `Seq5_score`, `Attendance_percentage`, `Study_hours_per_day`, `Homework_completion`, `Extra_lessons`, `Class_participation`, `Sleep_hours`, `Parental_support`, `Teacher_quality`, `Peer_influence`, `Access_to_resources`, `Family_income`, `Internet_access`, `School_type`, `Learning_disabilities`, `Physical_activity`, `Pass`

### C.3 Feature Transformation Summary

| Final Feature | UCI Source → Transformation | Kaggle Source → Transformation |
|---|---|---|
| `Term1_avg` | `G1` (direct) | `Previous_Scores` (direct — **0–100 scale, not 0–20**) |
| `Term2_avg` | `G2` (direct) | `Previous_Scores` (same value as Term1) |
| `Seq5_score` | `G2` (proxy — **same as Term2**) | `Previous_Scores` (same value) |
| `Attendance_percentage` | `100 - (absences/max_absences × 100)` | `Attendance` (direct) |
| `Study_hours_per_day` | `studytime` (1–4 ordinal, direct map) | `Hours_Studied / 7` (weekly→daily) |
| `Homework_completion` | `studytime × 20` (simulated) | `Hours_Studied × 5` (simulated) |
| `Extra_lessons` | `schoolsup` (yes/no → 1/0) | `Tutoring_Sessions` (numeric count) |
| `Class_participation` | `activities` (yes/no → 1/0) | `Motivation_Level` (Low/Med/High → 0/1/2) |
| `Parental_support` | `famsup` (yes/no → 1/0) | `Parental_Involvement` (Low/Med/High → 0/1/2) |
| `Late_count` | `absences × 0.3` (simulated) | *(not created)* |

### C.4 Critical Data Issues

> [!CAUTION]
> **Scale mismatch**: The Kaggle processed data stores `Term1_avg`, `Term2_avg`, `Seq5_score` in a **0–100 scale** (from `Previous_Scores`), while the API validates these fields as **0–20**. The **academic model** is trained on UCI data (0–20 scale) but the **behavioral model** is trained on Kaggle data (0–100 scale). This means the two models operate on fundamentally different feature scales for shared feature names.

> [!WARNING]
> **Proxy features**: `Seq5_score` is a copy of `G2` (UCI) or `Previous_Scores` (Kaggle) — it is NOT a real Sequence 5 exam score. `Homework_completion` is directly derived from study hours, not actual homework data. `Class_participation` is mapped from `activities` (a yes/no extracurricular flag) in UCI — not real class participation.

> [!WARNING]
> **Identical columns**: In the Kaggle alignment, `Term1_avg = Term2_avg = Seq5_score = Previous_Scores`. All three academic features are the **exact same value**, meaning the academic-style features carry zero additional information for Kaggle-trained models.

### C.5 Class Balance

| Dataset | Pass | Fail | Ratio |
|---|---|---|---|
| UCI (aligned) | ~265 (67%) | ~130 (33%) | ~2:1 |
| Kaggle (aligned) | ~2,643 (40%) | ~3,964 (60%) | ~2:3 |

> [!NOTE]
> The UCI dataset is **imbalanced toward Pass**, while the Kaggle dataset uses a percentile threshold that creates a **60/40 Fail/Pass split**. The two datasets have opposite class imbalances. Neither training script applies any resampling (SMOTE, class weights, etc.).

### C.6 Missing Values

- **UCI aligned**: 0 missing values.
- **Kaggle aligned**: All missing values filled with `fillna(0)` — this is a **blunt approach** that could distort Teacher_quality, Family_income, etc.

---

## D. Model Architecture

### D.1 Dual-Model Weighted Ensemble

```
                    ┌─────────────────────┐
                    │  Student Input Data  │
                    └──────────┬──────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
   ┌───────────────────────┐     ┌───────────────────────────┐
   │   Academic Model      │     │   Behavioral Model        │
   │   (Logistic Reg.)     │     │   (XGBoost)               │
   │   Features:           │     │   Features:               │
   │   • term1_avg         │     │   • engagement_score      │
   │   • term2_avg         │     │   • attendance_percentage │
   │   • seq5_score        │     │   • extra_lessons         │
   │   • attendance_%      │     │   • class_participation   │
   │   • parental_support  │     │                           │
   └────────┬──────────────┘     └──────────┬────────────────┘
            │                                │
            │ P(pass) × 0.7                  │ P(pass) × 0.3
            │                                │
            └───────────┬────────────────────┘
                        ▼
              ┌──────────────────┐
              │ Final Probability│
              │ = 0.7×A + 0.3×B │
              │                  │
              │ ≥ 0.5 → Pass    │
              │ < 0.5 → Fail    │
              └──────────────────┘
```

### D.2 Model Details

| Property | Academic Model | Behavioral Model |
|---|---|---|
| Algorithm | Logistic Regression | XGBoost |
| Calibration | Platt scaling (`CalibratedClassifierCV`, sigmoid, 5-fold) | Platt scaling (`CalibratedClassifierCV`, sigmoid, 5-fold) |
| Training data | [aligned_student_data.csv](file:///home/eddy/projects/defense_project/student-performance-prediction-model/data/processed/aligned_student_data.csv) (UCI, 395 rows) | [aligned_kaggle_data_full.csv](file:///home/eddy/projects/defense_project/student-performance-prediction-model/data/processed/aligned_kaggle_data_full.csv) (Kaggle, 6607 rows) |
| Number of features | 5 | 4 (after feature engineering) |
| Feature engineering | None | [engagement_score](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/predict_system.py#91-100) = normalized avg of `study_hours` + `homework` |
| Ensemble weight | 70% | 30% |
| Saved artifact | [models/academic_model.pkl](file:///home/eddy/projects/defense_project/student-performance-prediction-model/models/academic_model.pkl) | [models/behavioral_model.pkl](file:///home/eddy/projects/defense_project/student-performance-prediction-model/models/behavioral_model.pkl) + [models/engagement_scaler.pkl](file:///home/eddy/projects/defense_project/student-performance-prediction-model/models/engagement_scaler.pkl) |

### D.3 Prediction Logic ([predict_system.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/predict_system.py))

1. **Validate** all 9 input fields against range constraints.
2. **Extract** academic features (5 fields, raw values).
3. **Compute** engagement score from `study_hours_per_day` and `homework_completion` using the saved `MinMaxScaler`.
4. **Assemble** behavioral features: `[engagement_score, attendance_percentage, extra_lessons, class_participation]`.
5. **Predict** calibrated probabilities from each model.
6. **Combine**: `final_prob = 0.7 × academic_prob + 0.3 × behavioral_prob`.
7. **Decide**: `Pass` if `final_prob ≥ 0.5`, else `Fail`.

### D.4 Threshold

- Fixed at **0.5**. Not tunable via API or config.
- No risk level classification (Low/Medium/High) is computed.

---

## E. Training and Evaluation Summary

### E.1 Training Pipeline

| Step | Action | Script |
|---|---|---|
| 1 | Load processed CSV data | [train_academic_model.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/train_academic_model.py) L26 / [train_behavioral_model.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/train_behavioral_model.py) L27 |
| 2 | Select features & target | Hardcoded feature lists |
| 3 | Feature engineering (behavioral only) | `MinMaxScaler` → engagement score |
| 4 | 80/20 stratified train/test split | `random_state=42` |
| 5 | Train base model (uncalibrated) | LogReg / XGBoost |
| 6 | Evaluate uncalibrated | Accuracy, AUC, Brier |
| 7 | Wrap with `CalibratedClassifierCV` | Platt scaling, 5-fold CV |
| 8 | Evaluate calibrated | Accuracy, AUC, Brier |
| 9 | Save [.pkl](file:///home/eddy/projects/defense_project/student-performance-prediction-model/models/academic_model.pkl) artifacts | `joblib.dump()` |

**Commands to retrain**:
```bash
python dataset_alignment.py           # Preprocess UCI → aligned CSV
python kaggle_dataset_alignment.py     # Preprocess Kaggle → aligned CSV
python train_models.py                 # Trains both models sequentially
```

### E.2 Reported Metrics (from README)

| Metric | Academic Model | Behavioral Model |
|---|---|---|
| AUC | ~0.97 | ~0.92 |
| Accuracy | *(printed during training, not persisted)* | *(printed during training, not persisted)* |
| Brier Score | *(printed during training, not persisted)* | *(printed during training, not persisted)* |

### E.3 Missing Evaluation Artifacts

> [!IMPORTANT]
> The following metrics are **not persisted** anywhere — they are only printed to stdout during training:

| Missing Item | Status |
|---|---|
| Precision | ❌ Not computed |
| Recall | ❌ Not computed |
| F1-Score | ❌ Not computed |
| Confusion Matrix | ❌ Not computed |
| Classification Report | ❌ Not computed |
| Calibration Curve Plot | ❌ Not saved |
| Feature Importance Plot | ❌ Not saved |
| Evaluation Report (JSON/CSV) | ❌ Does not exist |
| Cross-validation results | ❌ Only used inside calibration, not reported |

> [!CAUTION]
> **Recall is critical** for an early warning system. Missing a failing student (false negative) is far more harmful than incorrectly flagging a passing student (false positive). Without recall metrics, it is impossible to evaluate the system's fitness for its stated purpose.

---

## F. API and Integration Summary

### F.1 Endpoints

| Method | Path | Description | Auth Required |
|---|---|---|---|
| GET | `/` | Root status message | No |
| GET | `/health` | Health check with timestamp | No |
| POST | `/predict` | Single student prediction | No |
| GET | `/info` | API metadata & model info | No |
| GET | `/docs` | Swagger UI (auto-generated) | No |
| GET | `/redoc` | ReDoc documentation | No |

### F.2 `POST /predict` — Request Body

```json
{
  "term1_avg": 10.0,          // float, 0–20, required
  "term2_avg": 10.0,          // float, 0–20, required
  "seq5_score": 10.0,         // float, 0–20, required
  "attendance_percentage": 70.0, // float, 0–100, required
  "parental_support": 0,      // int, 0 or 1, required
  "study_hours_per_day": 2.5,  // float, 0–24, required
  "homework_completion": 65.0, // float, 0–100, required
  "class_participation": 2.5,  // float, 0–5, required
  "extra_lessons": 0           // int, ≥0, required
}
```

### F.3 Response Body

```json
{
  "prediction": "Pass",        // "Pass" or "Fail"
  "probability": 0.612,        // Combined pass probability (0–1)
  "academic_prob": 0.750,      // Academic model output
  "behavioral_prob": 0.320     // Behavioral model output
}
```

### F.4 Validation

- ✅ Pydantic `Field` constraints with [ge](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/predict_system.py#91-100)/`le` enforce ranges.
- ✅ Custom [validate_input()](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/predict_system.py#63-83) in [predict_system.py](file:///home/eddy/projects/defense_project/student-performance-prediction-model/src/models/predict_system.py) adds a second layer.
- ✅ Missing fields return 422 Unprocessable Entity.
- ❌ No authentication or API key required.
- ❌ No rate limiting.

### F.5 CORS Configuration

```python
allow_origins=["*"]  # Wide open — must be restricted before production
```

### F.6 Backend Integration Pattern

The API expects **aggregated** values. A Django/Node backend should:
1. Collect raw daily records (attendance, homework submissions, etc.).
2. Aggregate them into the 9 required features.
3. `POST` to `/predict` with the JSON body.
4. Receive and store the prediction result.

> [!NOTE]
> No student ID, timestamp, or metadata is accepted by the API. The backend must handle prediction history storage itself.

---

## G. Available Features

### Feature Inventory Table

| Feature | Category | Data Type | Expected Range | UCI Source | Kaggle Source | Used by Model? | MVP Required? | Collectible in School? |
|---|---|---|---|---|---|---|---|---|
| `term1_avg` | Academic | float | 0–20 | `G1` | `Previous_Scores` (0–100 ⚠️) | ✅ Academic | ✅ Yes | ✅ Yes |
| `term2_avg` | Academic | float | 0–20 | `G2` | `Previous_Scores` (same) | ✅ Academic | ✅ Yes | ✅ Yes |
| `seq5_score` | Academic | float | 0–20 | `G2` (proxy) | `Previous_Scores` (same) | ✅ Academic | ✅ Yes | ✅ Yes |
| `attendance_percentage` | Attendance | float | 0–100 | `absences` (inverted) | `Attendance` | ✅ Both | ✅ Yes | ✅ Yes |
| `parental_support` | Family/Support | int | 0–1 | `famsup` (binary) | `Parental_Involvement` (ordinal→0–2) | ✅ Academic | ✅ Yes | ⚠️ Partially |
| `study_hours_per_day` | Behavioural | float | 0–24 | `studytime` (ordinal 1–4) | `Hours_Studied / 7` | ✅ Behavioral (via engagement) | ✅ Yes | ⚠️ Self-reported |
| `homework_completion` | Behavioural | float | 0–100 | `studytime × 20` (proxy) | `Hours_Studied × 5` (proxy) | ✅ Behavioral (via engagement) | ✅ Yes | ✅ Yes (if tracked) |
| `class_participation` | Behavioural | float | 0–5 | `activities` (binary) | `Motivation_Level` (ordinal) | ✅ Behavioral | ✅ Yes | ⚠️ Subjective |
| `extra_lessons` | Academic | int | ≥ 0 | `schoolsup` (binary) | `Tutoring_Sessions` (count) | ✅ Behavioral | ✅ Yes | ✅ Yes |

### Extended / Future Features (Kaggle only – saved but NOT used)

| Feature | Category | Data Type | Range | Used? | MVP? | Collectible? |
|---|---|---|---|---|---|---|
| `teacher_quality` | School/Env | int | 0–2 | ❌ | Low | ⚠️ Sensitive |
| `peer_influence` | Behavioural | int | 0–2 | ❌ | Low | ❌ Subjective |
| `access_to_resources` | School/Env | int | 0–2 | ❌ | Low | ⚠️ Survey |
| `family_income` | Family/Support | int | 0–2 | ❌ | Low | ❌ Sensitive/private |
| `internet_access` | School/Env | int | 0–1 | ❌ | Low | ✅ Yes |
| `learning_disabilities` | Behavioural | int | 0–1 | ❌ | Medium | ⚠️ Medical records |
| `physical_activity` | Behavioural | int | 0–6 | ❌ | Low | ⚠️ Self-reported |
| `school_type` | School/Env | int | 0–1 | ❌ | Low | ✅ Yes |
| `sleep_hours` | Behavioural | int | 4–10 | ❌ (removed) | Low | ❌ Self-reported |

---

## H. Missing MVP Features

| # | MVP Feature | Available? | Current Implementation | What Is Missing | Priority |
|---|---|---|---|---|---|
| 1 | Predict student pass/fail | ✅ Yes | `POST /predict` returns `"Pass"` or `"Fail"` | — | — |
| 2 | Return probability of passing | ✅ Yes | Returns `probability` (0–1) | — | — |
| 3 | Return risk level (Low/Medium/High) | ❌ No | Not implemented | Add risk classification logic based on probability thresholds | **High** |
| 4 | Accept student academic records from backend | ✅ Yes | Accepts term scores via JSON | — | — |
| 5 | Accept attendance percentage | ✅ Yes | `attendance_percentage` field | — | — |
| 6 | Accept behavioural indicators | ✅ Yes | `study_hours`, `homework`, `participation`, `extra_lessons` | — | — |
| 7 | Store prediction history | ❌ No | No database, no storage | Add database (SQLite/PostgreSQL) + history endpoint | **High** |
| 8 | Show teacher dashboard data | ❌ No | No dashboard endpoint | Create `GET /students/{id}/predictions` or similar | **High** |
| 9 | Generate early warning alerts | ❌ No | No alert logic | Add threshold-based alert system | **High** |
| 10 | Generate parent notifications | ❌ No | No notification system | Integrate with notification service | **Medium** |
| 11 | Recommend revision/intervention plan | ❌ No | No recommendation logic | Add rule-based or ML-based recommendations | **Medium** |
| 12 | Explain key prediction reasons | ❌ No | No explainability | Add SHAP or LIME feature importance per prediction | **High** |
| 13 | Support batch prediction | ❌ No | Single student only | Add `POST /predict/batch` accepting array | **Medium** |
| 14 | Provide API documentation | ✅ Partial | Swagger UI auto-generated at `/docs` | Add versioning, detailed field descriptions | **Low** |
| 15 | Provide model evaluation report | ❌ No | Metrics only printed to stdout | Persist evaluation results as JSON + generate report | **High** |
| 16 | Provide deployment instructions | ⚠️ Partial | README has basic instructions + Dockerfile snippet | Add actual Dockerfile, docker-compose, env config | **Medium** |

---

## I. Academic Report Notes

*The following is written in academic style for adaptation into Chapters 3–5 of a project report.*

### I.1 Data Sources (Chapter 3)

Two publicly available datasets were employed in the development of this system:

1. **UCI Student Performance Dataset** (Cortez & Silva, 2008): Contains academic records of 395 Portuguese students enrolled in Mathematics courses at two secondary schools. The dataset includes 33 attributes covering student demographics, family background, study habits, and three grading periods (G1, G2, G3) on a 0–20 scale. The final grade G3 was binarised using a threshold of 10 (the minimum passing grade in the Portuguese grading system) to create the target variable.

2. **Kaggle Student Performance Factors Dataset**: A larger dataset of 6,607 student records containing 20 attributes including study hours, attendance, parental involvement, motivation level, and exam scores on a 0–100 scale. The target variable was created using a percentile-based threshold at the 60th percentile of exam scores.

### I.2 Data Preprocessing (Chapter 3)

Feature alignment was performed to standardise both datasets into a common schema. The following transformations were applied:

- **Academic features**: First and second term averages were mapped from `G1` and `G2` (UCI) and `Previous_Scores` (Kaggle). A proxy Sequence 5 score was derived from the second-term grade.
- **Attendance**: For the UCI dataset, raw absence counts were converted to percentage using the formula: `Attendance% = 100 − (absences / max_absences × 100)`.
- **Engagement indicators**: Study time and homework completion were derived from the `studytime` variable (UCI) and `Hours_Studied` (Kaggle) using linear scaling transformations.
- **Categorical encoding**: Binary variables (parental support, extra lessons, class participation) were encoded as 0/1 integers. Ordinal variables (motivation level, parental involvement) were mapped to numeric scales.

### I.3 Feature Engineering (Chapter 3)

An **Engagement Score** was engineered for the behavioural model by normalising `study_hours_per_day` and `homework_completion` using Min-Max scaling, then computing their mean. This reduced feature redundancy and provided a single composite indicator of student engagement.

### I.4 Model Algorithms (Chapter 4)

A dual-model ensemble architecture was adopted:

- **Academic Model**: Logistic Regression was selected for its interpretability and suitability for well-structured academic data. The model's inherent probability calibration makes it appropriate for producing meaningful confidence scores.
- **Behavioural Model**: Gradient Boosted Trees (XGBoost) were employed to capture non-linear relationships among engagement, attendance, and participation variables. XGBoost's regularisation parameters (`max_depth=4`, `subsample=0.8`) were configured to mitigate overfitting.

### I.5 Probability Calibration (Chapter 4)

Both models underwent Platt scaling (sigmoid calibration) via `CalibratedClassifierCV` with 5-fold cross-validation. This post-hoc calibration ensures that predicted probabilities closely approximate true class frequencies, which is essential for an early warning system where probability thresholds drive intervention decisions.

### I.6 Model Justification (Chapter 4)

The weighted ensemble approach (`P_final = 0.7 × P_academic + 0.3 × P_behavioral`) was designed to reflect the primacy of academic performance in predicting outcomes while augmenting predictions with behavioural signals. The 70/30 weighting acknowledges that past grades are the strongest predictor of future performance, while behavioural indicators provide early signals that may precede grade deterioration.

### I.7 Evaluation Metrics (Chapter 5)

The models were evaluated using Accuracy, ROC AUC, and Brier Score Loss. The academic model achieved an AUC of approximately 0.97, and the behavioural model achieved an AUC of approximately 0.92. Additional metrics including Precision, Recall, F1-Score, and confusion matrices are recommended but were not computed in the current implementation.

### I.8 System Architecture (Chapter 4)

The system follows a microservice architecture where the ML prediction engine is deployed as a RESTful API using FastAPI. The API receives pre-aggregated student features from a school management backend, processes them through the dual-model ensemble, and returns prediction results including pass/fail classification and calibrated probabilities.

### I.9 Limitations (Chapter 5)

1. The system was trained exclusively on publicly available datasets that may not represent the demographic, curricular, and grading characteristics of Cameroonian secondary schools.
2. Several features are proxies (e.g., `Seq5_score` duplicates `Term2_avg`; `homework_completion` is derived from study hours rather than actual submissions).
3. The two models are trained on datasets with different scales (0–20 vs 0–100), which may introduce inconsistencies in the ensemble.
4. No real student data from the target population was used for training or validation.
5. The fixed 0.5 threshold may not be optimal for minimising false negatives in an early warning context.

### I.10 Ethical Considerations (Chapter 5)

- **Bias risk**: Training on European/synthetic data and deploying for Cameroonian students may introduce cultural and systemic biases.
- **Privacy**: The system processes academic and behavioural data that constitutes personally identifiable information (PII). Data protection measures must be implemented.
- **Labelling risk**: Incorrectly labelling students as "at-risk" may create stigmatisation; incorrectly labelling them as "safe" may prevent timely intervention.
- **Consent**: Students and parents should be informed about the use of AI in academic monitoring.

### I.11 Future Improvements (Chapter 6)

- Collect and train on anonymised data from Cameroonian secondary schools.
- Add model explainability (SHAP values) for transparent decision-making.
- Implement risk level classification (Low/Medium/High) with configurable thresholds.
- Add intervention recommendation engine based on identified weak features.
- Deploy with proper authentication, rate limiting, and monitoring.
- Implement continuous model retraining as new cohort data becomes available.

---

## J. Risks and Recommendations

### J.1 Technical Risks

| # | Risk | Severity | Detail |
|---|---|---|---|
| 1 | **Data leakage** | 🔴 High | `Seq5_score` is a copy of `G2`/`Term2_avg`. Using both in the academic model gives the model the same information twice, inflating apparent accuracy. In a real scenario, `Seq5_score` would not be available before prediction time. |
| 2 | **Scale mismatch** | 🔴 High | Academic model trained on 0–20 scale (UCI), behavioral model trained on 0–100 scale (Kaggle). The API validates all academic scores as 0–20, but the behavioral model's scaler was fit on 0–100 range data. At inference, `attendance_percentage`, `extra_lessons`, and `class_participation` pass directly without scaling, so the behavioral model receives values within training range — but the engagement scaler was fit on Kaggle-scale data that may differ from real inputs. |
| 3 | **Proxy features** | 🔴 High | `homework_completion = study_hours × factor` is not real homework data. The model cannot distinguish homework behavior from study behavior. |
| 4 | **Target variable inconsistency** | 🟡 Medium | UCI uses G3 ≥ 10 (absolute threshold). Kaggle uses 60th percentile (relative threshold). The two models define "Pass" differently. |
| 5 | **Overfitting risk (academic model)** | 🟡 Medium | Only 395 training samples with AUC ~0.97. Suspiciously high — likely because G1, G2 strongly correlate with G3 (grades from the same student in the same year). |
| 6 | **No real data** | 🔴 High | No Cameroonian secondary school data. Model behavior on real target population is unknown. |
| 7 | **No confusion matrix** | 🟡 Medium | Cannot assess false negative rate. Critical for an early warning system. |
| 8 | **API security** | 🟡 Medium | No authentication, no rate limiting, CORS wide open. |
| 9 | **No model versioning** | 🟡 Medium | [.pkl](file:///home/eddy/projects/defense_project/student-performance-prediction-model/models/academic_model.pkl) files are overwritten on retrain with no version history. |
| 10 | **No monitoring** | 🟡 Medium | No prediction logging, drift detection, or performance monitoring in production. |

### J.2 Recommendations (Pre-MVP)

| Priority | Recommendation |
|---|---|
| 🔴 **P0** | **Add risk level classification**: Map probability to Low (≥ 0.7), Medium (0.5–0.7), High (< 0.5) risk levels. |
| 🔴 **P0** | **Compute and persist full evaluation metrics**: Precision, Recall, F1, Confusion Matrix, Classification Report → save as `evaluation_results.json`. |
| 🔴 **P0** | **Add recall optimisation**: Threshold should favour minimising False Negatives. Consider 0.4 threshold instead of 0.5. |
| 🔴 **P0** | **Fix or document scale mismatch**: Either rescale Kaggle features to 0–20 or document the discrepancy. |
| 🟡 **P1** | **Add SHAP explainability**: Return top 3 contributing features per prediction. |
| 🟡 **P1** | **Add batch prediction endpoint**: `POST /predict/batch` accepting an array of students. |
| 🟡 **P1** | **Create Dockerfile**: Actual file in repository, not just README snippet. |
| 🟡 **P1** | **Add prediction history database**: SQLite for MVP, PostgreSQL for production. |
| 🟡 **P1** | **Add student ID and timestamp**: Accept `student_id` in request, return timestamp in response. |
| 🟢 **P2** | **Add authentication**: API key or JWT token for production. |
| 🟢 **P2** | **Add model versioning**: Include model version in response, use timestamped [.pkl](file:///home/eddy/projects/defense_project/student-performance-prediction-model/models/academic_model.pkl) files. |
| 🟢 **P2** | **Collect real data**: Pilot with one Cameroonian secondary school to validate and retrain. |
| 🟢 **P2** | **Add intervention recommendations**: Rule-based suggestions based on weakest features. |

---

## K. Final MVP Readiness Score

| Category | Weight | Score (0–10) | Weighted |
|---|---|---|---|
| Core prediction pipeline | 20% | 8 | 1.6 |
| API functionality | 15% | 7 | 1.05 |
| Model evaluation & metrics | 15% | 2 | 0.30 |
| Risk level classification | 10% | 0 | 0.00 |
| Batch prediction | 5% | 0 | 0.00 |
| Explainability | 10% | 0 | 0.00 |
| Data quality & realism | 10% | 3 | 0.30 |
| Deployment readiness | 10% | 3 | 0.30 |
| Documentation | 5% | 7 | 0.35 |
| **Total** | **100%** | | **3.90 / 10** |

### **Final MVP Readiness Score: 39 / 100**

---

## 📋 Pre-MVP Deployment Checklist

- [ ] **Add risk level** (Low/Medium/High) to prediction response
- [ ] **Compute & save** Precision, Recall, F1, Confusion Matrix for both models
- [ ] **Optimise recall** — lower threshold or add class weights to reduce false negatives
- [ ] **Fix `Seq5_score` proxy** — document or replace with distinct feature
- [ ] **Verify scale consistency** between UCI (0–20) and Kaggle (0–100) features
- [ ] **Add SHAP explanation** — return top contributing factors per prediction
- [ ] **Add `POST /predict/batch`** endpoint for class-level predictions
- [ ] **Add `student_id` field** to request schema for tracking
- [ ] **Add prediction history** — database + `GET /predictions/{student_id}` endpoint
- [ ] **Create actual `Dockerfile`** and `docker-compose.yml`
- [ ] **Restrict CORS** to production origins
- [ ] **Add API authentication** (API key or JWT)
- [ ] **Generate evaluation report** artifact (JSON + markdown)
- [ ] **Add early warning alert logic** with configurable thresholds
- [ ] **Add intervention suggestions** based on weak features
- [ ] **Pilot-test with real Cameroonian school data**
