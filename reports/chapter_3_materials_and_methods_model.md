# CHAPTER THREE: MATERIALS AND METHODS FOR THE MODEL COMPONENT

---

## 3.1 Introduction

This chapter presents a comprehensive description of the materials, datasets, tools, preprocessing techniques, model development process, evaluation procedures, and API implementation used in the development of the AI-based Student Performance Prediction Model. The model constitutes the core machine learning component of the larger **AI-based Student Performance Prediction and Early Warning System**, which is designed to assist secondary school educators in identifying students at risk of academic failure before the end of the academic year.

The prediction model is deployed as a standalone RESTful API service, built to be consumed by external systems such as a teacher dashboard, parent notification interface, and school administrator panel. These consumer applications are planned for future development in separate repositories. This chapter focuses exclusively on the model component: its design, data pipeline, training methodology, evaluation, and the API through which predictions are served.

The system is contextualised for the **Cameroonian secondary school environment**, which uses a 0–20 grading scale and an academic structure comprising six examination sequences per year.

---

## 3.2 Description of the Model Component

The Student Performance Prediction Model is a supervised binary classification system that predicts whether a given student is likely to **pass** or **fail** their end-of-year examination, based on a combination of academic records, attendance data, and behavioural indicators.

### 3.2.1 The Prediction Task

The model performs the following tasks for each student:

1. **Binary Classification**: Predicts whether the student will pass or fail (threshold: probability ≥ 0.5).
2. **Probability Estimation**: Returns a calibrated probability of passing, ranging from 0.0 to 1.0.
3. **Risk Level Classification**: Assigns a risk category — **Low Risk** (probability ≥ 0.70), **Medium Risk** (0.45–0.69), or **High Risk** (below 0.45).
4. **Rule-based Explanation**: Generates a list of human-readable reasons for the prediction (e.g., "Critical: Very low attendance (below 60%)", "Weak academic foundation: Average scores below 10/20").

### 3.2.2 Supporting Early Academic Intervention

The model is designed as an **early warning mechanism**. By analysing a student's academic performance across the first two terms (Term 1 and Term 2), the fifth sequence score (Seq5), attendance records, and behavioural indicators such as study habits, homework completion, and parental support, the system can flag students who are at risk of failure **before the final examination (Sequence 6)**. This enables teachers, parents, and school administrators to intervene early with targeted academic support, remedial sessions, or parental engagement.

### 3.2.3 Role Within the Larger System

The prediction model is one component of a broader system architecture:

| Component | Status | Repository |
|---|---|---|
| ML Prediction API | ✅ Implemented | Current repository |
| Teacher Dashboard | 🔲 Planned | Separate repository |
| Parent Dashboard | 🔲 Planned | Separate repository |
| School Admin Panel | 🔲 Planned | Separate repository |
| Django/Node Backend | 🔲 Planned | Separate repository |

The model API is designed to be consumed by these future components via HTTP POST requests. It accepts student data in JSON format and returns structured prediction results.

---

## 3.3 Materials Used

### 3.3.1 Hardware Requirements

The model was developed and tested on a standard development machine. The following are the minimum recommended hardware requirements for running the training pipeline and the prediction API:

| Component | Minimum Requirement |
|---|---|
| Processor | Dual-core CPU (Intel i5 or equivalent) |
| RAM | 4 GB (8 GB recommended) |
| Storage | 500 MB free disk space |
| Operating System | Linux (Ubuntu 20.04+), macOS, or Windows 10+ |
| GPU | Not required (CPU-based training) |
| Network | Internet access for dependency installation |

> **Note:** The model training pipeline completes in approximately 2–3 minutes on a standard laptop. No GPU acceleration is required, as the dataset sizes are modest (395 and 6,607 records respectively).

### 3.3.2 Software Requirements

The following software libraries and frameworks were identified from the repository's `requirements.txt` and source code:

| Software | Version | Purpose in Project |
|---|---|---|
| **Python** | 3.12 | Primary programming language (specified in Dockerfile) |
| **FastAPI** | ≥ 0.104.0 | Web framework for building the RESTful prediction API |
| **Uvicorn** | ≥ 0.24.0 (with standard extras) | ASGI server for running the FastAPI application |
| **Pandas** | ≥ 1.5.0 | Data loading, manipulation, and preprocessing |
| **NumPy** | ≥ 1.26.0 | Numerical computations and array operations |
| **Scikit-learn** | ≥ 1.2.0 | Machine learning algorithms (Logistic Regression, calibration, evaluation metrics, train/test splitting, MinMaxScaler) |
| **XGBoost** | ≥ 1.7.0 | Gradient boosted tree algorithm for the behavioural model |
| **Joblib** | ≥ 1.2.0 | Serialisation and deserialisation of trained model artifacts (.pkl files) |
| **Matplotlib** | ≥ 3.6.0 | Available for plotting (evaluation visualisations) |
| **Pydantic** | ≥ 2.0.0 | Input validation and schema definition for API request/response models |
| **Pytest** | ≥ 7.0.0 | Automated testing framework for unit and integration tests |
| **HTTPX** | ≥ 0.23.0 | HTTP client for API testing |
| **python-dotenv** | ≥ 1.0.0 | Environment variable management from `.env` files |
| **Docker** | — | Containerisation for deployment (Dockerfile present in repository) |

### 3.3.3 Development Tools

The following development tools were employed during the project:

| Tool | Purpose |
|---|---|
| Code Editor / IDE | Development of Python source code (e.g., VS Code, PyCharm) |
| Git / GitHub | Version control and source code hosting |
| Terminal / Command Line | Script execution, dependency installation, and API testing |
| Swagger UI (`/docs`) | Interactive API documentation and testing (auto-generated by FastAPI) |
| ReDoc (`/redoc`) | Alternative API documentation viewer (auto-generated by FastAPI) |
| cURL / Postman | Manual API endpoint testing |
| Virtual Environment (`venv`) | Python dependency isolation |

---

## 3.4 Data Sources

Two publicly available datasets were employed in the development of the prediction model. Neither dataset originates from Cameroonian schools; they serve as benchmark training data for the current minimum viable product (MVP).

### 3.4.1 UCI Student Performance Dataset

| Property | Detail |
|---|---|
| **Dataset Name** | UCI Student Performance Dataset |
| **Source** | UCI Machine Learning Repository (Cortez & Silva, 2008) |
| **Original File Name** | `student-mat.csv` |
| **Type of Data** | Real academic records from Portuguese secondary school students |
| **Subject** | Mathematics |
| **Number of Records** | 395 |
| **Number of Columns** | 33 |
| **File Format** | CSV (semicolon-separated) |
| **Target Variable** | `G3` — final grade on a 0–20 scale, binarised as Pass (G3 ≥ 10) or Fail (G3 < 10) |
| **Missing Values** | 0 |
| **Relevance** | Provides real academic records with three grading periods (G1, G2, G3) on the same 0–20 scale used in Cameroonian schools. The multi-period grading structure aligns well with the term-based prediction approach. |

**Key columns used:** `G1` (first period grade), `G2` (second period grade), `G3` (final grade), `absences`, `studytime`, `schoolsup` (extra educational support), `famsup` (family educational support), `activities` (extracurricular activities).

### 3.4.2 Kaggle Student Performance Factors Dataset

| Property | Detail |
|---|---|
| **Dataset Name** | Kaggle Student Performance Factors Dataset |
| **Source** | Kaggle |
| **Original File Name** | `StudentPerformanceFactors.csv` |
| **Type of Data** | Synthetic/survey-based student performance factors |
| **Number of Records** | 6,607 |
| **Number of Columns** | 20 |
| **File Format** | CSV (comma-separated) |
| **Target Variable** | `Exam_Score` — on a 0–100 scale, binarised using the 60th percentile threshold |
| **Missing Values** | Potentially present in `Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home` |
| **Relevance** | Provides a large dataset with diverse behavioural and environmental features (study hours, attendance, motivation, parental involvement, tutoring sessions) that complement the academic focus of the UCI dataset. |

**Key columns used:** `Hours_Studied`, `Attendance`, `Parental_Involvement`, `Previous_Scores`, `Motivation_Level`, `Tutoring_Sessions`, `Sleep_Hours`, `Access_to_Resources`, `Family_Income`, `Internet_Access`, `Teacher_Quality`, `School_Type`, `Learning_Disabilities`, `Peer_Influence`, `Physical_Activity`.

### 3.4.3 Data Nature Summary

| Aspect | Detail |
|---|---|
| Real school data | Not used. No Cameroonian school data was available for this MVP. |
| Public benchmark data | ✅ Both datasets are publicly available benchmark datasets. |
| Simulated/synthetic data | The Kaggle dataset is partially synthetic/survey-based. |
| Transformed/aligned data | Both raw datasets undergo extensive transformation to align with the project's standardised feature schema before training. |

---

## 3.5 Data Description

### 3.5.1 Core Features Used by the Model

The following table describes all features accepted by the prediction API and used by the trained models:

| Feature Name | Description | Data Type | Expected Range | Category | UCI Source Column | Kaggle Source Column | Transformation | Used in MVP |
|---|---|---|---|---|---|---|---|---|
| `term1_avg` | First term average score | float | 0–20 | Academic | `G1` (direct, already 0–20) | `Previous_Scores / 5` (normalised from 0–100) | Direct (UCI) / Scaled (Kaggle) | ✅ Yes (Academic model) |
| `term2_avg` | Second term average score | float | 0–20 | Academic | `G2` (direct, already 0–20) | `Previous_Scores / 5` (same value as term1) | Direct (UCI) / Proxy (Kaggle) | ✅ Yes (Academic model) |
| `seq5_score` | Sequence 5 examination score | float | 0–20 | Academic | `G2` (proxy — same as term2) | `Previous_Scores / 5` (same value) | Proxy-based | ✅ Yes (Academic model) |
| `attendance_percentage` | Percentage of classes attended | float | 0–100 | Attendance | `100 - (absences / max_absences × 100)` | `Attendance` (direct) | Transformed (UCI) / Direct (Kaggle) | ✅ Yes (Both models) |
| `parental_support` | Whether the student has parental educational support | int | 0–1 | Support | `famsup` (yes/no → 1/0) | `Parental_Involvement` (Low/Med/High → 0/1/2) | Encoded | ✅ Yes (Academic model) |
| `study_hours_per_day` | Daily study hours | float | 0–24 | Behavioural | `studytime` (ordinal 1–4) | `Hours_Studied / 7` (weekly to daily) | Transformed | ✅ Yes (Engagement score) |
| `homework_completion` | Percentage of homework completed | float | 0–100 | Behavioural | `studytime × 20` (proxy estimate) | `Hours_Studied × 5` (proxy, clamped to 100) | Proxy-based | ✅ Yes (Engagement score) |
| `class_participation` | Level of classroom participation | float | 0–5 | Behavioural | `activities` (yes/no → 1/0, binary proxy) | `Motivation_Level` (Low/Med/High → 0/1/2) | Proxy-based | ✅ Yes (Behavioural model) |
| `extra_lessons` | Number of extra lessons or tutoring sessions | int | ≥ 0 | Academic | `schoolsup` (yes/no → 1/0) | `Tutoring_Sessions` (numeric count) | Encoded (UCI) / Direct (Kaggle) | ✅ Yes (Behavioural model) |

### 3.5.2 Extended Features (Saved but Not Currently Used)

The Kaggle dataset alignment script produces additional features that are saved in the processed dataset but are **not** used by the current MVP models. These are reserved for future model expansion:

| Feature Name | Description | Data Type | Range | Category | Source Column | Used in MVP |
|---|---|---|---|---|---|---|
| `Sleep_hours` | Hours of sleep per night | int | 4–10 | Behavioural | `Sleep_Hours` | ❌ No |
| `Teacher_quality` | Perceived quality of teaching | int | 0–2 | Environmental | `Teacher_Quality` | ❌ No |
| `Peer_influence` | Influence of peers on academic performance | int | 0–2 | Behavioural | `Peer_Influence` | ❌ No |
| `Access_to_resources` | Access to learning resources | int | 0–2 | Environmental | `Access_to_Resources` | ❌ No |
| `Family_income` | Family income level | int | 0–2 | Support | `Family_Income` | ❌ No |
| `Internet_access` | Whether the student has internet access | int | 0–1 | Environmental | `Internet_Access` | ❌ No |
| `School_type` | Public or private school | int | 0–1 | Environmental | `School_Type` | ❌ No |
| `Learning_disabilities` | Whether the student has a learning disability | int | 0–1 | Behavioural | `Learning_Disabilities` | ❌ No |
| `Physical_activity` | Physical activity level | int | 0–6 | Behavioural | `Physical_Activity` | ❌ No |

---

## 3.6 Data Preprocessing

Data preprocessing is performed by two dedicated alignment scripts that transform the raw datasets into a standardised feature schema suitable for model training. The scripts responsible are:

- **`dataset_alignment.py`** — processes the UCI `student-mat.csv` dataset
- **`kaggle_dataset_alignment.py`** — processes the Kaggle `StudentPerformanceFactors.csv` dataset

### 3.6.1 Step-by-Step Preprocessing Pipeline

#### Step 1: Loading Raw CSV Files

Each script loads its respective CSV file using Pandas:
- UCI dataset: loaded with `sep=";"` (semicolon-separated)
- Kaggle dataset: loaded with default comma separation; column names are cleaned (stripped, lowercased, spaces replaced with underscores)

#### Step 2: Creating the Target Variable

- **UCI**: `Pass = 1` if `G3 ≥ 10`, else `Pass = 0`. The threshold of 10/20 represents the minimum passing grade.
- **Kaggle**: `Pass = 1` if `Exam_Score ≥ 60th percentile`, else `Pass = 0`. A percentile-based threshold is used to create a balanced class distribution.

#### Step 3: Feature Alignment and Transformation

Both datasets are transformed into a common feature schema. Key transformations include:

- **Academic scores** (UCI): `G1` → `Term1_avg`, `G2` → `Term2_avg` and `Seq5_score` (already on 0–20 scale).
- **Academic scores** (Kaggle): `Previous_Scores / 5` → `Term1_avg`, `Term2_avg`, and `Seq5_score` (normalised from 0–100 to 0–20 scale).
- **Attendance** (UCI): Calculated as `100 - (absences / max_absences × 100)` to convert raw absence counts to a percentage.
- **Attendance** (Kaggle): `Attendance` column used directly (already a percentage).
- **Study hours** (UCI): `studytime` ordinal values (1–4) mapped directly to hours.
- **Study hours** (Kaggle): `Hours_Studied / 7` to convert weekly hours to daily hours.
- **Homework completion**: Proxy feature — `studytime × 20` (UCI) and `Hours_Studied × 5` (Kaggle), clamped to a maximum of 100.

#### Step 4: Encoding Categorical Variables

- **Binary variables** (UCI): `schoolsup`, `famsup`, `activities` mapped from "yes"/"no" to 1/0.
- **Ordinal variables** (Kaggle): `Motivation_Level`, `Parental_Involvement`, `Teacher_Quality`, `Access_to_Resources`, `Family_Income` mapped as Low=0, Medium=1, High=2.
- **Binary variables** (Kaggle): `Internet_Access`, `Learning_Disabilities` mapped from "yes"/"no" to 1/0.
- **School type** (Kaggle): "public"=0, "private"=1.
- **Peer influence** (Kaggle): "negative"=0, "neutral"=1, "positive"=2.

#### Step 5: Handling Missing Values

Both scripts employ **median imputation** for numeric columns and **mode imputation** for categorical columns:

```python
# Numeric columns: fill missing with median
for col in numeric_cols:
    if df_aligned[col].isnull().any():
        median_val = df_aligned[col].median()
        df_aligned[col] = df_aligned[col].fillna(median_val)
```

This approach was chosen over naive `fillna(0)` to avoid distorting the distribution of features.

#### Step 6: Validation and Quality Checks

Both scripts verify:
- Score ranges are within expected bounds (0–20 for academic scores)
- No missing values remain after imputation
- Class distribution is printed for verification

#### Step 7: Saving Processed Data

- UCI → `data/processed/aligned_student_data.csv` (395 rows, 10 columns)
- Kaggle → `data/processed/aligned_kaggle_data_full.csv` (6,607 rows, 19 columns)

### 3.6.2 Proxy Features and Limitations

Several features in the processed datasets are **proxy-based** rather than directly measured:

| Feature | Proxy Source | Limitation |
|---|---|---|
| `Seq5_score` | Duplicates `G2` (UCI) or `Previous_Scores` (Kaggle) | Not a real Sequence 5 exam score; provides no additional information beyond Term 2 average |
| `Homework_completion` | Derived from `studytime` (UCI) or `Hours_Studied` (Kaggle) | Not actual homework submission data; assumes linear relationship between study hours and homework completion |
| `Class_participation` | Mapped from `activities` (UCI, binary) or `Motivation_Level` (Kaggle, ordinal) | Binary flag for extracurricular activities is a poor proxy for continuous class participation |

These proxy features are documented as limitations and would be replaced with real school data in a production deployment.

---

## 3.7 Target Variable Creation

### 3.7.1 Definition of Pass and Fail

The target variable `Pass` is binary:

| Label | Value | Meaning |
|---|---|---|
| Pass | 1 | The student is predicted to pass the final examination |
| Fail | 0 | The student is predicted to fail the final examination |

### 3.7.2 Thresholds by Dataset

| Dataset | Source Variable | Pass Threshold | Rationale |
|---|---|---|---|
| UCI | `G3` (final grade, 0–20) | G3 ≥ 10 | 10/20 is the standard minimum passing grade in the Portuguese and Cameroonian grading systems |
| Kaggle | `Exam_Score` (0–100) | Exam_Score ≥ 60th percentile | Percentile-based threshold produces a more balanced class distribution (approximately 40% Pass, 60% Fail) |

### 3.7.3 Probability and Risk Level

Beyond the binary pass/fail decision, the model produces:

- **Probability of passing**: A calibrated probability between 0.0 and 1.0, indicating the model's confidence.
- **Risk level classification**:
  - **Low Risk**: Probability ≥ 0.70 — student is likely to pass
  - **Medium Risk**: Probability between 0.45 and 0.69 — student may be borderline
  - **High Risk**: Probability below 0.45 — student is at significant risk of failure

### 3.7.4 Suitability for Early Warning

Pass/fail prediction is appropriate for early warning in secondary schools because:

1. It provides a clear, actionable signal for teachers and parents.
2. The probability score allows for nuanced decision-making (not just binary).
3. Risk levels enable prioritisation of interventions for the most at-risk students.
4. The model uses mid-year data (Term 1, Term 2, Sequence 5) to predict end-of-year outcomes, allowing sufficient time for intervention.

---

## 3.8 Model Development Method

### 3.8.1 Dual-Model Ensemble Architecture

The prediction system employs a **weighted dual-model ensemble** that combines two independently trained models:

```
                    ┌─────────────────────┐
                    │  Student Input Data  │
                    └──────────┬──────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
   ┌───────────────────────┐     ┌───────────────────────────┐
   │   Academic Model      │     │   Behavioural Model       │
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

### 3.8.2 Academic Model

| Property | Detail |
|---|---|
| **Algorithm** | Logistic Regression |
| **Implementation** | `sklearn.linear_model.LogisticRegression` (max_iter=1000, random_state=42) |
| **Calibration** | Platt scaling via `CalibratedClassifierCV` (method="sigmoid", cv=5) |
| **Training Data** | `aligned_student_data.csv` (UCI, 395 rows) |
| **Features** (5) | `Term1_avg`, `Term2_avg`, `Seq5_score`, `Attendance_percentage`, `Parental_support` |
| **Ensemble Weight** | 70% |

**Justification**: Logistic Regression was selected for its interpretability, computational efficiency, and suitability for linearly separable academic data. Its natural probabilistic output makes it well-suited for calibrated probability estimation.

### 3.8.3 Behavioural Model

| Property | Detail |
|---|---|
| **Algorithm** | XGBoost (Extreme Gradient Boosting) |
| **Implementation** | `xgboost.XGBClassifier` |
| **Hyperparameters** | n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss' |
| **Calibration** | Platt scaling via `CalibratedClassifierCV` (method="sigmoid", cv=5) |
| **Training Data** | `aligned_kaggle_data_full.csv` (Kaggle, 6,607 rows) |
| **Features** (4) | `Engagement_score` (engineered), `Attendance_percentage`, `Extra_lessons`, `Class_participation` |
| **Ensemble Weight** | 30% |

**Justification**: XGBoost was selected for its ability to capture non-linear relationships among engagement, attendance, and participation variables. The regularisation parameters (`max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`) were configured to mitigate overfitting on the larger Kaggle dataset.

### 3.8.4 Feature Engineering: Engagement Score

The behavioural model uses an engineered **Engagement Score** that combines two raw features into a single composite indicator:

1. `Study_hours_per_day` and `Homework_completion` are normalised using **MinMaxScaler** (fitted on training data).
2. The engagement score is calculated as the **mean** of the two scaled values: `Engagement_score = (scaled_study_hours + scaled_homework) / 2`.

This reduces feature redundancy (study hours and homework are correlated) and provides a single measure of student engagement.

### 3.8.5 Probability Calibration

Both models undergo **Platt scaling** (sigmoid calibration) using `CalibratedClassifierCV` with 5-fold cross-validation. This post-hoc calibration ensures that predicted probabilities closely approximate true class frequencies. Well-calibrated probabilities are essential for a system where probability thresholds are used to drive intervention decisions.

### 3.8.6 The Final Probability Formula

```
P(pass) = 0.7 × P_academic(pass) + 0.3 × P_behavioural(pass)
```

The 70/30 weighting reflects the primacy of academic performance in predicting outcomes, while behavioural indicators provide complementary signals that may detect risk factors before they manifest as poor grades.

### 3.8.7 Decision Threshold and Risk Classification

- **Pass/Fail threshold**: `P(pass) ≥ 0.5` → Pass; `P(pass) < 0.5` → Fail
- **Risk levels**:
  - Low Risk: `P(pass) ≥ 0.70`
  - Medium Risk: `0.45 ≤ P(pass) < 0.70`
  - High Risk: `P(pass) < 0.45`

---

## 3.9 Training Procedure

### 3.9.1 Overview

The complete training pipeline is orchestrated by `train_models.py`, which sequentially executes the two training scripts.

### 3.9.2 Detailed Steps

| Step | Action | Script |
|---|---|---|
| 1 | Load processed CSV data | `train_academic_model.py` / `train_behavioral_model.py` |
| 2 | Select features and target variable | Hardcoded feature lists in each script |
| 3 | Guard against missing values (median imputation) | Both scripts |
| 4 | Feature engineering (behavioural only) | `MinMaxScaler` → Engagement Score |
| 5 | Stratified train/test split (80/20, random_state=42) | Both scripts |
| 6 | Train base model (Logistic Regression / XGBoost) | Both scripts |
| 7 | Wrap with `CalibratedClassifierCV` (sigmoid, 5-fold CV) | Both scripts |
| 8 | Evaluate on test set (accuracy, ROC-AUC, Brier score) | Both scripts |
| 9 | Save model artifact (.pkl) and metrics (.json) | Both scripts |

### 3.9.3 Files Generated After Training

| File | Description |
|---|---|
| `models/academic_model.pkl` | Serialised calibrated Logistic Regression model |
| `models/behavioral_model.pkl` | Serialised calibrated XGBoost model |
| `models/engagement_scaler.pkl` | Serialised MinMaxScaler for engagement score computation |
| `reports/academic_model_metrics.json` | Academic model metrics (accuracy, ROC-AUC, Brier score) |
| `reports/behavioral_model_metrics.json` | Behavioural model metrics (accuracy, ROC-AUC, Brier score) |

### 3.9.4 Commands to Retrain

```bash
# Step 1: Preprocess datasets (if raw data has changed)
python dataset_alignment.py
python kaggle_dataset_alignment.py

# Step 2: Train both models
python train_models.py --non-interactive
```

### 3.9.5 Training Scripts and Their Roles

| Script | Role |
|---|---|
| `dataset_alignment.py` | Preprocesses UCI raw data → `data/processed/aligned_student_data.csv` |
| `kaggle_dataset_alignment.py` | Preprocesses Kaggle raw data → `data/processed/aligned_kaggle_data_full.csv` |
| `src/models/train_academic_model.py` | Trains and calibrates the academic model, saves `.pkl` and metrics |
| `src/models/train_behavioral_model.py` | Trains and calibrates the behavioural model, saves `.pkl`, scaler, and metrics |
| `train_models.py` | Pipeline orchestrator that runs both training scripts sequentially |

---

## 3.10 Model Evaluation

### 3.10.1 Evaluation Procedure

The model evaluation is performed by `evaluate_model.py`, which:

1. Loads both trained models and the engagement scaler.
2. Loads the processed UCI dataset (`aligned_student_data.csv`).
3. Prepares academic and behavioural features.
4. Generates calibrated probabilities from both models.
5. Computes the final ensemble probability using the weighted formula.
6. Calculates comprehensive metrics and saves them to `reports/model_metrics.json` and `reports/model_evaluation.md`.

### 3.10.2 Evaluation Results

The following evaluation results were obtained from the repository (generated on 28 April 2026):

#### Overall Ensemble Performance

| Metric | Value | Explanation |
|---|---|---|
| **Accuracy** | 88.35% | Proportion of correct predictions (both Pass and Fail) |
| **Precision** | 95.06% | Of students predicted to pass, 95.06% actually passed (low false positive rate) |
| **Recall (Pass class)** | 87.17% | Of students who actually passed, 87.17% were correctly identified |
| **Recall (Fail/Risk class)** | **90.77%** 🎯 | Of students who actually failed, **90.77% were correctly identified** — critical for early warning |
| **F1-Score** | 90.94% | Harmonic mean of precision and recall, indicating balanced performance |
| **ROC-AUC** | 0.9554 | The model has excellent ability to distinguish between Pass and Fail students |

#### Individual Model Performance

| Metric | Academic Model | Behavioural Model |
|---|---|---|
| Accuracy | 83.54% | 83.28% |
| ROC-AUC | 0.9572 | 0.9221 |
| Brier Score | 0.0806 | 0.1164 |

#### Confusion Matrix

```
              Predicted Fail  Predicted Pass
Actual Fail         118           12
Actual Pass          34          231
```

- **True Negatives (correctly identified failures)**: 118
- **False Positives (failures predicted as pass)**: 12
- **False Negatives (passes predicted as fail)**: 34
- **True Positives (correctly identified passes)**: 231

### 3.10.3 Interpretation of Key Metrics

**Fail Recall (90.77%)**: This is the most critical metric for an early warning system. It measures the system's ability to correctly identify students who are genuinely at risk of failure. A recall of 90.77% means that approximately 91 out of every 100 at-risk students would be correctly flagged by the system. Only about 9% of truly at-risk students would be missed (false negatives). In an educational context, this high recall rate ensures that the vast majority of struggling students receive timely intervention.

**ROC-AUC (0.9554)**: A value above 0.95 indicates excellent discriminative ability. The model is highly effective at distinguishing between students who will pass and those who will fail.

**Brier Score**: The academic model's lower Brier score (0.0806) compared to the behavioural model (0.1164) indicates better probability calibration, justifying its higher weight (70%) in the ensemble.

---

## 3.11 Prediction Workflow

The following steps describe the complete prediction workflow from input to output:

| Step | Action | Component |
|---|---|---|
| 1 | User or backend system sends student data as a JSON POST request | External client |
| 2 | FastAPI validates input using Pydantic schema (`StudentInput`) with field constraints (ranges, types) | `app.py` |
| 3 | Additional validation is performed by `validate_input()` in the prediction system | `predict_system.py` |
| 4 | Academic features are extracted: `[term1_avg, term2_avg, seq5_score, attendance_percentage, parental_support]` | `predict_system.py` |
| 5 | Engagement score is computed from `study_hours_per_day` and `homework_completion` using the saved MinMaxScaler | `predict_system.py` |
| 6 | Behavioural features are assembled: `[engagement_score, attendance_percentage, extra_lessons, class_participation]` | `predict_system.py` |
| 7 | Academic model predicts `P_academic(pass)` | `predict_system.py` |
| 8 | Behavioural model predicts `P_behavioural(pass)` | `predict_system.py` |
| 9 | Final probability is calculated: `P = 0.7 × P_academic + 0.3 × P_behavioural` | `predict_system.py` |
| 10 | Pass/Fail decision is made: Pass if `P ≥ 0.5`, Fail otherwise | `predict_system.py` |
| 11 | Risk level is assigned: Low (≥0.70), Medium (0.45–0.69), High (<0.45) | `predict_system.py` |
| 12 | Rule-based explanations are generated based on input thresholds | `predict_system.py` |
| 13 | Confidence message is generated based on probability range | `predict_system.py` |
| 14 | API returns the complete prediction response as JSON | `app.py` |

### 3.11.1 Rule-Based Explanation Logic

The system generates human-readable explanations based on the following rules:

| Condition | Explanation Generated |
|---|---|
| Attendance < 60% | "Critical: Very low attendance (below 60%)" |
| Attendance < 80% | "Low attendance (below 80%)" |
| Average of term1 and term2 < 10 | "Weak academic foundation: Average scores below 10/20" |
| Seq5 score < 10 | "Downward trend: Sequence 5 score is failing" |
| Study hours < 1.5/day | "Insufficient study time (below 1.5h/day)" |
| Homework completion < 50% | "Poor homework completion (below 50%)" |
| Parental support = 0 | "Lack of parental support at home" |
| Probability ≥ 0.7 and no flags | "Consistent performance across all metrics" |
| No specific flags triggered | "Performances are borderline across multiple factors" |

---

## 3.12 API Implementation

### 3.12.1 Overview

The prediction model is served through a **FastAPI** RESTful API, defined in `app.py`. The API provides single and batch prediction endpoints, a health check, and informational endpoints.

### 3.12.2 All Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Root endpoint — returns API name, version, and docs link |
| `GET` | `/health` | Health check — returns status and timestamp |
| `POST` | `/predict` | Single student prediction |
| `POST` | `/predict/batch` | Batch prediction for multiple students |
| `GET` | `/info` | Detailed API information including thresholds and features |
| `GET` | `/docs` | Swagger UI interactive documentation (auto-generated) |
| `GET` | `/redoc` | ReDoc documentation (auto-generated) |

### 3.12.3 Single Prediction — Request Body

```json
{
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
```

All fields are required with the following constraints:

| Field | Type | Min | Max |
|---|---|---|---|
| `term1_avg` | float | 0 | 20 |
| `term2_avg` | float | 0 | 20 |
| `seq5_score` | float | 0 | 20 |
| `attendance_percentage` | float | 0 | 100 |
| `parental_support` | int | 0 | 1 |
| `study_hours_per_day` | float | 0 | 24 |
| `homework_completion` | float | 0 | 100 |
| `class_participation` | float | 0 | 5 |
| `extra_lessons` | int | 0 | — |

### 3.12.4 Single Prediction — Response Body

```json
{
    "prediction": "Pass",
    "probability": 0.723,
    "risk_level": "Low Risk",
    "academic_prob": 0.801,
    "behavioral_prob": 0.542,
    "confidence": "Moderate confidence prediction",
    "explanations": [
        "Consistent performance across all metrics"
    ]
}
```

### 3.12.5 Batch Prediction

The batch endpoint (`POST /predict/batch`) accepts an array of students and returns individual predictions for each:

**Request:**
```json
{
    "students": [
        { "term1_avg": 15, "term2_avg": 14, ... },
        { "term1_avg": 6, "term2_avg": 7, ... }
    ]
}
```

**Response:**
```json
{
    "predictions": [
        { "index": 0, "status": "success", "result": { ... } },
        { "index": 1, "status": "success", "result": { ... } }
    ],
    "count": 2,
    "timestamp": "2026-04-28T12:00:00"
}
```

Each item includes an `index`, `status` ("success" or "error"), and either the full prediction `result` or an error `message`.

### 3.12.6 Future Integration Points

The API is designed to serve the following future system components:

| Consumer | Integration Method | Use Case |
|---|---|---|
| **Django Backend** | HTTP POST to `/predict` or `/predict/batch` | Central backend aggregates student data from school records and sends to prediction API |
| **Teacher Dashboard** | Via Django backend | Teachers view individual and class-level predictions, risk levels, and explanations |
| **Parent Dashboard** | Via Django backend | Parents view their child's risk status and recommendations |
| **School Administrator Panel** | Via Django backend | Administrators monitor school-wide risk distributions and intervention effectiveness |

> **Note:** These dashboards are not implemented in this repository. They are planned for future development as separate applications that consume this API.

---

## 3.13 Model Use Cases

The model supports the following use cases:

1. **Predicting whether a student is likely to pass or fail** — core binary classification.
2. **Detecting high-risk students early** — using probability thresholds and risk levels.
3. **Helping teachers identify students needing support** — through the teacher dashboard (future integration).
4. **Supporting class-level risk analysis** — through batch prediction enabling teachers to assess an entire class at once.
5. **Helping school administrators monitor academic risk** — aggregated risk data across classes and year groups.
6. **Supporting parent notification systems** — flagging at-risk students for parental engagement (future).
7. **Supporting personalised revision planning** — recommendations based on weak areas identified by the model (future).
8. **Supporting examination preparedness monitoring** — tracking student readiness across sequences.
9. **Supporting academic decision-making using data** — providing evidence-based insights rather than subjective judgements.

---

## 3.14 Model Expansion and Future Improvements

The following improvements are planned for future iterations of the model:

| Improvement | Description |
|---|---|
| **Real Cameroonian data** | Training with anonymised data from Cameroonian secondary schools to improve relevance and accuracy |
| **Subject-specific prediction** | Separate models for different subjects (Mathematics, English, Sciences, etc.) |
| **Term-by-term progression analysis** | Tracking student performance across all six sequences to identify trends |
| **SHAP explainability** | Replacing rule-based explanations with SHAP (SHapley Additive exPlanations) for model-intrinsic feature importance |
| **Recommendation engine** | Generating personalised revision plans based on identified weak areas |
| **Prediction history** | Storing prediction results in a database for longitudinal tracking |
| **Continuous retraining** | Automatically retraining models as new cohort data becomes available |
| **Model monitoring** | Implementing drift detection and performance monitoring in production |
| **Multi-school support** | Supporting multiple schools with school-specific or federated models |
| **Fairness and bias checks** | Auditing models for demographic biases (gender, socioeconomic status, school type) |
| **Security and privacy** | Implementing authentication, encryption, and data protection measures for student data |

---

## 3.15 Ethical Considerations

### 3.15.1 Student Data Privacy

The system processes sensitive academic and behavioural data that constitutes personally identifiable information (PII). In a production deployment:

- All student data must be **anonymised** or **pseudonymised** before storage and processing.
- Data must be stored securely with appropriate encryption at rest and in transit.
- Access to prediction results must be restricted to authorised personnel (teachers, administrators, parents).
- The system must comply with applicable data protection regulations.

### 3.15.2 Responsible Use of Predictions

- The model should **support** teachers and school administrators, **not replace** professional judgement.
- Predictions are probabilistic estimates, not definitive outcomes. They should be used as one input among many in academic decision-making.
- No student should be denied educational opportunities solely based on model predictions.

### 3.15.3 Avoiding Unfair Labelling

- Labelling a student as "High Risk" could lead to stigmatisation if not handled sensitively.
- Conversely, failing to identify an at-risk student (false negative) could result in missed intervention opportunities.
- **Human review** must be conducted before any intervention is initiated based on model predictions.

### 3.15.4 Consent and Transparency

- Students and parents should be informed about the use of AI in academic monitoring.
- The rule-based explanations provided by the system support transparency by making predictions interpretable.

### 3.15.5 Bias Considerations

- The model is trained on European/synthetic datasets and deployed for Cameroonian students, which may introduce cultural and systemic biases.
- Regular bias audits should be conducted once real school data is available.

---

## 3.16 Limitations of the Current Model

The following limitations are acknowledged:

1. **Public datasets may not fully represent Cameroonian secondary schools.** The UCI dataset contains Portuguese student records, and the Kaggle dataset is synthetic/survey-based. Neither captures the specific curricular, cultural, and socioeconomic context of Cameroon.

2. **Several features are proxy-based.** `Seq5_score` duplicates `Term2_avg`; `Homework_completion` is derived from study hours rather than actual submissions; `Class_participation` is mapped from binary extracurricular activity flags.

3. **Real school deployment requires local validation.** The model's performance on real Cameroonian data is unknown and must be validated before production deployment.

4. **Predictions are probabilistic, not absolute truth.** Model outputs are statistical estimates based on historical patterns and should not be treated as certainties.

5. **Model performance depends on data quality.** Inaccurate or incomplete student records will degrade prediction accuracy.

6. **Target variable inconsistency.** The two datasets define "Pass" differently — UCI uses an absolute threshold (G3 ≥ 10), while Kaggle uses a relative threshold (60th percentile). This means the two models are trained with slightly different definitions of success.

7. **Scale considerations.** The academic model is trained on 0–20 scale data (UCI), while the behavioural model is trained on data that was normalised from 0–100 to 0–20 (Kaggle). The engagement scaler was fit on Kaggle-scale data.

8. **No class weighting or resampling.** Neither model applies techniques such as SMOTE or class weights to address class imbalance.

9. **Fixed prediction threshold.** The 0.5 threshold is not configurable and may not be optimal for minimising false negatives in an early warning context.

10. **More testing is needed with real school records** to ensure the model generalises well to the target population.

---

## 3.17 Summary

This chapter has presented the materials and methods used in the development of the AI-based Student Performance Prediction Model. The key elements are summarised as follows:

- **Materials**: The model was developed using Python 3.12, Scikit-learn, XGBoost, FastAPI, and related libraries, on a standard development machine.
- **Datasets**: Two publicly available datasets were used — the UCI Student Performance Dataset (395 records, real academic data) and the Kaggle Student Performance Factors Dataset (6,607 records, synthetic/survey-based behavioural data).
- **Preprocessing**: Raw datasets were transformed into a standardised feature schema through alignment scripts that handle feature mapping, score normalisation (0–20 scale), categorical encoding, and median/mode imputation.
- **Model Architecture**: A dual-model weighted ensemble combining a calibrated Logistic Regression (academic, 70% weight) and a calibrated XGBoost (behavioural, 30% weight).
- **Evaluation**: The ensemble achieves 88.35% accuracy, 0.9554 ROC-AUC, and critically, **90.77% recall for the Fail class**, ensuring the vast majority of at-risk students are identified.
- **API**: A FastAPI service with single and batch prediction endpoints, risk level classification, rule-based explanations, and comprehensive input validation.
- **Deployment**: Docker support is available through a Dockerfile for containerised deployment.

The model is designed to serve as the prediction engine for the larger AI-based Student Performance Prediction and Early Warning System, which will include teacher dashboards, parent notifications, and school administrator panels in future development phases.

---

## Information Available for Project Report Writing

The following academic information can be directly reused or adapted for the project report:

### Materials Used
- Python 3.12, Scikit-learn (≥1.2.0), XGBoost (≥1.7.0), FastAPI (≥0.104.0), Uvicorn, Pandas, NumPy, Pydantic, Joblib, Pytest, Docker.
- Standard laptop/desktop computer with at least 4 GB RAM.

### Dataset Sources
- UCI Student Performance Dataset (Cortez & Silva, 2008) — 395 records, 33 attributes, Mathematics subject, 0–20 grading scale.
- Kaggle Student Performance Factors Dataset — 6,607 records, 20 attributes, synthetic/survey-based.

### Data Preprocessing Method
- Feature alignment to standardised schema, score normalisation (0–100 → 0–20), categorical encoding (binary and ordinal), median imputation for missing numeric values, mode imputation for missing categorical values. Processed by `dataset_alignment.py` and `kaggle_dataset_alignment.py`.

### Model Training Method
- Dual-model ensemble: Calibrated Logistic Regression (academic, 70% weight) + Calibrated XGBoost (behavioural, 30% weight).
- Platt scaling (sigmoid calibration) via `CalibratedClassifierCV` with 5-fold cross-validation.
- 80/20 stratified train/test split with `random_state=42`.
- Feature engineering: Engagement Score from MinMaxScaler normalisation of study hours and homework completion.

### Model Evaluation Method
- Evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Special emphasis on Fail-class recall (90.77%) as the critical metric for early warning.
- Ensemble accuracy: 88.35%, ROC-AUC: 0.9554.

### API Implementation Method
- FastAPI RESTful API with Pydantic input validation.
- Endpoints: single prediction (`POST /predict`), batch prediction (`POST /predict/batch`), health check (`GET /health`), information (`GET /info`).
- Risk level classification (Low/Medium/High) and rule-based explanations included in response.

### Limitations
- Public datasets not representative of Cameroonian schools; proxy features; no real student data; fixed threshold; target variable inconsistency between datasets; no class resampling.

### Ethical Considerations
- Student data privacy and anonymisation; responsible use of predictions; avoiding unfair labelling; need for human review; cultural bias risk from training on non-local data; consent and transparency.

### Future Work
- Real Cameroonian data collection; subject-specific models; SHAP explainability; recommendation engine; prediction history; continuous retraining; model monitoring; multi-school support; fairness audits; security hardening.
