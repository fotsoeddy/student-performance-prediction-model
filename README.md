# 🧠 Student Success Prediction System

A machine learning system that predicts the **probability that a student will pass Sequence 6 (final sequence)** using a dual-model architecture combining academic performance and behavioral patterns.

## 🎯 Objective

Provide early prediction of student success by analyzing:
- 📊 **Academic Performance** — Past results and academic trajectory
- 📈 **Behavioral Data** — Study habits, engagement, and attendance

## 🏗️ System Architecture

```
Frontend (UI)
    ↓
Backend (Data Aggregation)
    ↓
Prediction API (THIS PROJECT)
    ↓
Machine Learning Models
```

## 🧠 Core Concept

| Model | Algorithm | Purpose | Weight |
|-------|-----------|---------|--------|
| Academic Model | Logistic Regression (calibrated) | Evaluates past academic performance | 70% |
| Behavioral Model | XGBoost (calibrated) | Evaluates student habits and engagement | 30% |

```
Final Probability = 0.7 × Academic Probability + 0.3 × Behavioral Probability
```

## 📦 Project Structure

```
student-success-prediction/
│
├── data/
│   ├── raw/                           # Original datasets
│   └── processed/                     # Aligned and cleaned data
│
├── models/
│   ├── academic_model.pkl             # Trained academic model
│   ├── behavioral_model.pkl           # Trained behavioral model
│   └── engagement_scaler.pkl          # Scaler for engagement score
│
├── src/
│   └── models/
│       ├── train_academic_model.py    # Academic model training
│       ├── train_behavioral_model.py  # Behavioral model training
│       └── predict_system.py          # Core prediction logic
│
├── app.py                             # FastAPI application
├── test_api.py                        # API test suite
├── train_models.py                    # Run full training pipeline
├── dataset_alignment.py               # UCI dataset preprocessing
├── kaggle_dataset_alignment.py        # Kaggle dataset preprocessing
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Quick Start

### Step 1: Clone Repository
```bash
git clone https://github.com/fotsoeddy/student-performance-prediction-model.git
cd student-performance-prediction-model
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the API
```bash
uvicorn app:app --reload
```

### Step 5: Access the API
- **API Base URL:** http://127.0.0.1:8000
- **Interactive Docs:** http://127.0.0.1:8000/docs
- **Alternative Docs:** http://127.0.0.1:8000/redoc

---

## 📡 API Usage

### Endpoint

```
POST /predict
```

### Request Body

```json
{
  "term1_avg": 10.0,
  "term2_avg": 10.0,
  "seq5_score": 10.0,
  "attendance_percentage": 70.0,
  "parental_support": 0,
  "study_hours_per_day": 2.5,
  "homework_completion": 65.0,
  "class_participation": 2.5,
  "extra_lessons": 0
}
```

### Response

```json
{
  "prediction": "Pass",
  "probability": 0.612,
  "academic_prob": 0.750,
  "behavioral_prob": 0.320
}
```

### cURL Example
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "term1_avg": 10.0,
    "term2_avg": 10.0,
    "seq5_score": 10.0,
    "attendance_percentage": 70.0,
    "parental_support": 0,
    "study_hours_per_day": 2.5,
    "homework_completion": 65.0,
    "class_participation": 2.5,
    "extra_lessons": 0
  }'
```

### Python Example
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "term1_avg": 10.0,
        "term2_avg": 10.0,
        "seq5_score": 10.0,
        "attendance_percentage": 70.0,
        "parental_support": 0,
        "study_hours_per_day": 2.5,
        "homework_completion": 65.0,
        "class_participation": 2.5,
        "extra_lessons": 0
    }
)
print(response.json())
```

### JavaScript/Fetch Example
```javascript
fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    term1_avg: 10.0,
    term2_avg: 10.0,
    seq5_score: 10.0,
    attendance_percentage: 70.0,
    parental_support: 0,
    study_hours_per_day: 2.5,
    homework_completion: 65.0,
    class_participation: 2.5,
    extra_lessons: 0
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root message |
| GET | `/health` | Health check |
| POST | `/predict` | Make prediction |
| GET | `/info` | API and model information |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/redoc` | Alternative documentation |

---

## 📊 Input Features

### 🎓 Academic Features

| Field | Description | Type | Range | Source |
|-------|-------------|------|-------|--------|
| `term1_avg` | Term 1 average score | float | 0–20 | School records |
| `term2_avg` | Term 2 average score | float | 0–20 | School records |
| `seq5_score` | Sequence 5 score | float | 0–20 | School records |
| `attendance_percentage` | Attendance percentage | float | 0–100 | Aggregated |
| `parental_support` | Parental support | int | 0 or 1 | Student profile |

### 📈 Behavioral Features

| Field | Description | Type | Range | Source |
|-------|-------------|------|-------|--------|
| `study_hours_per_day` | Avg daily study hours | float | 0–24 | Aggregated |
| `homework_completion` | Homework completion % | float | 0–100 | Aggregated |
| `class_participation` | Participation level | float | 0–5 | Aggregated |
| `extra_lessons` | Number of extra lessons | int | ≥ 0 | School records |

> `study_hours_per_day` and `homework_completion` are internally combined into an **Engagement Score** before being passed to the behavioral model. You still send them as raw values — the API handles the transformation.

---

## ⚠️ Data Requirements

### ✅ Send Aggregated Values
```json
{
  "attendance_percentage": 85.0,
  "study_hours_per_day": 3.2,
  "homework_completion": 78.5,
  "class_participation": 3.0
}
```

### ❌ Do Not Send Raw Daily Records
```json
{
  "attendance": [1, 0, 1, 1],
  "study_hours": [2, 3, 4, 2]
}
```

---

## 🔄 Data Flow

```
1. Teacher inputs daily records
   ↓
2. Backend aggregates (attendance %, avg study hours, etc.)
   ↓
3. Backend combines with academic data (term scores)
   ↓
4. POST /predict with aggregated features
   ↓
5. API returns prediction + probabilities
```

---

## 🧪 Testing

```bash
# Start API first
uvicorn app:app --reload

# In a new terminal
python test_api.py
```

---

## 🎯 Model Details

### Academic Model
- Algorithm: Logistic Regression with Platt calibration
- AUC: ~0.97
- Features: 5 academic indicators

### Behavioral Model
- Algorithm: XGBoost with Platt calibration
- AUC: ~0.92
- Features: 4 (engagement score + attendance + extra lessons + participation)
- Engagement Score = normalized average of study hours and homework completion

---

## 🔧 Retrain Models (Optional)

Pre-trained models are included. To retrain from scratch:

```bash
# Preprocess raw data
python dataset_alignment.py
python kaggle_dataset_alignment.py

# Train both models
python train_models.py
```

---

## 🚀 Production Deployment

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🧠 Backend Integration Guide

Aggregate raw daily data before calling the API:

```python
attendance_percentage = (days_present / total_days) * 100
study_hours_per_day   = sum(daily_study_hours) / num_days
homework_completion   = sum(homework_scores) / num_assignments
class_participation   = sum(participation_scores) / num_days
```

---

## 🔐 CORS

CORS is open by default. For production, restrict origins in `app.py`:

```python
allow_origins=["https://your-frontend-domain.com"]
```

---

## 📚 Data Sources

- UCI Student Performance Dataset
- Kaggle Student Performance Factors Dataset

---

## 📄 License

MIT License
