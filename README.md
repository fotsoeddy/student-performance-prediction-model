# 🧠 Student Success Prediction System

A machine learning system that predicts the **probability that a student will pass Sequence 6 (final sequence)** using dual-model architecture combining academic performance and behavioral patterns.

## 🎯 Objective

This system provides early prediction of student success by analyzing:
- 📊 **Academic Performance** - Past results and academic trajectory
- 📈 **Behavioral Data** - Daily habits, study patterns, and engagement

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

The system uses **two complementary machine learning models**:

| Model | Algorithm | Purpose | Weight |
|-------|-----------|---------|--------|
| Academic Model | Logistic Regression | Evaluates past academic performance | 60% |
| Behavioral Model | XGBoost | Evaluates student habits and engagement | 40% |

**Final Prediction Formula:**
```
Final Probability = 0.6 × Academic Probability + 0.4 × Behavioral Probability
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
│   ├── academic_model.pkl             # Pre-trained academic model
│   └── behavioral_model.pkl           # Pre-trained behavioral model
│
├── src/
│   └── models/
│       ├── train_academic_model.py    # Academic model training
│       ├── train_behavioral_model.py  # Behavioral model training
│       └── predict_system.py          # Core prediction logic
│
├── app.py                             # FastAPI application (API server)
├── test_api.py                        # API test suite
├── dataset_alignment.py               # UCI dataset preprocessing
├── kaggle_dataset_alignment.py        # Kaggle dataset preprocessing
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Quick Start

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd student-success-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate it
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

### Make a Prediction

**Endpoint:** `POST /predict`

#### cURL Example
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

#### Python Example
```python
import requests

url = "http://127.0.0.1:8000/predict"

data = {
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

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.1%}")
```

#### JavaScript/Fetch Example
```javascript
fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    term1_avg: 12.0,
    term2_avg: 13.0,
    seq5_score: 14.0,
    attendance_percentage: 85.0,
    parental_support: 1,
    study_hours_per_day: 3.0,
    sleep_hours: 7.0,
    class_participation: 3,
    homework_completion: 80.0,
    extra_lessons: 1
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

#### Response
```json
{
  "prediction": "Pass",
  "probability": 0.867,
  "academic_prob": 0.950,
  "behavioral_prob": 0.742
}
```

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root message |
| GET | `/health` | Health check |
| POST | `/predict` | Make prediction |
| GET | `/info` | API information |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/redoc` | Alternative documentation |

---

## 📊 Input Features Specification

### 🎓 Academic Features

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `term1_avg` | Term 1 average score | float | 0-20 | School records |
| `term2_avg` | Term 2 average score | float | 0-20 | School records |
| `seq5_score` | Sequence 5 score | float | 0-20 | School records |
| `attendance_percentage` | Attendance percentage | float | 0-100 | Aggregated from daily records |
| `parental_support` | Parental support level | int | 0-1 | Student profile |

### 📈 Behavioral Features

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `study_hours_per_day` | Average daily study time | float | 0-24 | Aggregated from daily logs |
| `sleep_hours` | Average sleep hours | float | 0-24 | Student self-report |
| `class_participation` | Average participation level | int | 0-5 | Aggregated from daily records |
| `homework_completion` | Average homework completion % | float | 0-100 | Aggregated from daily records |
| `extra_lessons` | Number of tutoring sessions | int | ≥0 | School records |

---

## ⚠️ CRITICAL: Data Requirements

### ✅ CORRECT - Send Aggregated Values
```python
{
    "attendance_percentage": 85.0,      # Calculated: (days_present / total_days) * 100
    "study_hours_per_day": 3.2,         # Calculated: average(daily_study_hours)
    "homework_completion": 78.5,        # Calculated: average(homework_scores)
    "class_participation": 3            # Calculated: average(participation_scores)
}
```

### ❌ WRONG - Don't Send Raw Daily Data
```python
{
    "attendance": [1, 0, 1, 1, 1],      # ❌ Raw daily records
    "study_hours": [2, 3, 4, 2, 3]      # ❌ Raw daily records
}
```

---

## 🔄 Data Flow

```
1. Teacher inputs daily records
   ↓
2. Backend aggregates data (attendance %, avg study hours, etc.)
   ↓
3. Backend combines with academic data (term scores)
   ↓
4. Backend calls prediction API
   ↓
5. API returns prediction with probability
```

---

## 🧪 Testing

### Test the API
```bash
# Make sure API is running first
uvicorn app:app --reload

# In a new terminal, run tests
python test_api.py
```

### Test Prediction Function Directly
```bash
python src/models/predict_system.py
```

---

## 🎯 Model Performance

### Academic Model
- **Algorithm:** Logistic Regression
- **AUC Score:** ~0.97
- **Strength:** High accuracy on academic trajectory
- **Features:** 5 academic indicators

### Behavioral Model
- **Algorithm:** XGBoost
- **AUC Score:** ~0.85
- **Strength:** Early risk detection from habits
- **Features:** 6 behavioral indicators

---

## 🔧 Advanced Usage

### Retrain Models (Optional)

The repository includes pre-trained models. To retrain from scratch:

```bash
# Preprocess data
python dataset_alignment.py
python kaggle_dataset_alignment.py

# Train models
python src/models/train_academic_model.py
python src/models/train_behavioral_model.py
```

### Production Deployment

```bash
# Run with multiple workers
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🔐 Security & Configuration

### CORS Settings

By default, CORS is enabled for all origins. For production, modify `app.py`:

```python
allow_origins=["https://your-frontend-domain.com"]
```

### Environment Variables

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=info
```

---

## 📝 Logging

All requests are automatically logged with:
- Timestamp
- Input data
- Prediction results
- Errors (if any)

Check console output for logs.

---

## 🧠 Backend Integration Guide

### Your Responsibilities (Backend Developer)

1. **Store raw daily data** from teacher inputs
2. **Aggregate data** into model-ready features:
   ```python
   attendance_percentage = (days_present / total_days) * 100
   study_hours_per_day = sum(daily_study_hours) / num_days
   homework_completion = sum(homework_scores) / num_assignments
   class_participation = sum(participation_scores) / num_days
   ```
3. **Call prediction API** with aggregated features
4. **Display results** to teachers/administrators

---

## 📚 Data Sources

- **UCI Student Performance Dataset** - Academic performance data
- **Kaggle Student Performance Factors** - Behavioral and environmental factors

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT License

---

## 📞 Support

For questions or issues:
1. Check the interactive docs at `/docs`
2. Review the examples above
3. Open a GitHub issue

---

## 🎉 You're Ready!

The system is production-ready and can be integrated with any frontend or backend system. Start the API and visit http://127.0.0.1:8000/docs to explore!
