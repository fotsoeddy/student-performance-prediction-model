# Model Defence Summary — Student Performance Prediction System

> **Purpose:** This document summarises the model component in simple terms for oral project defence.

---

## What the Model Does

The Student Performance Prediction Model predicts whether a secondary school student is likely to **pass or fail** their end-of-year examination. Beyond a simple pass/fail label, it returns:

- A **calibrated probability** of passing (0.0–1.0)
- A **risk level**: Low Risk, Medium Risk, or High Risk
- **Rule-based explanations** describing why the prediction was made (e.g., "low attendance", "weak academic foundation")

The model is designed as an **early warning tool** — it uses mid-year data (Terms 1 & 2, Sequence 5 score) to flag students who may fail before the final exam, giving teachers time to intervene.

---

## What Data It Uses

The model accepts **9 input features** per student:

| Feature | Description | Scale |
|---|---|---|
| `term1_avg` | Term 1 average score | 0–20 |
| `term2_avg` | Term 2 average score | 0–20 |
| `seq5_score` | Sequence 5 exam score | 0–20 |
| `attendance_percentage` | Attendance rate | 0–100% |
| `parental_support` | Has parental support? | 0 or 1 |
| `study_hours_per_day` | Daily study hours | 0–24 |
| `homework_completion` | Homework completion rate | 0–100% |
| `class_participation` | Participation level | 0–5 |
| `extra_lessons` | Number of extra lessons | ≥ 0 |

**Training data:**
- **UCI Student Performance Dataset** (395 records) — real Portuguese student grades on 0–20 scale
- **Kaggle Student Performance Factors** (6,607 records) — synthetic behavioural data

---

## How It Predicts

The system uses a **dual-model weighted ensemble**:

1. **Academic Model** (Logistic Regression, 70% weight) — focuses on term scores, attendance, and parental support.
2. **Behavioural Model** (XGBoost, 30% weight) — focuses on engagement score (study + homework), attendance, extra lessons, and participation.

**Final probability = 0.7 × Academic probability + 0.3 × Behavioural probability**

- If final probability ≥ 0.5 → **Pass**
- If final probability < 0.5 → **Fail**

Both models are **Platt-calibrated** (sigmoid method, 5-fold CV) so that probabilities are well-calibrated — a 70% probability truly means approximately 70% chance of passing.

---

## Why Two Models Are Used

| Reason | Explanation |
|---|---|
| **Different data perspectives** | The academic model captures grade-based patterns; the behavioural model captures engagement and effort patterns |
| **Complementary signals** | A student may have decent grades but declining engagement (behavioural model catches this) |
| **Robustness** | Combining two models reduces the impact of individual model errors |
| **Weighted by importance** | Academic performance (70%) is weighted more heavily because past grades are the strongest predictor of future grades |

---

## What the API Returns

For each prediction, the API returns:

```json
{
    "prediction": "Pass",
    "probability": 0.723,
    "risk_level": "Low Risk",
    "academic_prob": 0.801,
    "behavioral_prob": 0.542,
    "confidence": "Moderate confidence prediction",
    "explanations": ["Consistent performance across all metrics"]
}
```

**Available endpoints:**
- `POST /predict` — single student prediction
- `POST /predict/batch` — batch prediction for an entire class
- `GET /health` — health check
- `GET /info` — API metadata and thresholds
- `GET /docs` — interactive Swagger UI documentation

---

## Why the Model Is Useful

1. **Early detection** — identifies at-risk students before the final exam.
2. **Actionable** — provides specific reasons (explanations) for each prediction.
3. **Scalable** — batch prediction enables class-level and school-level analysis.
4. **Transparent** — risk levels and explanations make results understandable to non-technical users.
5. **Integrable** — REST API design allows future connection to teacher and parent dashboards.

---

## Current Evaluation Scores

| Metric | Value |
|---|---|
| Overall Accuracy | **88.35%** |
| Precision | 95.06% |
| Recall (Pass class) | 87.17% |
| **Recall (Fail/At-Risk class)** | **90.77%** 🎯 |
| F1-Score | 90.94% |
| ROC-AUC | **0.9554** |

> **Key takeaway:** The model correctly identifies **91% of students who are truly at risk of failure**. This high recall for the Fail class is critical because missing an at-risk student is far more costly than a false alarm.

---

## Current Limitations

1. Trained on **public datasets** (Portuguese + synthetic), not real Cameroonian school data.
2. Some features are **proxy-based** (e.g., `Seq5_score` duplicates `Term2_avg`; `homework_completion` derived from study hours).
3. The two training datasets use **different pass thresholds** (absolute vs. percentile).
4. Predictions are **probabilistic estimates**, not certainties.
5. No **SHAP-based explainability** yet — explanations are rule-based.
6. No **prediction history storage** — each prediction is independent.
7. Fixed 0.5 threshold — not optimised for minimising false negatives.

---

## Future Improvements

1. Collect and train on **real Cameroonian school data**.
2. Add **SHAP explainability** for model-intrinsic feature importance.
3. Add **subject-specific prediction** models.
4. Add **term-by-term progression tracking**.
5. Build **teacher dashboard** and **parent dashboard** as separate applications.
6. Add **recommendation engine** for personalised revision plans.
7. Implement **continuous retraining** with new cohort data.
8. Add **fairness audits** for demographic bias.
9. Implement **authentication and security** for production deployment.
