# =========================================
# 1. IMPORTS
# =========================================
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# =========================================
# 2. LOAD DATA
# =========================================
df = pd.read_csv("data/processed/aligned_kaggle_data_full.csv")
print("Dataset shape:", df.shape)

# =========================================
# 3. FEATURES & TARGET
# =========================================
TARGET = "Pass"

FEATURES = [
    "Study_hours_per_day",
    "Sleep_hours",
    "Class_participation",
    "Homework_completion",
    "Extra_lessons",
    "Attendance_percentage"
]

X = df[FEATURES]
y = df[TARGET]

# =========================================
# 4. TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 5. MODEL
# =========================================
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

# =========================================
# 6. TRAIN
# =========================================
model.fit(X_train, y_train)

# =========================================
# 7. EVALUATION
# =========================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# =========================================
# 8. SAVE MODEL
# =========================================
joblib.dump(model, "models/behavioral_model.pkl")

print("✅ Behavioral model saved as behavioral_model.pkl")
##testing for ovefitting
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

## visualizing most contributing features
import matplotlib.pyplot as plt

plt.barh(FEATURES, model.feature_importances_)
plt.title("Feature Importance")
plt.show()