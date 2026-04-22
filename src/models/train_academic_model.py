# =========================================
# 1. IMPORTS
# =========================================
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

# =========================================
# 2. LOAD DATA
# =========================================
df = pd.read_csv("data/processed/aligned_student_data.csv")
print(df.columns)
print("Dataset shape:", df.shape)

# =========================================
# 3. FEATURES & TARGET
# =========================================
TARGET = "Pass"

FEATURES = [
    "Term1_avg",
    "Term2_avg",
    "Seq5_score",
    "Attendance_percentage",
    "Parental_support"
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
model = LogisticRegression(max_iter=1000)

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

# Overfitting check
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

# =========================================
# 8. SAVE MODEL
# =========================================
joblib.dump(model, "models/academic_model.pkl")

print("✅ Academic model saved as academic_model.pkl")