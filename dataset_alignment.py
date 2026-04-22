# =========================================
# 1. IMPORTS
# =========================================
import pandas as pd

# =========================================
# 2. LOAD DATASET
# =========================================
# Use sep=";" for UCI dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

print("Original shape:", df.shape)
print(df.head())

# =========================================
# 3. CREATE TARGET (PASS / FAIL)
# =========================================
def pass_label(score):
    return 1 if score >= 10 else 0

df["Pass"] = df["G3"].apply(pass_label)

# =========================================
# 4. CREATE ALIGNED DATAFRAME (INPUT FEATURES ONLY)
# =========================================
df_aligned = pd.DataFrame()

# ---------- Core Academic ----------
df_aligned["Term1_avg"] = df["G1"]
df_aligned["Term2_avg"] = df["G2"]
df_aligned["Seq5_score"] = df["G2"]   # proxy for sequence before final

# ---------- Attendance ----------
max_absences = df["absences"].max()
df_aligned["Attendance_percentage"] = 100 - (df["absences"] / max_absences * 100)

# Simulated Late_count (optional)
df_aligned["Late_count"] = (df["absences"] * 0.3).astype(int)

# ---------- Study Habits ----------
study_map = {1: 1, 2: 2, 3: 3, 4: 4}
df_aligned["Study_hours_per_day"] = df["studytime"].map(study_map)

# Proxy for homework completion
df_aligned["Homework_completion"] = df["studytime"] * 20

# Extra lessons
df_aligned["Extra_lessons"] = df["schoolsup"]

# ---------- Engagement ----------
df_aligned["Class_participation"] = df["activities"]

# ---------- Support ----------
df_aligned["Parental_support"] = df["famsup"]

# =========================================
# 5. ENCODE CATEGORICAL FEATURES
# =========================================
binary_map = {"yes": 1, "no": 0}

df_aligned["Extra_lessons"] = df_aligned["Extra_lessons"].map(binary_map)
df_aligned["Class_participation"] = df_aligned["Class_participation"].map(binary_map)
df_aligned["Parental_support"] = df_aligned["Parental_support"].map(binary_map)

# =========================================
# 6. ADD TARGET COLUMN
# =========================================
df_aligned["Pass"] = df["Pass"]

# =========================================
# 7. FINAL CHECKS
# =========================================
print("\nAligned dataset preview:")
print(df_aligned.head())

print("\nMissing values:")
print(df_aligned.isnull().sum())

print("\nFinal shape:", df_aligned.shape)

# =========================================
# 8. SAVE CLEAN DATASET
# =========================================
df_aligned.to_csv("data/processed/aligned_student_data.csv", index=False)

print("\n✅ Dataset aligned and saved as 'aligned_student_data.csv'")

# Count of each class
print(df_aligned["Pass"].value_counts())

# Percentage distribution
print(df_aligned["Pass"].value_counts(normalize=True) * 100)