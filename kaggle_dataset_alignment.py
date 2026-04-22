# =========================================
# 1. IMPORTS
# =========================================
import pandas as pd

# =========================================
# 2. LOAD DATASET
# =========================================
df = pd.read_csv("data/raw/StudentPerformanceFactors.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Columns:", df.columns)

# =========================================
# 3. CREATE TARGET (PASS / FAIL)
# =========================================

# Use percentile-based threshold (balanced + realistic)
threshold = df["exam_score"].quantile(0.6)

df["pass"] = (df["exam_score"] >= threshold).astype(int)

print(f"\nThreshold used: {threshold}")
print("Class balance:")
print(df["pass"].value_counts(normalize=True))

# =========================================
# 4. INITIALIZE ALIGNED DATAFRAME
# =========================================
df_aligned = pd.DataFrame()

# =========================================
# 5. CORE FEATURES (USED NOW)
# =========================================

# Academic
df_aligned["Term1_avg"] = df["previous_scores"]
df_aligned["Term2_avg"] = df["previous_scores"]
df_aligned["Seq5_score"] = df["previous_scores"]

# Attendance
df_aligned["Attendance_percentage"] = df["attendance"]

# Study habits
df_aligned["Study_hours_per_day"] = df["hours_studied"] / 7  # weekly → daily
df_aligned["Homework_completion"] = df["hours_studied"] * 5  # proxy
df_aligned["Extra_lessons"] = df["tutoring_sessions"]

# Engagement
df_aligned["Class_participation"] = df["motivation_level"]

# Behavioral
df_aligned["Sleep_hours"] = df["sleep_hours"]

# =========================================
# 6. EXTENDED FEATURES (FUTURE USE)
# =========================================
df_aligned["Parental_support"] = df["parental_involvement"]
df_aligned["Teacher_quality"] = df["teacher_quality"]
df_aligned["Peer_influence"] = df["peer_influence"]
df_aligned["Access_to_resources"] = df["access_to_resources"]
df_aligned["Family_income"] = df["family_income"]
df_aligned["Internet_access"] = df["internet_access"]
df_aligned["School_type"] = df["school_type"]
df_aligned["Learning_disabilities"] = df["learning_disabilities"]
df_aligned["Physical_activity"] = df["physical_activity"]

# =========================================
# 7. ENCODING (ROBUST)
# =========================================

# Clean all string columns first
for col in df_aligned.select_dtypes(include="object").columns:
    df_aligned[col] = df_aligned[col].str.strip().str.lower()

# Ordinal mappings
level_map = {"low": 0, "medium": 1, "high": 2}

df_aligned["Class_participation"] = df_aligned["Class_participation"].map(level_map)
df_aligned["Parental_support"] = df_aligned["Parental_support"].map(level_map)
df_aligned["Teacher_quality"] = df_aligned["Teacher_quality"].map(level_map)
df_aligned["Access_to_resources"] = df_aligned["Access_to_resources"].map(level_map)
df_aligned["Family_income"] = df_aligned["Family_income"].map(level_map)

# Peer influence
peer_map = {"negative": 0, "neutral": 1, "positive": 2}
df_aligned["Peer_influence"] = df_aligned["Peer_influence"].map(peer_map)

# Binary mappings
binary_map = {"yes": 1, "no": 0}

df_aligned["Internet_access"] = df_aligned["Internet_access"].map(binary_map)
df_aligned["Learning_disabilities"] = df_aligned["Learning_disabilities"].map(binary_map)

# School type
df_aligned["School_type"] = df_aligned["School_type"].map({
    "public": 0,
    "private": 1
})

# =========================================
# 8. ADD TARGET
# =========================================
df_aligned["Pass"] = df["pass"]

# =========================================
# 9. CLEANING
# =========================================

# Fill missing values
df_aligned = df_aligned.fillna(0)

print("\nPreview:")
print(df_aligned.head())

print("\nInfo:")
print(df_aligned.info())

print("\nFinal class distribution:")
print(df_aligned["Pass"].value_counts())

print("\nMissing values after cleaning:")
print(df_aligned.isnull().sum())

# =========================================
# 10. SAVE
# =========================================
df_aligned.to_csv(
    "data/processed/aligned_kaggle_data_full.csv",
    index=False
)

print("\n✅ Saved as 'aligned_kaggle_data_full.csv'")

print(df_aligned["Pass"].value_counts())
