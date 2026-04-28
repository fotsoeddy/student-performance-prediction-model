#!/usr/bin/env python3
"""
UCI Dataset Alignment Script
==============================
Converts the UCI 'student-mat.csv' dataset into the project's standard
feature schema for training the academic model.

SCALING NOTE:
  - UCI G1, G2, G3 scores are already on the 0-20 scale — no conversion
    needed.

PROXY FEATURE WARNINGS:
  - Seq5_score is set to G2 (second period grade) as a proxy for the
    sequence-5 score.  Real Seq5 data would come from school records.
  - Homework_completion is estimated as `studytime * 20`, which is a rough
    proxy.  Real homework data would come from teacher records.
  - Class_participation is mapped from `activities` (yes=1 / no=0), which
    is a binary proxy for a continuous participation metric.
  - Attendance_percentage is derived from `absences` by inverting the
    ratio against the max observed absences in the dataset.
"""

import pandas as pd

# =========================================
# 1. LOAD DATASET
# =========================================
# Use sep=";" for UCI dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=";")

print("Original shape:", df.shape)
print(df.head())

# =========================================
# 2. CREATE TARGET (PASS / FAIL)
# =========================================
def pass_label(score):
    """Pass if final grade (G3) >= 10 on 0-20 scale."""
    return 1 if score >= 10 else 0

df["Pass"] = df["G3"].apply(pass_label)

# =========================================
# 3. CREATE ALIGNED DATAFRAME
# =========================================
df_aligned = pd.DataFrame()

# ---------- Core Academic (already 0-20 scale) ----------
df_aligned["Term1_avg"] = df["G1"]
df_aligned["Term2_avg"] = df["G2"]
df_aligned["Seq5_score"] = df["G2"]   # PROXY: using G2 as best available estimate

# ---------- Attendance ----------
max_absences = df["absences"].max()
if max_absences > 0:
    df_aligned["Attendance_percentage"] = 100 - (df["absences"] / max_absences * 100)
else:
    df_aligned["Attendance_percentage"] = 100.0

# ---------- Study Habits ----------
study_map = {1: 1, 2: 2, 3: 3, 4: 4}
df_aligned["Study_hours_per_day"] = df["studytime"].map(study_map)

# PROXY: Homework_completion estimated from study time
df_aligned["Homework_completion"] = (df["studytime"] * 20).clip(upper=100)

# Extra lessons
df_aligned["Extra_lessons"] = df["schoolsup"]

# ---------- Engagement ----------
# PROXY: Class_participation mapped from extracurricular activities (binary)
df_aligned["Class_participation"] = df["activities"]

# ---------- Support ----------
df_aligned["Parental_support"] = df["famsup"]

# =========================================
# 4. ENCODE CATEGORICAL FEATURES
# =========================================
binary_map = {"yes": 1, "no": 0}

df_aligned["Extra_lessons"] = df_aligned["Extra_lessons"].map(binary_map)
df_aligned["Class_participation"] = df_aligned["Class_participation"].map(binary_map)
df_aligned["Parental_support"] = df_aligned["Parental_support"].map(binary_map)

# =========================================
# 5. ADD TARGET COLUMN
# =========================================
df_aligned["Pass"] = df["Pass"]

# =========================================
# 6. IMPROVED MISSING VALUE HANDLING
# =========================================
numeric_cols = df_aligned.select_dtypes(include=["number"]).columns
for col in numeric_cols:
    if df_aligned[col].isnull().any():
        median_val = df_aligned[col].median()
        n_missing = df_aligned[col].isnull().sum()
        print(f"  Imputing {n_missing} missing in '{col}' with median={median_val:.2f}")
        df_aligned[col] = df_aligned[col].fillna(median_val)

# =========================================
# 7. FINAL CHECKS
# =========================================
print("\nAligned dataset preview:")
print(df_aligned.head())

print("\nMissing values:")
print(df_aligned.isnull().sum())

print("\nFinal shape:", df_aligned.shape)

# Verify score ranges
print(f"\nScore ranges (should be 0-20):")
for col in ["Term1_avg", "Term2_avg", "Seq5_score"]:
    print(f"  {col}: {df_aligned[col].min():.1f} - {df_aligned[col].max():.1f}")

# =========================================
# 8. SAVE CLEAN DATASET
# =========================================
df_aligned.to_csv("data/processed/aligned_student_data.csv", index=False)

print("\n✅ Dataset aligned and saved as 'aligned_student_data.csv'")

# Count of each class
print(df_aligned["Pass"].value_counts())
print(df_aligned["Pass"].value_counts(normalize=True) * 100)