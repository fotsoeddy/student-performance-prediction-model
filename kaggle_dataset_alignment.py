#!/usr/bin/env python3
"""
Kaggle Dataset Alignment Script
================================
Converts the Kaggle 'StudentPerformanceFactors.csv' dataset into the project's
standard feature schema for training the behavioral model.

IMPORTANT SCALING DECISIONS:
  - This project uses the 0-20 grading scale (standard for primary school
    in Cameroon). Kaggle's `previous_scores` column is on a 0-100 scale,
    so we divide by 5 to normalise to 0-20.
  - `Homework_completion` is a PROXY derived from `hours_studied * 5`,
    clamped to [0, 100].

PROXY FEATURE WARNINGS:
  - Term1_avg, Term2_avg, Seq5_score are ALL set to the same normalised
    `previous_scores` value because the Kaggle dataset has only one score
    column.  This means the academic model cannot distinguish between
    terms when trained on this data alone.
  - Homework_completion is estimated from study hours, NOT directly measured.
  - Class_participation is mapped from `motivation_level` (Low/Medium/High),
    which is an imperfect proxy.
"""

import pandas as pd
import numpy as np

# =========================================
# 1. LOAD DATASET
# =========================================
df = pd.read_csv("data/raw/StudentPerformanceFactors.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Columns:", list(df.columns))
print(f"Raw dataset shape: {df.shape}")

# =========================================
# 2. CREATE TARGET (PASS / FAIL)
# =========================================

# Use percentile-based threshold (balanced + realistic)
threshold = df["exam_score"].quantile(0.6)

df["pass"] = (df["exam_score"] >= threshold).astype(int)

print(f"\nThreshold used: {threshold}")
print("Class balance:")
print(df["pass"].value_counts(normalize=True))

# =========================================
# 3. INITIALIZE ALIGNED DATAFRAME
# =========================================
df_aligned = pd.DataFrame()

# =========================================
# 4. CORE FEATURES — with proper scaling
# =========================================

# Academic — NORMALISE from 0-100 to 0-20 (primary school scale)
# PROXY: All three are the same underlying value since Kaggle has only
#         one score column (`previous_scores`).
df_aligned["Term1_avg"] = df["previous_scores"] / 5.0
df_aligned["Term2_avg"] = df["previous_scores"] / 5.0
df_aligned["Seq5_score"] = df["previous_scores"] / 5.0  # PROXY: same as Term averages

# Attendance — already 0-100 percentage
df_aligned["Attendance_percentage"] = df["attendance"]

# Study habits
df_aligned["Study_hours_per_day"] = df["hours_studied"] / 7  # weekly → daily

# PROXY: Homework_completion estimated from study hours, clamped to [0, 100]
df_aligned["Homework_completion"] = np.clip(df["hours_studied"] * 5, 0, 100)

df_aligned["Extra_lessons"] = df["tutoring_sessions"]

# PROXY: Class_participation mapped from motivation_level (Low/Medium/High)
df_aligned["Class_participation"] = df["motivation_level"]

# Behavioral
df_aligned["Sleep_hours"] = df["sleep_hours"]

# =========================================
# 5. EXTENDED FEATURES (FUTURE USE)
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
# 6. ENCODING (ROBUST)
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
# 7. ADD TARGET
# =========================================
df_aligned["Pass"] = df["pass"]

# =========================================
# 8. IMPROVED MISSING VALUE HANDLING
# =========================================
# Use median for numeric columns instead of naive fillna(0)
numeric_cols = df_aligned.select_dtypes(include=["number"]).columns
for col in numeric_cols:
    if df_aligned[col].isnull().any():
        median_val = df_aligned[col].median()
        n_missing = df_aligned[col].isnull().sum()
        print(f"  Imputing {n_missing} missing values in '{col}' with median={median_val:.2f}")
        df_aligned[col] = df_aligned[col].fillna(median_val)

# Use mode for any remaining categorical columns
cat_cols = df_aligned.select_dtypes(exclude=["number"]).columns
for col in cat_cols:
    if df_aligned[col].isnull().any():
        mode_val = df_aligned[col].mode()[0] if not df_aligned[col].mode().empty else "unknown"
        n_missing = df_aligned[col].isnull().sum()
        print(f"  Imputing {n_missing} missing values in '{col}' with mode='{mode_val}'")
        df_aligned[col] = df_aligned[col].fillna(mode_val)

print("\nPreview:")
print(df_aligned.head())

print("\nInfo:")
print(df_aligned.info())

print("\nFinal class distribution:")
print(df_aligned["Pass"].value_counts())

print("\nMissing values after cleaning:")
print(df_aligned.isnull().sum())

# Verify score ranges
print(f"\nScore ranges (should be 0-20):")
for col in ["Term1_avg", "Term2_avg", "Seq5_score"]:
    print(f"  {col}: {df_aligned[col].min():.1f} - {df_aligned[col].max():.1f}")
print(f"Homework_completion range (should be 0-100):")
print(f"  {df_aligned['Homework_completion'].min():.1f} - {df_aligned['Homework_completion'].max():.1f}")

# =========================================
# 9. SAVE
# =========================================
df_aligned.to_csv(
    "data/processed/aligned_kaggle_data_full.csv",
    index=False
)

print("\n✅ Saved as 'aligned_kaggle_data_full.csv'")
