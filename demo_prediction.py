import pandas as pd
import json
import sys
import os
from src.models.predict_system import predict_student

def run_demo():
    print("="*60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM - SUPERVISOR DEMO")
    print("="*60)
    
    # Load sample data
    data_path = "data/processed/aligned_student_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    # Take a few interesting samples (some pass, some fail)
    samples = df.sample(5)
    
    # Mapping CSV columns to API expected keys
    # CSV: Term1_avg, Term2_avg, Seq5_score, Attendance_percentage, Study_hours_per_day, Homework_completion, Extra_lessons, Class_participation, Parental_support
    # API: term1_avg, term2_avg, seq5_score, attendance_percentage, parental_support, study_hours_per_day, homework_completion, class_participation, extra_lessons
    
    print(f"\nRunning predictions on {len(samples)} random samples from {data_path}...\n")
    
    for i, (_, row) in enumerate(samples.iterrows()):
        student_data = {
            "term1_avg": float(row["Term1_avg"]),
            "term2_avg": float(row["Term2_avg"]),
            "seq5_score": float(row["Seq5_score"]),
            "attendance_percentage": float(row["Attendance_percentage"]),
            "parental_support": int(row["Parental_support"]),
            "study_hours_per_day": float(row["Study_hours_per_day"]),
            "homework_completion": float(row["Homework_completion"]),
            "class_participation": float(row["Class_participation"]),
            "extra_lessons": int(row["Extra_lessons"])
        }
        
        try:
            result = predict_student(student_data)
            
            print(f"--- Student #{i+1} ---")
            print(f"Academic: T1: {student_data['term1_avg']}, T2: {student_data['term2_avg']}, S5: {student_data['seq5_score']}")
            print(f"Behavioral: Attendance: {student_data['attendance_percentage']}%, Study: {student_data['study_hours_per_day']}h/day")
            
            color = "\033[92m" if result["prediction"] == "Pass" else "\033[91m"
            reset = "\033[0m"
            
            print(f"Prediction: {color}{result['prediction']}{reset} ({result['probability']*100:.1f}% Pass Probability)")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Explanations:")
            for exp in result["explanations"]:
                print(f"  - {exp}")
            print("\n")
            
        except Exception as e:
            print(f"Error predicting student #{i+1}: {e}")

if __name__ == "__main__":
    run_demo()
