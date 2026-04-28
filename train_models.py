#!/usr/bin/env python3
"""
Training Pipeline for Student Success Prediction Models
Trains calibrated academic and behavioral models.
"""

import subprocess
import sys
import os

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {description} failed: {str(e)}")
        return False

def main():
    """Run complete training pipeline."""
    print("=" * 70)
    print("STUDENT SUCCESS PREDICTION - MODEL TRAINING")
    print("=" * 70)
    print("\nThis will train:")
    print("  1. Academic Model (Logistic Regression with calibration)")
    print("  2. Behavioral Model (XGBoost with calibration)")
    print("\nEstimated time: 2-3 minutes")
    print("=" * 70)
    
    interactive = "--non-interactive" not in sys.argv
    if interactive:
        input("\nPress Enter to start training...")
    else:
        print("\nStarting training in non-interactive mode...")
    
    # Training scripts in order
    scripts = [
        ("src/models/train_academic_model.py", "Academic Model Training"),
        ("src/models/train_behavioral_model.py", "Behavioral Model Training"),
    ]
    
    results = []
    
    for script_path, description in scripts:
        if not os.path.exists(script_path):
            print(f"\n❌ Script not found: {script_path}")
            results.append(False)
            continue
        
        success = run_script(script_path, description)
        results.append(success)
        
        if not success:
            print(f"\n⚠️  Training stopped due to error in {description}")
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE SUMMARY")
    print("=" * 70)
    
    for (script_path, description), success in zip(scripts, results):
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {description}")
    
    if all(results):
        print("\n" + "=" * 70)
        print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 70)
        print("\nGenerated models:")
        print("  ✓ models/academic_model.pkl (calibrated)")
        print("  ✓ models/behavioral_model.pkl (calibrated)")
        print("\nPrediction uses weighted ensemble:")
        print("  • 70% Academic model")
        print("  • 30% Behavioral model")
        print("\nYou can now:")
        print("  1. Test predictions: python src/models/predict_system.py")
        print("  2. Run API: uvicorn app:app --reload")
        print("  3. Test API: python test_api.py")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠️  TRAINING INCOMPLETE")
        print("=" * 70)
        print("\nPlease check the error messages above and:")
        print("  1. Ensure all data files are present in data/processed/")
        print("  2. Ensure all dependencies are installed")
        print("  3. Check for any error messages")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
