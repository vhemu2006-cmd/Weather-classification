"""Stage 10 (Lab 9) - Failure injection and recovery tests.

Each scenario corrupts one input, runs the relevant stage, checks that the
stage HALTS (non-zero exit), then restores the original file. The last
scenario deletes the local model and recovers it from the MLflow Production
version. Run the normal pipeline first so every artifact exists.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from weather_common import PROJECT_ROOT, configure_mlflow, load_config


def run_script(script_path):
    return subprocess.run([sys.executable, str(PROJECT_ROOT / script_path)], cwd=PROJECT_ROOT).returncode


def restore_file(backup_path, original_path):
    if os.path.exists(original_path):
        os.remove(original_path)
    os.replace(backup_path, original_path)


def halt_result(scenario, exit_code):
    passed = exit_code != 0
    return {"scenario": scenario, "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED", "passed": passed}


def corrupt_raw_and_validate(config, scenario, corrupt):
    data_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_path = Path(str(data_path) + ".lab9_backup")
    shutil.copy2(data_path, backup_path)
    try:
        df = corrupt(pd.read_csv(data_path))
        df.to_csv(data_path, index=False)
        return halt_result(scenario, run_script("src/mlops_pipeline/02_validate_data.py"))
    finally:
        restore_file(backup_path, data_path)


def test_missing_data(config):
    print("\n[LAB 9] Scenario 1: Missing data")
    data_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_path = Path(str(data_path) + ".lab9_backup")
    os.replace(data_path, backup_path)
    try:
        return halt_result("missing_data", run_script("src/mlops_pipeline/02_validate_data.py"))
    finally:
        restore_file(backup_path, data_path)


def test_invalid_datatype(config):
    print("\n[LAB 9] Scenario 2: Invalid datatype")

    def corrupt(df):
        df["Temperature (C)"] = df["Temperature (C)"].astype(object)
        df.loc[0, "Temperature (C)"] = "INVALID_NUMBER"
        return df

    return corrupt_raw_and_validate(config, "invalid_datatype", corrupt)


def test_out_of_range_values(config):
    print("\n[LAB 9] Scenario 3: Out-of-range values")

    def corrupt(df):
        df.loc[0, "Humidity"] = 1.75           # humidity is a fraction 0..1
        df.loc[1, "Wind Bearing (degrees)"] = 450.0
        return df

    return corrupt_raw_and_validate(config, "out_of_range_values", corrupt)


def test_schema_mismatch(config):
    print("\n[LAB 9] Scenario 4: Schema mismatch")
    return corrupt_raw_and_validate(config, "schema_mismatch", lambda df: df.drop(columns=["Humidity"]))


def test_preprocessing_inconsistency(config):
    print("\n[LAB 9] Scenario 5: Preprocessing inconsistency")
    path = PROJECT_ROOT / config["data"]["processed_dir"] / "X_train_final.npy"
    if not path.exists():
        print("[ERROR] Processed training output is missing. Run the normal pipeline first.")
        return {"scenario": "preprocessing_inconsistency", "expected": "HALT",
                "actual": "NOT_TESTED", "passed": False}

    backup_path = Path(str(path) + ".lab9_backup")
    shutil.copy2(path, backup_path)
    try:
        X_train = np.load(path)
        X_train[0, 0] = np.nan
        np.save(path, X_train)
        return halt_result("preprocessing_inconsistency",
                           run_script("src/mlops_pipeline/04_validate_outputs.py"))
    finally:
        restore_file(backup_path, path)


def test_missing_model_with_recovery(config):
    print("\n[LAB 9] Scenario 6: Missing model artifact + recovery")
    model_path = PROJECT_ROOT / config["paths"]["model"]
    model_name = config["registry"]["model_name"]

    if not model_path.exists():
        print("[ERROR] Local model is missing before the test. Run the normal pipeline first.")
        return {"scenario": "missing_model_artifact", "expected": "RECOVER",
                "actual": "NOT_TESTED", "passed": False}

    backup_path = Path(str(model_path) + ".lab9_backup")
    shutil.copy2(model_path, backup_path)
    os.remove(model_path)

    try:
        failure_detected = run_script("src/mlops_pipeline/06_model_evaluation.py") != 0

        configure_mlflow(config)
        production_uri = f"models:/{model_name}/Production"
        print(f"[INFO] Recovering model from {production_uri}...")
        try:
            production_model = mlflow.sklearn.load_model(production_uri)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(production_model, model_path, compress=3)
            recovered = run_script("src/mlops_pipeline/06_model_evaluation.py") == 0
        except Exception as exc:
            print(f"[ERROR] Recovery from MLflow failed: {exc}")
            recovered = False

        return {
            "scenario": "missing_model_artifact",
            "expected": "RECOVER",
            "actual": "RECOVERED" if failure_detected and recovered else "RECOVERY_FAILED",
            "failure_detected": failure_detected,
            "recovery_succeeded": recovered,
            "passed": failure_detected and recovered,
        }
    finally:
        restore_file(backup_path, model_path)


def run_failure_tests():
    print("[INFO] ==================================================")
    print("[INFO] Failure Tests: Pipeline Failure Handling and Recovery")
    print("[INFO] ==================================================")

    config = load_config()
    repetitions = int(config["lab9"]["repetitions"])
    tests = [
        test_missing_data,
        test_invalid_datatype,
        test_out_of_range_values,
        test_schema_mismatch,
        test_preprocessing_inconsistency,
        test_missing_model_with_recovery,
    ]

    all_runs = []
    for repetition in range(1, repetitions + 1):
        print(f"\n{'=' * 70}\n[INFO] Lab 9 Repetition {repetition}/{repetitions}\n{'=' * 70}")
        results = [test(config) for test in tests]
        all_runs.append({"repetition": repetition, "results": results,
                         "passed": all(r["passed"] for r in results)})

    report = {
        "stage": "Failure Handling and Recovery",
        "repetitions": repetitions,
        "runs": all_runs,
        "overall_status": "PASSED" if all(run["passed"] for run in all_runs) else "FAILED",
    }

    report_path = PROJECT_ROOT / config["paths"]["failure_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if report["overall_status"] == "PASSED":
        print("\n[SUCCESS] All Lab 9 failure and recovery tests PASSED.")
    else:
        print("\n[ERROR] One or more Lab 9 scenarios FAILED.")
    print(f"[INFO] Report saved to {report_path}")
    return report["overall_status"] == "PASSED"


if __name__ == "__main__":
    sys.exit(0 if run_failure_tests() else 1)
