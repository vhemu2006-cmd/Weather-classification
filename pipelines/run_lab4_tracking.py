import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 4: MLflow Experiment Tracking")
    print("[INFO] =========================================")

    modules = [
        "src.preprocess_lab4",
        "src.train_mlflow",
        "src.evaluate_lab4",
        "src.validate_reproducibility",
    ]

    for module in modules:
        print(f"\n[INFO] ---> Executing {module}...")
        result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {module} failed with exit code {result.returncode}.")
            sys.exit(1)

    print("\n[SUCCESS] Lab 4: MLflow Experiment Tracking fully executed!")


if __name__ == "__main__":
    execute_pipeline()
