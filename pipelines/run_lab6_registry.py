import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 6: Model Registry & Lifecycle")
    print("[INFO] =========================================")

    modules = [
        "src.train_registry",
        "src.automate_lifecycle",
        "src.generate_registry_report",
    ]

    for module in modules:
        print(f"\n[INFO] ---> Executing {module}...")
        result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {module} failed with exit code {result.returncode}.")
            sys.exit(1)

    print("\n[SUCCESS] Lab 6: Model Registry & Lifecycle fully executed!")


if __name__ == "__main__":
    execute_pipeline()
