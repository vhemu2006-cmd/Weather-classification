"""Labs 7-13 - run the full weather MLOps pipeline stage by stage.

    python src/mlops_pipeline/weather_prediction_pipeline.py          # normal run
    python src/mlops_pipeline/weather_prediction_pipeline.py --force  # retrain even without a new batch
    python src/mlops_pipeline/weather_prediction_pipeline.py --force --skip-deploy  # CI without Kubernetes

Without --force the pipeline stops after Data Loading when no new batch was
ingested (status no_new_data / already_ingested), so the same data is never
retrained twice.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    ("Data Loading", "src/mlops_pipeline/01_load_data.py"),
    ("Raw Data Validation", "src/mlops_pipeline/02_validate_data.py"),
    ("Preprocessing", "src/mlops_pipeline/03_preprocessing.py"),
    ("Processed Output Validation", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Training + MLflow + Model Registry", "src/mlops_pipeline/05_train_registry.py"),
    ("Model Evaluation", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Model Quality Gate", "src/mlops_pipeline/07_quality_gate.py"),
    ("Automated Model Lifecycle", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Deployment Update", "src/mlops_pipeline/deploy_if_promoted.py"),
    ("Production Inference", "src/mlops_pipeline/09_inference.py"),
]


def load_data_load_report():
    report_path = PROJECT_ROOT / "artifacts" / "data_load_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Data load report not found: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(force=False, skip_deploy=False):
    print("[INFO] ==================================================")
    print("[INFO] Starting Weather Daily Summary MLOps Pipeline")
    print("[INFO] ==================================================")

    stages = [s for s in STAGES if not (skip_deploy and s[0] == "Deployment Update")]
    run_log = {"pipeline": "Weather Daily Summary MLOps", "stages": [], "force": force,
               "skip_deploy": skip_deploy}

    for stage_index, (stage_name, script_path) in enumerate(stages):
        print("\n" + "=" * 70)
        print(f"RUNNING: {stage_name}")
        print("=" * 70)

        result = subprocess.run([sys.executable, str(PROJECT_ROOT / script_path)], cwd=PROJECT_ROOT)
        status = "PASSED" if result.returncode == 0 else "FAILED"
        run_log["stages"].append({"stage": stage_name, "status": status, "exit_code": result.returncode})
        print(f"--> {stage_name}: {status}")

        if result.returncode != 0:
            print("[HALT] Pipeline stopped because this stage failed.")
            run_log["overall_status"] = "FAILED"
            break

        if stage_index == 0 and not force:
            data_status = load_data_load_report().get("status")
            if data_status in {"already_ingested", "no_new_data"}:
                print(f"[INFO] Data load status: {data_status}")
                print("[INFO] No new dataset was ingested. Retraining and downstream stages will not run.")
                print("[INFO] Use --force to retrain on the current dataset anyway.")
                run_log["overall_status"] = "NO_NEW_DATA"
                break
            if data_status != "ingested":
                print(f"[ERROR] Unexpected data load status: {data_status}")
                run_log["overall_status"] = "FAILED"
                break
            print("[INFO] New dataset was successfully ingested. Continuing with the ML pipeline.")

    if len(run_log["stages"]) == len(stages) and all(s["status"] == "PASSED" for s in run_log["stages"]):
        run_log["overall_status"] = "PASSED"

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "weather_prediction_pipeline_run_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    print("\n" + "=" * 70)
    status = run_log["overall_status"]
    if status == "PASSED":
        print("[SUCCESS] Weather Daily Summary MLOps Pipeline completed successfully.")
    elif status == "NO_NEW_DATA":
        print("[INFO] No new dataset available. Pipeline finished without retraining.")
    else:
        print("[ERROR] Weather Daily Summary MLOps Pipeline failed.")
    print(f"[INFO] Run log: {log_path}")
    print("=" * 70)

    return status in {"PASSED", "NO_NEW_DATA"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Weather Daily Summary MLOps pipeline.")
    parser.add_argument("--force", action="store_true",
                        help="Run all ML stages even when no new dataset was ingested.")
    parser.add_argument("--skip-deploy", action="store_true",
                        help="Leave out the Docker/Kubernetes deployment stage (e.g. on GitHub-hosted CI).")
    args = parser.parse_args()
    sys.exit(0 if run_pipeline(force=args.force, skip_deploy=args.skip_deploy) else 1)
