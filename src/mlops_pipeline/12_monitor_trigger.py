"""Lab 13 - monitoring + retraining trigger.

1. Refresh drift reports (data, prediction, concept) and combine them.
2. If drift is detected AND data/raw/new_weather.csv exists, run the full
   pipeline (which ingests the batch, retrains, gates, promotes and deploys).
3. Write outputs/monitoring/monitor_trigger_report.json with the decision.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from weather_common import PROJECT_ROOT, load_config, project_path

DRIFT_SCRIPTS = [
    "src/mlops_pipeline/12_data_drift.py",
    "src/mlops_pipeline/12_prediction_drift.py",
    "src/mlops_pipeline/12_concept_drift.py",
    "src/mlops_pipeline/12_drift_analysis.py",
]
DRIFT_REPORT = project_path("outputs/drift/drift_analysis_report.csv")
PIPELINE_SCRIPT = "src/mlops_pipeline/weather_prediction_pipeline.py"
TRIGGER_REPORT = project_path("outputs/monitoring/monitor_trigger_report.json")
DATA_LOAD_REPORT = project_path("artifacts/data_load_report.json")


def run(script):
    return subprocess.run([sys.executable, str(PROJECT_ROOT / script)], cwd=PROJECT_ROOT).returncode


def run_drift_analysis():
    print("[INFO] Running drift analysis...")
    for script in DRIFT_SCRIPTS:
        print(f"[INFO] -> {script}")
        if run(script) != 0:
            print(f"[ERROR] {script} failed.")
            return False
    return True


def detect_drift():
    if not DRIFT_REPORT.exists():
        raise FileNotFoundError(f"Drift report not found: {DRIFT_REPORT}")
    return pd.read_csv(DRIFT_REPORT)["status"].astype(str).str.upper().eq("DRIFT").any()


def get_data_load_status():
    if not DATA_LOAD_REPORT.exists():
        return None
    with DATA_LOAD_REPORT.open("r", encoding="utf-8") as f:
        return json.load(f).get("status")


def save_trigger_report(report):
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    TRIGGER_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with TRIGGER_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)


def run_monitor_trigger():
    print("=" * 70)
    print("LAB 13 MONITORING + RETRAINING TRIGGER")
    print("=" * 70)

    new_data_path = project_path(load_config()["data"]["new_data_path"])
    base = {"pipeline_called": False, "new_data_ingested": False, "retraining_executed": False}

    if not run_drift_analysis():
        save_trigger_report({"status": "drift_analysis_failed", "drift_detected": False,
                             "new_data_present": new_data_path.exists(), **base})
        print("[ERROR] Drift analysis failed.")
        return False

    drift_detected = bool(detect_drift())
    new_data_present = new_data_path.exists()
    print(f"[INFO] Drift detected: {drift_detected}")
    print(f"[INFO] New dataset present: {new_data_present}")

    if not drift_detected:
        save_trigger_report({"status": "no_drift", "drift_detected": False,
                             "new_data_present": new_data_present, **base})
        print("[INFO] No drift detected. Waiting for future monitoring cycles.")
        return True

    if not new_data_present:
        save_trigger_report({"status": "drift_waiting_for_data", "drift_detected": True,
                             "new_data_present": False, **base})
        print(f"[INFO] Drift detected, but no new dataset is available. Waiting for {new_data_path.name}.")
        return True

    print("[INFO] Drift detected and new dataset is available. Triggering the MLOps pipeline...")
    exit_code = run(PIPELINE_SCRIPT)

    data_load_status = get_data_load_status()
    ingested = data_load_status == "ingested"
    retrained = ingested and exit_code == 0

    save_trigger_report({
        "status": "retraining_completed" if retrained else "no_new_data_retraining_skipped",
        "drift_detected": True,
        "new_data_present": True,
        "pipeline_called": True,
        "pipeline_exit_code": exit_code,
        "pipeline_status": "passed" if exit_code == 0 else "failed",
        "data_load_status": data_load_status,
        "new_data_ingested": ingested,
        "retraining_executed": retrained,
    })

    if exit_code != 0:
        print("[ERROR] MLOps pipeline failed.")
        return False
    if ingested:
        print("[SUCCESS] New data was ingested and retraining completed.")
    else:
        print("[INFO] Pipeline completed, but no new data was ingested. Retraining was skipped.")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_monitor_trigger() else 1)
