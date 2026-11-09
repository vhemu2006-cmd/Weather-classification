"""Stage 07 (Lab 8) - Model quality gate on weighted F1."""
import json
import sys

from weather_common import load_config, project_path


def run_quality_gate():
    print("[INFO] Starting Model Quality Gate...")
    config = load_config()

    report_path = project_path(config["paths"]["evaluation_report"])
    metric_name = config["quality_gate"]["metric"]
    minimum_value = float(config["quality_gate"]["minimum_value"])

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Evaluation report not found: {report_path}")
        return False

    actual_value = float(report["metrics"].get(metric_name, 0.0))
    passed = actual_value >= minimum_value

    print(f"[INFO] Quality Gate: {metric_name}={actual_value:.4f} | required>={minimum_value:.4f}")

    gate_report_path = project_path(config["paths"]["quality_gate_report"])
    gate_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gate_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "stage": "Lab 8 - Model Quality Gate",
            "metric": metric_name,
            "minimum_value": minimum_value,
            "actual_value": actual_value,
            "status": "PASSED" if passed else "FAILED",
        }, f, indent=4)

    if passed:
        print("[SUCCESS] Model Quality Gate PASSED.")
    else:
        print("[ERROR] Model Quality Gate FAILED. Pipeline will stop before promotion.")
    return passed


if __name__ == "__main__":
    sys.exit(0 if run_quality_gate() else 1)
