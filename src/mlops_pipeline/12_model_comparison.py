"""Lab 12 - compare Production V1 (deployment bundle) with retrained V2 on the same fixed test set."""
import json

import joblib
import pandas as pd

from monitoring_common import RETRAIN_DIR, evaluate_with, load_production_loader, load_test_set
from weather_common import load_config, project_path

V2_MODEL_PATH = project_path("models/retrained_model_v2.pkl")
V2_PREPROCESSOR_PATH = project_path("models/retrained_preprocessor_v2.pkl")
V2_LABEL_ENCODER_PATH = project_path("models/retrained_label_encoder_v2.pkl")
REPORT_PATH = RETRAIN_DIR / "model_comparison_report.csv"
JSON_REPORT_PATH = RETRAIN_DIR / "model_comparison_report.json"


def main():
    print("=" * 70)
    print("LAB 12 - MODEL V1 VS V2 COMPARISON")
    print("=" * 70)

    config = load_config()
    gate_metric = config["quality_gate"]["metric"]
    gate_min = float(config["quality_gate"]["minimum_value"])

    X_test, y_test = load_test_set()
    print(f"\nEvaluation rows : {len(X_test)}")

    loader = load_production_loader()
    print(f"\nV1 (Production) : {loader.model_name} v{loader.model_version} | run {loader.run_id}")

    for path in (V2_MODEL_PATH, V2_PREPROCESSOR_PATH, V2_LABEL_ENCODER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"V2 artifact not found: {path}. Run 12_retrain_model.py first.")
    v2_bundle = tuple(joblib.load(p) for p in (V2_MODEL_PATH, V2_PREPROCESSOR_PATH, V2_LABEL_ENCODER_PATH))
    print(f"V2 (retrained)  : {V2_MODEL_PATH}")

    v1_metrics = evaluate_with(loader.model, loader.preprocessor, loader.label_encoder, X_test, y_test)
    v2_metrics = evaluate_with(*v2_bundle, X_test, y_test)

    comparison = [
        {"metric": m, "model_v1": v1_metrics[m], "model_v2": v2_metrics[m],
         "change_v2_minus_v1": v2_metrics[m] - v1_metrics[m]}
        for m in v1_metrics
    ]
    comparison_df = pd.DataFrame(comparison)

    v1_pass = v1_metrics[gate_metric] >= gate_min
    v2_pass = v2_metrics[gate_metric] >= gate_min
    v2_improved = v2_metrics[gate_metric] > v1_metrics[gate_metric]

    print("\n" + "=" * 70)
    print("V1 VS V2 COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False, float_format="{:.4f}".format))
    print("\nQUALITY CHECKS")
    print("-" * 70)
    print(f"Gate               : {gate_metric} >= {gate_min:.2f}")
    print(f"V1 gate            : {'PASS' if v1_pass else 'FAIL'}")
    print(f"V2 gate            : {'PASS' if v2_pass else 'FAIL'}")
    print(f"V2 vs V1 ({gate_metric}) : {'IMPROVED' if v2_improved else 'NOT IMPROVED'}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(REPORT_PATH, index=False)
    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "stage": "Lab 12 - Model Comparison",
            "evaluation_dataset": "data/processed/X_test_raw.csv",
            "test_rows": int(len(X_test)),
            "model_v1": {"version": loader.model_version, "run_id": loader.run_id,
                         "metrics": {k: round(v, 6) for k, v in v1_metrics.items()}},
            "model_v2": {"version": "V2", "metrics": {k: round(v, 6) for k, v in v2_metrics.items()}},
            "comparison": comparison,
            "quality_gate": {"metric": gate_metric, "threshold": gate_min, "v1_passed": v1_pass,
                             "v2_passed": v2_pass, "v2_improved_over_v1": v2_improved},
        }, f, indent=4)

    print(f"\nCSV  : {REPORT_PATH}")
    print(f"JSON : {JSON_REPORT_PATH}")


if __name__ == "__main__":
    main()
