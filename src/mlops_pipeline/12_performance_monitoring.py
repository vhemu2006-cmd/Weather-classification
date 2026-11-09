"""Lab 12 - Production model performance on the fixed test set, checked against the quality gate."""
import numpy as np

from monitoring_common import evaluate_with, load_production_loader, load_test_set
from weather_common import load_config


def main():
    config = load_config()
    gate_metric = config["quality_gate"]["metric"]
    gate_min = float(config["quality_gate"]["minimum_value"])

    X_test, y_test = load_test_set()
    loader = load_production_loader()
    metrics = evaluate_with(loader.model, loader.preprocessor, loader.label_encoder, X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE MONITORING")
    print("=" * 60)
    print("\nEvaluation dataset : X_test_raw.csv (fixed test set)")
    print(f"Test samples       : {len(X_test)}")
    print(f"Distinct classes   : {len(np.unique(y_test))}")
    print(f"\nModel name    : {loader.model_name}")
    print(f"Model version : {loader.model_version}")
    print(f"Run ID        : {loader.run_id}")

    print("\nPerformance metrics:")
    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")

    print("\nQuality gate:")
    print(f"{gate_metric} >= {gate_min:.2f} : {metrics[gate_metric] >= gate_min}")


if __name__ == "__main__":
    main()
