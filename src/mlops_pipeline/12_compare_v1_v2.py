
"""Lab 12 - Compare Production V1 and Retrained V2."""

import json
import joblib
import pandas as pd

from monitoring_common import (
    EVOLVED_LABELED_PATH,
    TARGET,
    evaluate_with,
    load_production_loader,
    load_test_set,
)

from weather_common import project_path

V2_MODEL_PATH = project_path("models/retrained_model_v2.pkl")
V2_PREPROCESSOR_PATH = project_path("models/retrained_preprocessor_v2.pkl")
V2_ENCODER_PATH = project_path("models/retrained_label_encoder_v2.pkl")

REPORT_PATH = project_path("outputs/retraining/v1_v2_comparison.csv")


def main():
    print("=" * 70)
    print("MODEL V1 vs V2 COMPARISON")
    print("=" * 70)

    # Load production V1
    loader = load_production_loader()

    v1 = (
        loader.model,
        loader.preprocessor,
        loader.label_encoder,
    )

    # Load retrained V2
    model_v2 = joblib.load(V2_MODEL_PATH)
    preprocessor_v2 = joblib.load(V2_PREPROCESSOR_PATH)
    encoder_v2 = joblib.load(V2_ENCODER_PATH)

    v2 = (
        model_v2,
        preprocessor_v2,
        encoder_v2,
    )

    print("\n[SUCCESS] V1 and V2 models loaded.")

    # Load baseline test data
    baseline_X, baseline_y = load_test_set()

    # Load evolved labelled data
    evolved = pd.read_csv(EVOLVED_LABELED_PATH)

    evolved_X = evolved.drop(columns=[TARGET])
    evolved_y = evolved[TARGET].to_numpy()

    datasets = {
        "Baseline": (baseline_X, baseline_y),
        "Evolved": (evolved_X, evolved_y),
    }

    models = {
        "V1": v1,
        "V2": v2,
    }

    results = []

    for dataset_name, (X, y) in datasets.items():

        print(f"\n{'=' * 70}")
        print(f"{dataset_name.upper()} DATASET")
        print("=" * 70)

        for model_name, bundle in models.items():

            metrics = evaluate_with(*bundle, X, y)

            print(f"\n{model_name} PERFORMANCE")
            print("-" * 50)

            for metric, value in metrics.items():
                print(f"{metric.upper():12}: {value:.4f}")

                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "metric": metric,
                    "value": value,
                })

    report = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    comparison = report.pivot(
        index="metric",
        columns=["dataset", "model"],
        values="value",
    )

    print(comparison.to_string())

    print(f"\nReport saved: {REPORT_PATH}")

    print("\nCOMPARISON COMPLETE")


if __name__ == "__main__":
    main()
