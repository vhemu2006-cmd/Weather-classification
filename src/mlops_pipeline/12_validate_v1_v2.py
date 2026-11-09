
"""Lab 12 - V1 vs V2 synthetic stress-test evaluation.

This is a synthetic robustness test, NOT independent real-world validation.
"""

import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

from monitoring_common import (
    load_production_loader,
    evaluate_with,
)

from weather_common import project_path


# Input dataset
VALIDATION_PATH = project_path(
    "data/monitoring/validation/synthetic_validation_data.csv"
)

# V2 artifacts
V2_MODEL_PATH = project_path("models/retrained_model_v2.pkl")
V2_PREPROCESSOR_PATH = project_path(
    "models/retrained_preprocessor_v2.pkl"
)
V2_ENCODER_PATH = project_path(
    "models/retrained_label_encoder_v2.pkl"
)

# Output directory
OUTPUT_DIR = project_path("outputs/retraining/validation")


def main():

    print("=" * 70)
    print("SYNTHETIC STRESS TEST: V1 vs V2")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load validation dataset
    # --------------------------------------------------

    df = pd.read_csv(VALIDATION_PATH)

    if "target" not in df.columns:
        raise ValueError("Target column missing from validation dataset.")

    X = df.drop(columns=["target"])
    y_ids = df["target"].to_numpy(dtype=int)

    print("\nDATASET INFORMATION")
    print("-" * 60)
    print("Records :", len(X))
    print("Features:", len(X.columns))

    # --------------------------------------------------
    # 2. Load V1
    # --------------------------------------------------

    loader = load_production_loader()

    model_v1 = loader.model
    preprocessor_v1 = loader.preprocessor
    encoder_v1 = loader.label_encoder

    # --------------------------------------------------
    # 3. Load V2
    # --------------------------------------------------

    model_v2 = joblib.load(V2_MODEL_PATH)
    preprocessor_v2 = joblib.load(V2_PREPROCESSOR_PATH)
    encoder_v2 = joblib.load(V2_ENCODER_PATH)

    print("\n[SUCCESS] V1 and V2 loaded.")

    # --------------------------------------------------
    # 4. Decode numeric labels
    # --------------------------------------------------

    if len(encoder_v1.classes_) != len(encoder_v2.classes_):
        raise ValueError("V1 and V2 class counts differ.")

    if not np.array_equal(
        encoder_v1.classes_,
        encoder_v2.classes_
    ):
        raise ValueError("V1 and V2 label encoders differ.")

    if y_ids.min() < 0 or y_ids.max() >= len(encoder_v1.classes_):
        raise ValueError("Validation target contains invalid class IDs.")

    y_text = encoder_v1.inverse_transform(y_ids)

    # --------------------------------------------------
    # 5. Evaluate both models
    # --------------------------------------------------

    models = {
        "V1": (model_v1, preprocessor_v1, encoder_v1),
        "V2": (model_v2, preprocessor_v2, encoder_v2),
    }

    results = []
    class_reports = {}

    for name, bundle in models.items():

        print("\n" + "=" * 70)
        print(f"{name} VALIDATION RESULTS")
        print("=" * 70)

        metrics = evaluate_with(
            *bundle,
            X,
            y_text
        )

        for metric, value in metrics.items():

            print(f"{metric.upper():12}: {value:.4f}")

            results.append({
                "model": name,
                "metric": metric,
                "value": float(value),
            })

        # Class-wise evaluation
        predictions, _ = __import__(
            "monitoring_common"
        ).predict_with(*bundle, X)

        report = classification_report(
            y_text,
            predictions,
            labels=encoder_v1.classes_,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()

        class_reports[name] = report_df

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        report_df.to_csv(
            OUTPUT_DIR / f"{name.lower()}_classwise_report.csv"
        )

    # --------------------------------------------------
    # 6. Save aggregate comparison
    # --------------------------------------------------

    comparison = pd.DataFrame(results)

    pivot = comparison.pivot(
        index="metric",
        columns="model",
        values="value",
    )

    pivot["V2_Improvement"] = pivot["V2"] - pivot["V1"]

    comparison.to_csv(
        OUTPUT_DIR / "synthetic_validation_metrics.csv",
        index=False,
    )

    pivot.to_csv(
        OUTPUT_DIR / "v1_v2_validation_comparison.csv"
    )

    # --------------------------------------------------
    # 7. Display final comparison
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL V1 vs V2 SYNTHETIC STRESS-TEST")
    print("=" * 70)

    print(pivot.to_string())

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)

    print("\nReports saved in:")
    print(OUTPUT_DIR)

    print(
        "\nNOTE: This is a synthetic robustness test using "
        "previously evaluated records."
    )

    # Save metadata
    metadata = {
        "validation_type": "Synthetic robustness stress test",
        "independent_real_world_validation": False,
        "records": int(len(X)),
        "features": int(len(X.columns)),
        "classes": int(len(encoder_v1.classes_)),
        "models_compared": ["V1", "V2"],
        "production_model_modified": False,
    }

    with open(
        OUTPUT_DIR / "validation_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
