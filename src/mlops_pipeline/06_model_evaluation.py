"""Stage 06 - Evaluate the trained model on the fixed held-out test set."""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from weather_common import classification_metrics, load_config, project_path


def run_evaluation():
    print("[INFO] Starting ML Pipeline Model Evaluation...")
    config = load_config()

    model_path = project_path(config["paths"]["model"])
    processed_dir = project_path(config["data"]["processed_dir"])

    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}. Run 05_train_registry.py first.")
        return False

    try:
        X_test = np.load(processed_dir / "X_test_final.npy")
        y_test = np.load(processed_dir / "y_test.npy")
    except FileNotFoundError:
        print("[ERROR] Test data not found. Run 03_preprocessing.py first.")
        return False

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    metrics = classification_metrics(y_test, y_pred)

    print("\n--- ML Pipeline: Model Evaluation ---")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():12}: {value:.4f}")
    print("--------------------------------\n")

    # Per-class report: with 214 classes a confusion matrix in JSON is unreadable,
    # so the 10 most frequent and 10 weakest classes are summarised instead.
    label_encoder = joblib.load(project_path(config["paths"]["label_encoder"]))
    per_class = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T
    per_class = per_class.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    per_class.index = [label_encoder.inverse_transform([int(float(i))])[0] for i in per_class.index]
    per_class = per_class.rename(columns={"f1-score": "f1"}).round(4)
    reports_dir = project_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    per_class.sort_values("support", ascending=False).to_csv(reports_dir / "per_class_report.csv")

    supported = per_class[per_class["support"] >= 20]
    report = {
        "stage": "ML Pipeline - Model Evaluation",
        "evaluated_on": "fixed held-out test split",
        "test_rows": int(len(y_test)),
        "n_classes_in_test": int(len(np.unique(y_test))),
        "metrics": {name: round(float(value), 4) for name, value in metrics.items()},
        "most_frequent_classes": per_class.sort_values("support", ascending=False)
                                          .head(10)[["f1", "support"]].to_dict(orient="index"),
        "weakest_classes_support_ge_20": supported.sort_values("f1")
                                                  .head(10)[["f1", "support"]].to_dict(orient="index"),
        "per_class_report": "reports/per_class_report.csv",
    }

    report_path = project_path(config["paths"]["evaluation_report"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Evaluation report saved to {config['paths']['evaluation_report']}")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_evaluation() else 1)
