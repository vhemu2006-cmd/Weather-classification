"""Lab 4 - evaluation with error analysis (exports misclassified observations)."""
import joblib
import numpy as np
import pandas as pd

from src.config import (DATE_COL, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, TARGET,
                        classification_metrics, load_raw)


def run_evaluation():
    print("Starting Model Evaluation...")
    X_test = np.load(PROCESSED_DIR / "X_test_final.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    test_index = np.load(PROCESSED_DIR / "test_index.npy")
    model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

    y_pred = model.predict(X_test)
    m = classification_metrics(y_test, y_pred)

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy          : {m['accuracy']:.4f}")
    print(f"Precision (wtd)   : {m['precision']:.4f}")
    print(f"Recall (wtd)      : {m['recall']:.4f}")
    print(f"F1-Score (wtd)    : {m['f1_score']:.4f}")
    print(f"F1-Score (macro)  : {m['f1_macro']:.4f}")
    print("-------------------------------\n")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    raw_test = load_raw().iloc[test_index].copy()
    errors = raw_test.assign(
        Actual_Summary=label_encoder.inverse_transform(y_test),
        Predicted_Summary=label_encoder.inverse_transform(y_pred),
    )
    errors = errors[errors["Actual_Summary"] != errors["Predicted_Summary"]]
    errors.to_csv(OUTPUTS_DIR / "misclassified_observations.csv", index=False)

    confusions = (
        errors.groupby(["Actual_Summary", "Predicted_Summary"]).size()
        .sort_values(ascending=False).head(25).rename("count").reset_index()
    )
    confusions.to_csv(OUTPUTS_DIR / "top_confusions.csv", index=False)
    print(f"{len(errors)} misclassified rows exported; top confusions saved.")


if __name__ == "__main__":
    run_evaluation()
