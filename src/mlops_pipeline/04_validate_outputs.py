"""Stage 04 - Validate the processed matrices before training."""
import json
import sys

import numpy as np

from weather_common import load_config, project_path


def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs...")
    config = load_config()
    processed_dir = project_path(config["data"]["processed_dir"])

    try:
        X_train = np.load(processed_dir / "X_train_final.npy")
        X_test = np.load(processed_dir / "X_test_final.npy")
        y_train = np.load(processed_dir / "y_train.npy")
        y_test = np.load(processed_dir / "y_test.npy")
    except FileNotFoundError as exc:
        print(f"[ERROR] Could not find processed data: {exc}")
        return False

    errors = []
    if np.isnan(X_train).sum() > 0:
        errors.append("NaNs detected in X_train after preprocessing.")
    if np.isnan(X_test).sum() > 0:
        errors.append("NaNs detected in X_test after preprocessing.")
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature mismatch: X_train has {X_train.shape[1]} cols, X_test has {X_test.shape[1]} cols.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Row mismatch between X_train and y_train.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Row mismatch between X_test and y_test.")

    unseen = sorted(set(np.unique(y_test)) - set(np.unique(y_train)))
    if unseen:
        errors.append(f"{len(unseen)} test classes never appear in training.")

    for name in ["X_train_raw.csv", "X_test_raw.csv", "test_row_keys.csv"]:
        if not (processed_dir / name).exists():
            errors.append(f"Required split artifact missing: {config['data']['processed_dir']}/{name}")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
        },
        "data_quality": {
            "missing_values_X_train": int(np.isnan(X_train).sum()),
            "missing_values_X_test": int(np.isnan(X_test).sum()),
            "n_classes_train": int(len(np.unique(y_train))),
            "n_classes_test": int(len(np.unique(y_test))),
        },
        "errors": errors,
    }

    artifacts_dir = project_path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "preprocessing_summary_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output Validation FAILED!")
        for error in errors:
            print(f"  - {error}")
        return False

    print("[SUCCESS] Output Validation PASSED.")
    print(f"[INFO] Summary report saved to {report_path}")
    return True


if __name__ == "__main__":
    sys.exit(0 if validate_preprocessing_outputs() else 1)
