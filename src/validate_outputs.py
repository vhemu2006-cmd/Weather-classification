"""Lab 5 - validate preprocessing outputs and write a summary report."""
import json

import numpy as np

from src.config import ARTIFACTS_DIR, MODELS_DIR, load_xy_arrays


def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs...")
    try:
        X_train, X_test, y_train, y_test = load_xy_arrays()
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find processed data: {e}")
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
    # Every class must appear in train (stratified split guarantees it).
    missing_classes = sorted(set(np.unique(y_test)) - set(np.unique(y_train)))
    if missing_classes:
        errors.append(f"{len(missing_classes)} test classes never seen in training.")

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
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    report_path = ARTIFACTS_DIR / "preprocessing_summary_report.json"
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    if errors:
        print("[ERROR] Output Validation FAILED!")
        for e in errors:
            print(f"  - {e}")
        return False
    print("[SUCCESS] Output Validation PASSED.")
    print(f"[INFO] Summary report saved to {report_path}")
    return True


if __name__ == "__main__":
    if not validate_preprocessing_outputs():
        raise SystemExit(1)
