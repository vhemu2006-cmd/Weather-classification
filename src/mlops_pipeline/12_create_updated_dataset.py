"""Lab 12 - build a post-drift labelled batch and the updated training dataset for retraining.

The batch re-samples training rows, applies the same environment drift as the
evolved dataset and the same concept-drift relabelling, then is appended to
the original training rows. The fixed test set is never touched.
"""
import importlib

import numpy as np
import pandas as pd

from monitoring_common import RETRAIN_DIR, TARGET, X_TEST_RAW, X_TRAIN_RAW, Y_TRAIN, labels_from_ids
from weather_common import project_path

evolved_module = importlib.import_module("12_create_evolved_dataset")
concept_module = importlib.import_module("12_create_concept_drift_dataset")

NEW_BATCH_PATH = project_path("data/monitoring/drift/post_drift_training_data.csv")
UPDATED_DATASET_PATH = project_path("data/processed/updated_training_data.csv")
SUMMARY_PATH = RETRAIN_DIR / "updated_dataset_summary.csv"

RANDOM_SEED = 42
NEW_BATCH_SIZE = 5000


def create_post_drift_batch(X_train, y_train, rng):
    idx = rng.choice(len(X_train), size=NEW_BATCH_SIZE, replace=True)
    X_new = X_train.iloc[idx].reset_index(drop=True)
    y_new = np.asarray(y_train, dtype=object)[idx]

    X_new = evolved_module.apply_environment_drift(X_new, rng)
    y_new = concept_module.apply_concept_drift(X_new, y_new, rng)
    return X_new, y_new


def validate_new_batch(X_new, y_new, X_test):
    if list(X_new.columns) != list(X_test.columns):
        raise ValueError("Feature schema mismatch between new batch and test feature schema.")
    if len(X_new) != len(y_new):
        raise ValueError("New batch feature/label count mismatch.")
    if X_new.drop(columns=["Precip Type"]).isnull().any().any():
        raise ValueError("New batch contains unexpected missing values.")
    if not X_new["Humidity"].between(0, 1).all():
        raise ValueError("Humidity outside [0, 1].")
    if not X_new["Wind Speed (km/h)"].between(0, 150).all():
        raise ValueError("Wind speed outside [0, 150].")

    # The post-drift batch must never contain an exact copy of a held-out test row.
    test_hashes = set(pd.util.hash_pandas_object(X_test, index=False).astype("uint64"))
    new_hashes = pd.util.hash_pandas_object(X_new, index=False).astype("uint64")
    overlap = int(sum(h in test_hashes for h in new_hashes))
    if overlap:
        raise ValueError(f"Detected {overlap} exact feature-row overlaps with the test set.")
    return overlap


def summarize(name, labels):
    labels = pd.Series(labels)
    top = labels.value_counts()
    return {
        "dataset": name,
        "rows": len(labels),
        "distinct_classes": int(top.size),
        "top_class": top.index[0],
        "top_class_share": round(float(top.iloc[0] / len(labels)), 6),
        "foggy_all_day_share": round(float((labels == concept_module.HUMID_LABEL).mean()), 6),
    }


def main():
    print("=" * 70)
    print("UPDATED TRAINING DATASET CREATION")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_SEED)
    X_train = pd.read_csv(X_TRAIN_RAW)
    y_train = labels_from_ids(np.load(Y_TRAIN))
    X_test = pd.read_csv(X_TEST_RAW)
    if len(X_train) != len(y_train):
        raise ValueError(f"Training feature/label mismatch: X={len(X_train)}, y={len(y_train)}")

    print(f"\nTraining features : {X_train.shape}")
    print(f"Test features     : {X_test.shape}")

    X_new, y_new = create_post_drift_batch(X_train, y_train, rng)
    overlap = validate_new_batch(X_new, y_new, X_test)
    print(f"\nPost-drift batch  : {X_new.shape} | test-set overlap: {overlap}")

    original = X_train.copy()
    original[TARGET] = y_train
    batch = X_new.copy()
    batch[TARGET] = y_new
    updated = pd.concat([original, batch], ignore_index=True)

    NEW_BATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(NEW_BATCH_PATH, index=False)
    updated.to_csv(UPDATED_DATASET_PATH, index=False)

    summary = pd.DataFrame([
        summarize("original_training", original[TARGET]),
        summarize("post_drift_batch", batch[TARGET]),
        summarize("updated_training", updated[TARGET]),
    ])
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print()
    print(summary.to_string(index=False))
    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print(f"Post-drift batch  : {NEW_BATCH_PATH}")
    print(f"Updated dataset   : {UPDATED_DATASET_PATH}")
    print(f"Summary           : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
