"""Stage 03 - Preprocessing with a fixed held-out test set.

The first run makes a stratified 80/20 split and saves the test-row timestamps
to data/processed/test_row_keys.csv. Every later run (e.g. after new batches are
ingested) reuses exactly the same test rows, and all other rows - including new
ones - go to training. This keeps model versions comparable.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from weather_common import build_preprocessor, clean_dataframe, load_config, project_path

TEST_KEYS_FILE = "test_row_keys.csv"


def deduplicate(df, key_column):
    """The raw file contains 24 exact duplicate rows (same timestamp); keep one of each."""
    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset=[key_column], keep="first")
    removed = before - len(df)
    if removed:
        print(f"[INFO] Removed {removed} duplicate rows.")
    return df.reset_index(drop=True)


def create_initial_test_keys(df, y, config, processed_dir):
    key_column = config["data"]["id_column"]
    test_keys_path = processed_dir / TEST_KEYS_FILE

    print(f"[INFO] {TEST_KEYS_FILE} not found. Creating the original stratified split...")
    counts = pd.Series(y).value_counts()
    stratify = y if counts.min() >= 2 else None
    if stratify is None:
        print("[WARNING] Some classes have a single row; falling back to a non-stratified split.")

    _, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=stratify,
    )
    test_keys = df.loc[idx_test, key_column].astype(str)
    test_keys.to_frame().to_csv(test_keys_path, index=False)

    print(f"[SUCCESS] Original test-set keys saved to {test_keys_path}")
    print(f"[INFO] Fixed test set size: {len(test_keys)}")
    return set(test_keys)


def split_raw_data(df, y, config, processed_dir):
    key_column = config["data"]["id_column"]
    test_keys_path = processed_dir / TEST_KEYS_FILE

    if test_keys_path.exists():
        test_keys = set(pd.read_csv(test_keys_path)[key_column].astype(str))
        print(f"[INFO] Using fixed test set from {test_keys_path}")
    else:
        test_keys = create_initial_test_keys(df, y, config, processed_dir)

    test_mask = df[key_column].astype(str).isin(test_keys).to_numpy()
    if not test_mask.any():
        raise ValueError("None of the fixed test keys were found in the current dataset.")

    target = config["data"]["target_column"]
    X_train_raw = df.loc[~test_mask].drop(columns=[target]).reset_index(drop=True)
    X_test_raw = df.loc[test_mask].drop(columns=[target]).reset_index(drop=True)
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"[INFO] Training records: {len(X_train_raw)}")
    print(f"[INFO] Fixed test records: {len(X_test_raw)}")
    return X_train_raw, X_test_raw, y_train, y_test


def run_preprocessing():
    print("[INFO] Starting Preprocessing...")
    config = load_config()

    data_path = project_path(config["data"]["raw_path"])
    processed_dir = project_path(config["data"]["processed_dir"])
    target = config["data"]["target_column"]
    key_column = config["data"]["id_column"]

    if not data_path.exists():
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    processed_dir.mkdir(parents=True, exist_ok=True)
    project_path("models").mkdir(parents=True, exist_ok=True)

    raw_df = deduplicate(pd.read_csv(data_path), key_column)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_df[target]).astype(np.int64)

    X_train_raw, X_test_raw, y_train, y_test = split_raw_data(raw_df, y, config, processed_dir)

    preprocessor = build_preprocessor(config)
    print("[INFO] Fitting preprocessing pipeline on training data...")
    X_train_final = preprocessor.fit_transform(clean_dataframe(X_train_raw, config))
    X_test_final = preprocessor.transform(clean_dataframe(X_test_raw, config))

    np.save(processed_dir / "X_train_final.npy", X_train_final)
    np.save(processed_dir / "X_test_final.npy", X_test_final)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "y_test.npy", y_test)

    X_train_raw.to_csv(processed_dir / "X_train_raw.csv", index=False)
    X_test_raw.to_csv(processed_dir / "X_test_raw.csv", index=False)

    joblib.dump(preprocessor, project_path(config["paths"]["preprocessor"]))
    joblib.dump(label_encoder, project_path(config["paths"]["label_encoder"]))

    metadata = {
        "dataset_name": "Weather History - Daily Summary Classification",
        "target_column": target,
        "n_classes": int(len(label_encoder.classes_)),
        "rows_after_deduplication": int(len(raw_df)),
        "train_shape_raw": list(X_train_raw.shape),
        "test_shape_raw": list(X_test_raw.shape),
        "train_shape_transformed": list(X_train_final.shape),
        "test_shape_transformed": list(X_test_final.shape),
        "numerical_features": config["data"]["numerical_features"],
        "categorical_features": config["data"]["categorical_features"],
        "dropped_columns": config["data"]["dropped_columns"],
        "feature_names": [str(n) for n in preprocessor.get_feature_names_out()],
        "split_test_size": config["split"]["test_size"],
        "split_random_state": config["split"]["random_state"],
        "fixed_test_set": True,
        "test_row_keys_path": f"{config['data']['processed_dir']}/{TEST_KEYS_FILE}",
        "raw_train_path": f"{config['data']['processed_dir']}/X_train_raw.csv",
        "raw_test_path": f"{config['data']['processed_dir']}/X_test_raw.csv",
    }
    with open(processed_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Preprocessing complete. Preprocessor saved to {config['paths']['preprocessor']}")
    print(f"[SUCCESS] Label encoder saved to {config['paths']['label_encoder']} "
          f"({len(label_encoder.classes_)} classes)")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_preprocessing() else 1)
