"""Lab 4 - preprocessing to numpy arrays (manual scaler + one-hot encoder)."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import (CATEGORICAL, DROPPED_COLUMNS, MODELS_DIR, NUMERIC,
                        PROCESSED_DIR, RANDOM_STATE, TARGET, TEST_SIZE,
                        add_time_features, load_raw)


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    df = load_raw().drop(columns=DROPPED_COLUMNS)
    df = add_time_features(df)
    # Missing Precip Type (517 rows) becomes its own category.
    df["Precip Type"] = df["Precip Type"].fillna("unknown")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])
    X = df.drop(columns=[TARGET])

    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[NUMERIC])
    x_test_scaled = scaler.transform(X_test[NUMERIC])

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    x_train_encoded = ohe.fit_transform(X_train[CATEGORICAL])
    x_test_encoded = ohe.transform(X_test[CATEGORICAL])

    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))

    np.save(PROCESSED_DIR / "X_train_final.npy", X_train_final)
    np.save(PROCESSED_DIR / "X_test_final.npy", X_test_final)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.astype(np.int64))
    np.save(PROCESSED_DIR / "y_test.npy", y_test.astype(np.int64))
    np.save(PROCESSED_DIR / "test_index.npy", idx_test)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(ohe, MODELS_DIR / "ohe.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    metadata = {
        "dataset_name": "Weather History - Daily Summary Classification",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "n_classes": int(len(label_encoder.classes_)),
    }
    (PROCESSED_DIR / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=4), encoding="utf-8"
    )
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
