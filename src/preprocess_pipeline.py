"""Lab 5 - production preprocessing with a single scikit-learn ColumnTransformer."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import (CATEGORICAL, DROPPED_COLUMNS, MODELS_DIR, NUMERIC,
                        PROCESSED_DIR, RANDOM_STATE, TARGET, TEST_SIZE,
                        add_time_features, load_raw)


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    df = load_raw().drop(columns=DROPPED_COLUMNS)
    df = add_time_features(df)  # deterministic row-wise features: no leakage risk

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])
    X = df.drop(columns=[TARGET])

    # Split BEFORE fitting any transformer to prevent leakage.
    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, NUMERIC),
        ("cat", cat_pipeline, CATEGORICAL),
    ])

    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    np.save(PROCESSED_DIR / "X_train_final.npy", X_train_final)
    np.save(PROCESSED_DIR / "X_test_final.npy", X_test_final)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.astype(np.int64))
    np.save(PROCESSED_DIR / "y_test.npy", y_test.astype(np.int64))
    np.save(PROCESSED_DIR / "test_index.npy", idx_test)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    metadata = {
        "dataset_name": "Weather History - Daily Summary Classification",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "feature_names": [str(n) for n in preprocessor.get_feature_names_out()],
        "n_classes": int(len(label_encoder.classes_)),
    }
    (PROCESSED_DIR / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=4), encoding="utf-8"
    )
    print("[SUCCESS] Preprocessing completed! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
