"""Lab 3 - baseline Random Forest training."""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import (ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE,
                        RF_PARAMS, TARGET_ENCODED, TEST_SIZE)

DATA = PROCESSED_DIR / "processed_weather.csv"
MODEL = MODELS_DIR / "random_forest_model.pkl"
SPLIT = ARTIFACTS_DIR / "split_data.json"


def train():
    df = pd.read_csv(DATA)
    X = df.drop(columns=[TARGET_ENCODED])
    y = df[TARGET_ENCODED]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL, compress=3)
    SPLIT.write_text(json.dumps({
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified": True,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "model_params": RF_PARAMS,
    }, indent=2), encoding="utf-8")
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = train()
    print(f"Model saved to {MODEL}")
    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")
