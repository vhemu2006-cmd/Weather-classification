"""Lab 3 - baseline preprocessing: time features, one-hot encoding, label encoding."""
import json

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (ARTIFACTS_DIR, CATEGORICAL, DROPPED_COLUMNS, MODELS_DIR,
                        PROCESSED_DIR, TARGET, TARGET_ENCODED, add_time_features,
                        load_raw)

PROCESSED = PROCESSED_DIR / "processed_weather.csv"
METADATA = PROCESSED_DIR / "dataset_metadata.json"
ARTIFACT = ARTIFACTS_DIR / "preprocessing_metadata.json"


def preprocess():
    df = load_raw()
    original_shape = list(df.shape)

    df = df.drop(columns=DROPPED_COLUMNS)
    missing_precip = int(df["Precip Type"].isna().sum())
    df = add_time_features(df)
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=False, dtype=int)

    label_encoder = LabelEncoder()
    df[TARGET_ENCODED] = label_encoder.fit_transform(df.pop(TARGET))
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    df.to_csv(PROCESSED, index=False)

    metadata = {
        "original_shape": original_shape,
        "processed_shape": list(df.shape),
        "dropped_columns": DROPPED_COLUMNS,
        "missing_precip_type_before_encoding": missing_precip,
        "precip_type_missing_handling": "all dummy columns are 0 for missing values",
        "date_features": ["Year", "Month", "Day", "Hour", "DayOfWeek"],
        "one_hot_encoding": "pd.get_dummies(columns=['Summary', 'Precip Type'])",
        "target": TARGET_ENCODED,
        "target_source": TARGET,
        "n_classes": int(len(label_encoder.classes_)),
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ARTIFACT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return df


if __name__ == "__main__":
    df = preprocess()
    print(f"Processed dataset saved to {PROCESSED}")
    print(f"Shape: {df.shape}")
