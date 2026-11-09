"""Shared paths, constants and helpers for the weather-daily-summary project."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "weatherHistory.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
ARTIFACTS_DIR = ROOT / "artifacts"
LOGS_DIR = ROOT / "logs"

TARGET = "Daily Summary"
TARGET_ENCODED = "Daily_Summary_Encoded"
DATE_COL = "Formatted Date"
CATEGORICAL = ["Summary", "Precip Type"]
NUMERIC_RAW = [
    "Temperature (C)",
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)",
]
TIME_FEATURES = ["Year", "Month", "Day", "Hour", "DayOfWeek"]
NUMERIC = NUMERIC_RAW + TIME_FEATURES
# 'Loud Cover' (sic) is 0.0 in every row, so it carries no information.
DROPPED_COLUMNS = ["Loud Cover"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "Weather_Daily_Summary_Production_Model"
MLFLOW_EXPERIMENT = "Weather_Daily_Summary_Classification"

# With ~214 classes every leaf stores a probability vector, so unconstrained
# forests grow to several GB. These limits keep models in the tens/hundreds of MB.
RF_PARAMS = {
    "n_estimators": 50,
    "max_depth": 20,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

for _d in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, ARTIFACTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'Formatted Date' (to UTC) and derive Year/Month/Day/Hour/DayOfWeek."""
    df = df.copy()
    dt = pd.to_datetime(df[DATE_COL], utc=True)
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Hour"] = dt.dt.hour
    df["DayOfWeek"] = dt.dt.dayofweek
    return df.drop(columns=[DATE_COL])


def load_xy_arrays():
    """Load the processed numpy arrays produced by the Lab 4 / Lab 5 preprocessing."""
    import numpy as np

    return (
        np.load(PROCESSED_DIR / "X_train_final.npy"),
        np.load(PROCESSED_DIR / "X_test_final.npy"),
        np.load(PROCESSED_DIR / "y_train.npy"),
        np.load(PROCESSED_DIR / "y_test.npy"),
    )


def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
