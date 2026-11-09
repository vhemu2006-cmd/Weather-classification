"""Shared helpers for the Lab 7-13 weather MLOps pipeline stages.

Every stage script is run as ``python src/mlops_pipeline/<stage>.py`` from the
project root, so this module is importable as ``from weather_common import ...``.
Paths are resolved against the project root, so stages also work when launched
from another working directory.
"""
import os
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
TIME_FEATURES = ["Year", "Month", "Day", "Hour", "DayOfWeek"]


def project_path(relative):
    """Return an absolute path inside the project."""
    return PROJECT_ROOT / relative


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_mlflow(config):
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def add_time_features(df, date_column="Formatted Date"):
    """Parse the timestamp (to UTC) and derive Year/Month/Day/Hour/DayOfWeek."""
    df = df.copy()
    dt = pd.to_datetime(df[date_column], utc=True)
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Hour"] = dt.dt.hour
    df["DayOfWeek"] = dt.dt.dayofweek
    return df.drop(columns=[date_column])


def clean_dataframe(df, config):
    """Deterministic cleaning shared by training, evaluation and inference.

    Drops the target and constant columns, turns the timestamp into time
    features and returns only the model feature columns in a fixed order.
    """
    data_cfg = config["data"]
    df = df.copy()

    drop = [c for c in data_cfg["dropped_columns"] + [data_cfg["target_column"]] if c in df.columns]
    df = df.drop(columns=drop)
    df = add_time_features(df, data_cfg["date_column"])

    features = data_cfg["numerical_features"] + data_cfg["categorical_features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[features]


def build_preprocessor(config):
    """The single canonical ColumnTransformer used by training and retraining."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # 517 rows have no Precip Type; treat missing as its own category.
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", num_pipeline, config["data"]["numerical_features"]),
        ("cat", cat_pipeline, config["data"]["categorical_features"]),
    ])


def classification_metrics(y_true, y_pred):
    """Multiclass metrics. Weighted averages suit the 214 imbalanced classes."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
