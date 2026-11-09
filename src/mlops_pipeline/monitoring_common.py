"""Shared helpers for the Lab 12 monitoring / drift / retraining scripts."""
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from weather_common import PROJECT_ROOT, classification_metrics, load_config, project_path

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from api.model_loader import ModelLoader  # noqa: E402
from api.preprocessing import clean_prediction_data  # noqa: E402

# Very rare hourly Summary values (e.g. 'Windy and Dry') may be missing from the
# training split; the encoder ignores them by design, so silence the repeated warning.
warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)

TARGET = "Daily Summary"
X_TEST_RAW = project_path("data/processed/X_test_raw.csv")
Y_TEST = project_path("data/processed/y_test.npy")
X_TRAIN_RAW = project_path("data/processed/X_train_raw.csv")
Y_TRAIN = project_path("data/processed/y_train.npy")
EVOLVED_PATH = project_path("data/monitoring/drift/evolved_data.csv")
EVOLVED_LABELED_PATH = project_path("data/monitoring/drift/evolved_labeled_data.csv")
DRIFT_DIR = project_path("outputs/drift")
RETRAIN_DIR = project_path("outputs/retraining")
INFERENCE_LOG = project_path("logs/inference/inference_log.jsonl")


def load_production_loader():
    loader = ModelLoader()
    loader.load()
    if not loader.is_ready():
        raise RuntimeError("Production model bundle is not available")
    return loader


def current_label_encoder():
    return joblib.load(project_path(load_config()["paths"]["label_encoder"]))


def labels_from_ids(ids):
    """Decode integer class ids saved by 03_preprocessing into Daily Summary text."""
    return current_label_encoder().inverse_transform(np.asarray(ids).astype(int))


def load_test_set():
    X_test = pd.read_csv(X_TEST_RAW)
    y_test = labels_from_ids(np.load(Y_TEST))
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test and y_test size mismatch: {len(X_test)} != {len(y_test)}")
    return X_test, y_test


def predict_with(model, preprocessor, label_encoder, X_raw):
    """Return (predicted Daily Summary text, max class probability) for raw rows."""
    X = preprocessor.transform(clean_prediction_data(X_raw))
    probabilities = model.predict_proba(X)
    best = probabilities.argmax(axis=1)
    class_ids = model.classes_.astype(int)[best]
    return label_encoder.inverse_transform(class_ids), probabilities.max(axis=1)


def predict_loader(loader, X_raw):
    return predict_with(loader.model, loader.preprocessor, loader.label_encoder, X_raw)


def evaluate_with(model, preprocessor, label_encoder, X_raw, y_text):
    """Metrics are computed on label text so models with different encoders compare fairly."""
    predictions, _ = predict_with(model, preprocessor, label_encoder, X_raw)
    return classification_metrics(y_text, predictions)


def load_inference_logs():
    if not INFERENCE_LOG.exists():
        raise FileNotFoundError(f"Log file not found: {INFERENCE_LOG}. Call the API first.")
    with INFERENCE_LOG.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError("Inference log is empty")
    return pd.DataFrame(records)


def js_divergence(baseline, evolved):
    """Jensen-Shannon divergence (base 2, range 0..1) between two categorical samples."""
    baseline = pd.Series(baseline).astype(str)
    evolved = pd.Series(evolved).astype(str)
    categories = sorted(set(baseline.unique()) | set(evolved.unique()))
    p = baseline.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
    q = evolved.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return float((kl(p, m) + kl(q, m)) / 2)
