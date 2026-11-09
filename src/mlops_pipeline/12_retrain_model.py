"""Lab 12 - retrain model V2 on the updated (post-drift) dataset and log it to MLflow.

    python src/mlops_pipeline/12_retrain_model.py            # train + log to MLflow
    python src/mlops_pipeline/12_retrain_model.py --no-mlflow  # train + save locally only
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from monitoring_common import RETRAIN_DIR, TARGET, current_label_encoder, evaluate_with, load_test_set
from weather_common import build_preprocessor, clean_dataframe, configure_mlflow, load_config, project_path

UPDATED_DATASET_PATH = project_path("data/processed/updated_training_data.csv")
MODEL_V2_PATH = project_path("models/retrained_model_v2.pkl")
PREPROCESSOR_V2_PATH = project_path("models/retrained_preprocessor_v2.pkl")
LABEL_ENCODER_V2_PATH = project_path("models/retrained_label_encoder_v2.pkl")
REPORT_PATH = RETRAIN_DIR / "retraining_report.json"


def main(use_mlflow=True):
    print("=" * 70)
    print("LAB 12 - MODEL RETRAINING (V2)")
    print("=" * 70)

    config = load_config()
    if not UPDATED_DATASET_PATH.exists():
        raise FileNotFoundError(f"Updated training dataset not found: {UPDATED_DATASET_PATH}")

    updated = pd.read_csv(UPDATED_DATASET_PATH)
    X_test, y_test = load_test_set()

    label_encoder = current_label_encoder()
    unknown = set(updated[TARGET]) - set(label_encoder.classes_)
    if unknown:
        raise ValueError(f"Updated dataset contains {len(unknown)} labels unknown to the label encoder.")

    X_train = updated.drop(columns=[TARGET])
    y_train = label_encoder.transform(updated[TARGET])

    print(f"\nUpdated training data : {updated.shape}")
    print(f"Held-out test data    : {X_test.shape}")

    params = dict(config["model"])
    print("\nMODEL PARAMETERS")
    print("-" * 70)
    for name, value in params.items():
        print(f"{name:18}: {value}")

    preprocessor = build_preprocessor(config)
    X_train_transformed = preprocessor.fit_transform(clean_dataframe(X_train, config))
    print(f"\nTransformed features : {X_train_transformed.shape[1]}")

    print("\nTRAINING MODEL V2...")
    model = RandomForestClassifier(**params).fit(X_train_transformed, y_train)

    print("\nEVALUATING MODEL V2 ON THE ORIGINAL FIXED TEST SET")
    print("-" * 70)
    metrics = evaluate_with(model, preprocessor, label_encoder, X_test, y_test)
    for name, value in metrics.items():
        print(f"{name.upper():10}: {value:.4f}")

    MODEL_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_V2_PATH, compress=3)
    joblib.dump(preprocessor, PREPROCESSOR_V2_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_V2_PATH)
    print(f"\nModel V2        : {MODEL_V2_PATH}")
    print(f"Preprocessor V2 : {PREPROCESSOR_V2_PATH}")
    print(f"Label encoder   : {LABEL_ENCODER_V2_PATH}")

    run_id = None
    if use_mlflow:
        import mlflow

        configure_mlflow(config)
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="Lab12_Retraining_Model_V2") as run:
            mlflow.log_params(params)
            mlflow.log_params({
                "model_family": "RandomForest",
                "lab": "Lab 12 - Continuous Training",
                "training_dataset": "data/processed/updated_training_data.csv",
                "training_rows": len(X_train),
                "test_rows": len(X_test),
            })
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(UPDATED_DATASET_PATH), artifact_path="retraining_dataset")
            run_id = run.info.run_id
            print(f"\nMLflow run ID : {run_id}")

    report = {
        "stage": "Lab 12 - Retraining",
        "model_version": "V2",
        "training_dataset": "data/processed/updated_training_data.csv",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_classes": int(len(np.unique(y_train))),
        "model_parameters": params,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "model_path": "models/retrained_model_v2.pkl",
        "preprocessor_path": "models/retrained_preprocessor_v2.pkl",
        "label_encoder_path": "models/retrained_label_encoder_v2.pkl",
        "mlflow_run_id": run_id,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 70)
    print("RETRAINING COMPLETE")
    print(f"Report : {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    main(use_mlflow=not parser.parse_args().no_mlflow)
