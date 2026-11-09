"""Stage 05 - Train a Random Forest, track it in MLflow and register a model version."""
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from weather_common import classification_metrics, configure_mlflow, load_config, project_path


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")
    config = load_config()

    configure_mlflow(config)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    processed_dir = project_path(config["data"]["processed_dir"])
    try:
        X_train = np.load(processed_dir / "X_train_final.npy")
        X_test = np.load(processed_dir / "X_test_final.npy")
        y_train = np.load(processed_dir / "y_train.npy")
        y_test = np.load(processed_dir / "y_test.npy")
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run 03_preprocessing.py first.")
        return False

    params = dict(config["model"])

    with mlflow.start_run(run_name="RandomForest") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")
        mlflow.log_param("lab", "ML Pipeline")
        mlflow.log_param("n_classes_train", int(len(np.unique(y_train))))
        mlflow.log_param("training_rows", int(len(y_train)))

        if os.getenv("GITHUB_RUN_ID"):
            mlflow.log_param("github_run_id", os.getenv("GITHUB_RUN_ID"))
        if os.getenv("GITHUB_SHA"):
            mlflow.log_param("github_commit", os.getenv("GITHUB_SHA"))

        print("[INFO] Training RandomForest model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        print("[INFO] Evaluating model on the held-out test set...")
        metrics = classification_metrics(y_test, model.predict(X_test))
        mlflow.log_metrics(metrics)

        model_path = project_path(config["paths"]["model"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path, compress=3)

        # The preprocessor and label encoder travel with the model so that any
        # registered version can be deployed as a self-consistent bundle.
        for key in ("preprocessor", "label_encoder"):
            artifact = project_path(config["paths"][key])
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="preprocessing_pipeline")

        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            registered_model_name=config["registry"]["model_name"],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(
            f"[SUCCESS] Accuracy={metrics['accuracy']:.4f} | "
            f"F1 (weighted)={metrics['f1_score']:.4f} | "
            f"F1 (macro)={metrics['f1_macro']:.4f}"
        )
        print(f"[SUCCESS] Model saved to {config['paths']['model']}")
        print(f"[SUCCESS] Model registered as '{config['registry']['model_name']}'")
        print(f"[INFO] MLflow Run ID: {run.info.run_id}")

    return True


if __name__ == "__main__":
    sys.exit(0 if train_and_register_model() else 1)
