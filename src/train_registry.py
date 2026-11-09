"""Lab 6 - train a Random Forest and register it in the MLflow Model Registry."""
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from src.config import (MLFLOW_EXPERIMENT, MODELS_DIR, REGISTERED_MODEL_NAME,
                        classification_metrics, load_xy_arrays)


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    try:
        X_train, X_test, y_train, y_test = load_xy_arrays()
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run the Lab 5 pipeline first.")
        raise SystemExit(1)

    # Version 1 baseline params. Change these (e.g. deeper trees) to register a V2 challenger.
    params = {
        "n_estimators": 50,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    # params = {"n_estimators": 100, "max_depth": 25, "min_samples_leaf": 3,
    #           "random_state": 42, "n_jobs": -1}

    with mlflow.start_run(run_name="RandomForest_Registry_V1"):
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        print("[INFO] Training RandomForest model...")
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)

        print("[INFO] Evaluating model...")
        metrics = classification_metrics(y_test, rf_model.predict(X_test))
        mlflow.log_metrics(metrics)

        print("[INFO] Attaching preprocessing pipeline and label encoder to model artifacts...")
        mlflow.log_artifact(str(MODELS_DIR / "preprocessor.pkl"), artifact_path="preprocessing_pipeline")
        mlflow.log_artifact(str(MODELS_DIR / "label_encoder.pkl"), artifact_path="preprocessing_pipeline")

        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            name="random_forest_model",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"[SUCCESS] Metrics logged: Accuracy = {metrics['accuracy']:.4f} | "
              f"F1 (weighted) = {metrics['f1_score']:.4f}")
        print(f"[SUCCESS] Model registered under name: '{REGISTERED_MODEL_NAME}'")


if __name__ == "__main__":
    train_and_register_model()
