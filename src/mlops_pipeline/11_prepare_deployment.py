"""Stage 11 - Package the MLflow Production model as a deployment bundle.

deployment/
├── model.pkl            Production model from the registry
├── preprocessor.pkl     ColumnTransformer logged in the same MLflow run
├── label_encoder.pkl    class-id <-> Daily Summary text, same run
└── model_metadata.json  name, version, run id, source

Only the bundle files are replaced; deployment/deploy_production.py is kept.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from weather_common import configure_mlflow, load_config, project_path

BUNDLE_FILES = ["model.pkl", "preprocessor.pkl", "label_encoder.pkl", "model_metadata.json"]


def get_production_model(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        raise RuntimeError(f"No Production version found for '{model_name}'.")
    if len(production) > 1:
        raise RuntimeError(f"Multiple Production versions found for '{model_name}'.")
    return production[0]


def prepare_deployment():
    config = load_config()
    configure_mlflow(config)

    deployment_dir = project_path("deployment")
    deployment_dir.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        (deployment_dir / name).unlink(missing_ok=True)

    client = MlflowClient()
    mv = get_production_model(client, config["registry"]["model_name"])

    print("[INFO] Production model found")
    print(f"[INFO] Model name : {mv.name}")
    print(f"[INFO] Version    : {mv.version}")
    print(f"[INFO] Run ID     : {mv.run_id}")
    print(f"[INFO] Source     : {mv.source}")

    print("[INFO] Downloading Production model...")
    model = mlflow.sklearn.load_model(f"models:/{mv.name}/{mv.version}")
    model_path = deployment_dir / "model.pkl"
    joblib.dump(model, model_path, compress=3)
    print(f"[SUCCESS] Model saved to {model_path}")

    print("[INFO] Downloading matching preprocessor and label encoder...")
    with tempfile.TemporaryDirectory() as tmp:
        local_dir = Path(client.download_artifacts(mv.run_id, "preprocessing_pipeline", tmp))
        for name in ("preprocessor.pkl", "label_encoder.pkl"):
            if not (local_dir / name).exists():
                raise RuntimeError(f"Run {mv.run_id} has no preprocessing_pipeline/{name} artifact.")
            shutil.copy2(local_dir / name, deployment_dir / name)
            print(f"[SUCCESS] {name} saved to {deployment_dir / name}")

    print("[INFO] Verifying deployment artifacts...")
    loaded_model = joblib.load(model_path)
    preprocessor = joblib.load(deployment_dir / "preprocessor.pkl")
    label_encoder = joblib.load(deployment_dir / "label_encoder.pkl")
    if loaded_model is None or preprocessor is None or label_encoder is None:
        raise RuntimeError("Deployment artifact verification failed.")
    unknown = set(loaded_model.classes_.astype(int)) - set(range(len(label_encoder.classes_)))
    if unknown:
        raise RuntimeError("Model predicts class ids that the label encoder cannot decode.")

    metadata = {
        "model_name": mv.name,
        "model_version": str(mv.version),
        "model_stage": mv.current_stage,
        "run_id": mv.run_id,
        "model_source": mv.source,
        "model_id": mv.source.split("/")[-1],
        "preprocessor_artifact": "preprocessing_pipeline/preprocessor.pkl",
        "label_encoder_artifact": "preprocessing_pipeline/label_encoder.pkl",
        "n_classes": int(len(label_encoder.classes_)),
        "source": "MLflow",
    }
    with open(deployment_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SUCCESS] Metadata saved to {deployment_dir / 'model_metadata.json'}")

    print("\n[INFO] Deployment bundle:")
    print("deployment/")
    for name in BUNDLE_FILES:
        print(f"├── {name}")
    print("\n[SUCCESS] Deployment preparation completed.")


if __name__ == "__main__":
    try:
        prepare_deployment()
    except Exception as exc:
        print(f"[ERROR] Deployment preparation failed: {exc}")
        sys.exit(1)
