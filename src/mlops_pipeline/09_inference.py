"""Stage 09 - Production inference demo using the registered Production model.

The preprocessor and label encoder are downloaded from the same MLflow run as
the Production model, so the three always match.
"""
import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from weather_common import clean_dataframe, configure_mlflow, load_config, project_path


def load_production_bundle(config):
    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    production = [mv for mv in client.search_model_versions(f"name='{model_name}'")
                  if mv.current_stage == "Production"]
    if not production:
        raise RuntimeError(f"No Production version found for '{model_name}'. Run 08_automate_lifecycle.py first.")
    mv = production[0]

    print(f"[INFO] Loading Production model: models:/{model_name}/{mv.version}")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{mv.version}")

    with tempfile.TemporaryDirectory() as tmp:
        local_dir = Path(client.download_artifacts(mv.run_id, "preprocessing_pipeline", tmp))
        preprocessor = joblib.load(local_dir / "preprocessor.pkl")
        label_encoder = joblib.load(local_dir / "label_encoder.pkl")
    return model, preprocessor, label_encoder, mv.version


def run_inference_demo():
    print("[INFO] Starting ML Pipeline Production Inference...")
    config = load_config()
    configure_mlflow(config)

    try:
        model, preprocessor, label_encoder, version = load_production_bundle(config)
    except Exception as exc:
        print(f"[ERROR] Could not load Production model bundle from MLflow: {exc}")
        return False

    raw_path = project_path(config["data"]["raw_path"])
    if not raw_path.exists():
        print(f"[ERROR] Raw dataset not found at {raw_path}")
        return False

    df = pd.read_csv(raw_path)
    sample = df.sample(n=5, random_state=7)
    X_sample = preprocessor.transform(clean_dataframe(sample, config))

    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample).max(axis=1)

    target = config["data"]["target_column"]
    result = sample.copy()
    result["Predicted_Daily_Summary"] = label_encoder.inverse_transform(predictions.astype(int))
    result["Prediction_Probability"] = np.round(probabilities, 4)
    result["Correct"] = result["Predicted_Daily_Summary"] == result[target]
    result["Model_Version"] = version

    print("\n--- ML Pipeline Production Inference Results ---")
    print(result[["Formatted Date", target, "Predicted_Daily_Summary", "Prediction_Probability"]]
          .to_string(index=False))
    print("-----------------------------------------\n")

    output_path = project_path(config["paths"]["inference_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[SUCCESS] Predictions saved to {config['paths']['inference_output']}")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_inference_demo() else 1)
