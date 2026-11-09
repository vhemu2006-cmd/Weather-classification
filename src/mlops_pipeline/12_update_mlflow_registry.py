"""Lab 12 - register retrained V2 in the MLflow Model Registry as a Staging candidate.

V2 is logged together with its own preprocessor and label encoder under
preprocessing_pipeline/, exactly like stage 05, so 11_prepare_deployment.py
can package it if the lifecycle manager later promotes it. Production is not
changed here; run 08_automate_lifecycle.py to let V2 compete.
"""
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from monitoring_common import RETRAIN_DIR
from weather_common import configure_mlflow, load_config, project_path

V2_MODEL_PATH = project_path("models/retrained_model_v2.pkl")
V2_PREPROCESSOR_PATH = project_path("models/retrained_preprocessor_v2.pkl")
V2_LABEL_ENCODER_PATH = project_path("models/retrained_label_encoder_v2.pkl")
RETRAINING_REPORT_PATH = RETRAIN_DIR / "retraining_report.json"
COMPARISON_REPORT_PATH = RETRAIN_DIR / "model_comparison_report.json"
REGISTRY_REPORT_PATH = RETRAIN_DIR / "mlflow_registry_update_report.json"


def get_production_version(client, model_name):
    production = [v for v in client.search_model_versions(f"name='{model_name}'")
                  if v.current_stage == "Production"]
    return production[0] if production else None


def register_v2(config, retraining_report, comparison_report):
    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    decision_metric = config["registry"]["decision_metric"]
    gate_metric = config["quality_gate"]["metric"]
    gate_min = float(config["quality_gate"]["minimum_value"])

    v2_metrics = retraining_report["metrics"]
    v1_metrics = comparison_report["model_v1"]["metrics"]

    production = get_production_version(client, model_name)
    production_metric = None
    if production:
        production_metric = client.get_run(production.run_id).data.metrics.get(decision_metric, 0.0)
        print(f"Production version : {production.version} | {decision_metric}={production_metric:.4f}")
    else:
        print("Production version : None")

    v2_metric = float(v2_metrics[decision_metric])
    gate_passed = float(v2_metrics[gate_metric]) >= gate_min
    improves = production_metric is None or v2_metric > production_metric

    print(f"V2 {decision_metric}          : {v2_metric:.4f}")
    print(f"Quality gate        : {'PASS' if gate_passed else 'FAIL'}")
    print(f"Improves Production : {'YES' if improves else 'NO'}")

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="Lab12_Registry_V2") as run:
        mlflow.log_params({
            "lab": "Lab 12 - MLflow Registry Update",
            "model_family": "RandomForest",
            "model_candidate": "V2",
            "previous_production_version": production.version if production else "None",
            "source_retraining_run_id": retraining_report.get("mlflow_run_id") or "not_logged",
            "decision_metric": decision_metric,
            "training_dataset": retraining_report["training_dataset"],
            "training_rows": retraining_report["training_rows"],
        })
        mlflow.log_metrics({k: float(v) for k, v in v2_metrics.items()})
        mlflow.log_metric("v1_decision_metric", float(v1_metrics[decision_metric]))
        mlflow.log_metric("v2_vs_v1_decision_metric", v2_metric - float(v1_metrics[decision_metric]))

        # Same artifact layout as stage 05 so the deployment packager works for V2 too.
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy2(V2_PREPROCESSOR_PATH, Path(tmp) / "preprocessor.pkl")
            shutil.copy2(V2_LABEL_ENCODER_PATH, Path(tmp) / "label_encoder.pkl")
            mlflow.log_artifacts(tmp, artifact_path="preprocessing_pipeline")

        mlflow.log_artifact(str(RETRAINING_REPORT_PATH), artifact_path="retraining_reports")
        mlflow.log_artifact(str(COMPARISON_REPORT_PATH), artifact_path="retraining_reports")

        mlflow.sklearn.log_model(
            sk_model=joblib.load(V2_MODEL_PATH),
            name="retrained_random_forest_v2",
            registered_model_name=model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        registry_run_id = run.info.run_id

    matching = [v for v in client.search_model_versions(f"name='{model_name}'") if v.run_id == registry_run_id]
    if not matching:
        raise RuntimeError("Could not find the newly registered V2 model version.")
    v2_version = matching[0]

    tags = {
        "candidate": "V2",
        "source_retraining_run_id": retraining_report.get("mlflow_run_id") or "not_logged",
        "quality_gate": "PASSED" if gate_passed else "FAILED",
        "comparison_with_production": "IMPROVED" if improves else "NOT_IMPROVED",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name=model_name, version=v2_version.version, key=key, value=value)

    # Staging only - promotion is the lifecycle manager's decision.
    client.transition_model_version_stage(
        name=model_name, version=v2_version.version, stage="Staging", archive_existing_versions=False
    )

    production_after = get_production_version(client, model_name)
    report = {
        "stage": "Lab 12 - MLflow Registry Update",
        "model_name": model_name,
        "registered_version": int(v2_version.version),
        "registered_stage": "Staging",
        "registered_run_id": registry_run_id,
        "production_version_before": int(production.version) if production else None,
        "production_version_after": int(production_after.version) if production_after else None,
        "decision_metric": decision_metric,
        "v2_decision_metric": v2_metric,
        "quality_gate": tags["quality_gate"],
        "improves_production": improves,
        "promotion_to_production": False,
        "next_step": "python src/mlops_pipeline/08_automate_lifecycle.py",
    }
    REGISTRY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    return report


def main():
    print("=" * 70)
    print("LAB 12 - MLFLOW REGISTRY UPDATE")
    print("=" * 70)

    for path in (V2_MODEL_PATH, V2_PREPROCESSOR_PATH, V2_LABEL_ENCODER_PATH,
                 RETRAINING_REPORT_PATH, COMPARISON_REPORT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    config = load_config()
    configure_mlflow(config)
    with open(RETRAINING_REPORT_PATH, "r", encoding="utf-8") as f:
        retraining_report = json.load(f)
    with open(COMPARISON_REPORT_PATH, "r", encoding="utf-8") as f:
        comparison_report = json.load(f)

    report = register_v2(config, retraining_report, comparison_report)

    print("\n" + "=" * 70)
    print("REGISTRY UPDATE COMPLETE")
    print("=" * 70)
    print(f"Registered model    : {report['model_name']}")
    print(f"V2 registry version : {report['registered_version']} (Staging)")
    print(f"Production before   : {report['production_version_before']}")
    print(f"Production after    : {report['production_version_after']}")
    print(f"Quality gate        : {report['quality_gate']}")
    print(f"Better than Prod    : {'YES' if report['improves_production'] else 'NO'}")
    print(f"Report              : {REGISTRY_REPORT_PATH}")


if __name__ == "__main__":
    main()
