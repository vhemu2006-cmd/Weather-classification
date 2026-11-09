"""Lab 6 - write a deployment-ready JSON report for the Production model."""
import json

from mlflow.tracking import MlflowClient

from src.config import ARTIFACTS_DIR, REGISTERED_MODEL_NAME


def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")
    client = MlflowClient()
    model_name = REGISTERED_MODEL_NAME

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in versions if mv.current_stage == "Production"), None)
        if not champion:
            print("[ERROR] No model found in Production stage!")
            raise SystemExit(1)

        run = client.get_run(champion.run_id)
        m = run.data.metrics
        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source,
            },
            "performance_metrics": {
                "accuracy": m.get("accuracy"),
                "precision_weighted": m.get("precision"),
                "recall_weighted": m.get("recall"),
                "f1_weighted": m.get("f1_score"),
                "f1_macro": m.get("f1_macro"),
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": "models/preprocessor.pkl",
            "label_decoder": "models/label_encoder.pkl",
        }

        ARTIFACTS_DIR.mkdir(exist_ok=True)
        report_path = ARTIFACTS_DIR / "production_model_report.json"
        report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
        print(f"[SUCCESS] Report generated for Version {champion.version}")
        print(f"[INFO] Saved to: {report_path}")

    except SystemExit:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    generate_registry_report()
