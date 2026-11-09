"""Stage 08 - Champion/challenger lifecycle in the MLflow Model Registry.

New versions go None -> Staging. The best Staging version is compared with
Production on the decision metric (weighted F1). It is promoted only if it is
strictly better; otherwise it is archived. The decision is written to
reports/lifecycle_report.json, which deploy_if_promoted.py reads.
"""
import json
import sys

from mlflow.tracking import MlflowClient

from weather_common import configure_mlflow, load_config, project_path

REPORT_PATH = project_path("reports/lifecycle_report.json")


def save_lifecycle_report(report):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


def transition(client, model_name, version, stage):
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage, archive_existing_versions=False
    )


def automate_model_lifecycle():
    print("[INFO] Starting Automated Model Lifecycle Manager...")
    config = load_config()
    configure_mlflow(config)

    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    metric = config["registry"]["decision_metric"]

    report = {
        "status": "success",
        "model_name": model_name,
        "decision_metric": metric,
        "candidate_version": None,
        "candidate_run_id": None,
        "candidate_metric": None,
        "production_version_before": None,
        "production_metric_before": None,
        "promoted": False,
        "decision": None,
        "production_version_after": None,
    }

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        for mv in [mv for mv in all_versions if mv.current_stage == "None"]:
            transition(client, model_name, mv.version, "Staging")
            print(f"[INFO] Version {mv.version} moved to Staging.")

        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging = [mv for mv in all_versions if mv.current_stage == "Staging"]
        production = [mv for mv in all_versions if mv.current_stage == "Production"]
        production_model = production[0] if production else None

        production_score = None
        if production_model:
            production_score = client.get_run(production_model.run_id).data.metrics.get(metric, 0.0)
            report["production_version_before"] = int(production_model.version)
            report["production_metric_before"] = production_score

        if not staging:
            report["decision"] = "NO_CANDIDATE"
            report["production_version_after"] = report["production_version_before"]
            save_lifecycle_report(report)
            print("[INFO] No models in Staging. Nothing to promote.")
            return True

        best_model, best_score = None, -1.0
        for mv in staging:
            score = client.get_run(mv.run_id).data.metrics.get(metric, 0.0)
            print(f"[INFO] Staging Version {mv.version} | {metric}={score:.4f}")
            if score > best_score:
                best_model, best_score = mv, score

        report["candidate_version"] = int(best_model.version)
        report["candidate_run_id"] = best_model.run_id
        report["candidate_metric"] = best_score

        if production_model is None:
            promote = True
            print("[INFO] No Production model exists. Staging model will become Production.")
        else:
            print(f"[INFO] Production Version {production_model.version} | {metric}={production_score:.4f}")
            promote = best_score > production_score
            print("[SUCCESS] Staging model is better than Production." if promote
                  else "[INFO] Production model remains unchanged.")

        if promote:
            transition(client, model_name, best_model.version, "Production")
            if production_model:
                transition(client, model_name, production_model.version, "Archived")
            report.update(promoted=True, decision="PROMOTED",
                          production_version_after=int(best_model.version))
            print(f"[SUCCESS] Version {best_model.version} is now Production.")
        else:
            transition(client, model_name, best_model.version, "Archived")
            report.update(promoted=False, decision="ARCHIVED",
                          production_version_after=report["production_version_before"])
            print(f"[INFO] Version {best_model.version} archived because it did not improve.")

        for mv in client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == "Staging":
                transition(client, model_name, mv.version, "Archived")

        save_lifecycle_report(report)
        print("[SUCCESS] Automated Model Lifecycle execution complete!")
        print("[INFO] Lifecycle report saved to reports/lifecycle_report.json")
        return True

    except Exception as exc:
        report.update(status="failed", decision="ERROR", error=str(exc))
        save_lifecycle_report(report)
        print(f"[ERROR] Failed to automate model lifecycle: {exc}")
        return False


if __name__ == "__main__":
    sys.exit(0 if automate_model_lifecycle() else 1)
