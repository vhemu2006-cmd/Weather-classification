"""List every registered version of the weather model with its stage and F1."""
import mlflow
from mlflow.tracking import MlflowClient

from weather_common import configure_mlflow, load_config


def main():
    config = load_config()
    configure_mlflow(config)
    model_name = config["registry"]["model_name"]
    metric = config["registry"]["decision_metric"]

    client = MlflowClient()
    print(f"Model: {model_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print()
    print(f"VERSION | STAGE | {metric.upper()} | RUN_ID")

    versions = sorted(client.search_model_versions(f"name='{model_name}'"),
                      key=lambda v: int(v.version), reverse=True)
    for v in versions:
        score = client.get_run(v.run_id).data.metrics.get(metric)
        score_text = f"{score:.4f}" if score is not None else "n/a"
        print(f"{v.version} | {v.current_stage} | {score_text} | {v.run_id}")


if __name__ == "__main__":
    main()
