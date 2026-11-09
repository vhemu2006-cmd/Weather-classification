# MLflow Tracking for Labs 7–13

The tracking URI is read from

1. the `MLFLOW_TRACKING_URI` environment variable, or
2. `mlflow.tracking_uri` in `config.yaml`.

Leave both empty for local development (MLflow uses `./mlruns`).

Local server with a SQLite registry:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

PowerShell: `$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"`
Linux/macOS: `export MLFLOW_TRACKING_URI=http://127.0.0.1:5000`

For persistent CI history, store the server URL as the GitHub Actions secret
`MLFLOW_TRACKING_URI`. A GitHub-hosted runner cannot reach a server that only listens on
your laptop; use an externally reachable server or the self-hosted runner.

Every model version is logged with
`preprocessing_pipeline/preprocessor.pkl` and `preprocessing_pipeline/label_encoder.pkl`
so `11_prepare_deployment.py` can rebuild a consistent bundle for any version.

Helpers: `python check_mlflow.py`, `python inspect_model_version.py 1`,
`python inspect_mlflow_run.py <run_id>`, `python src/mlops_pipeline/check_mlflow_registry.py`.
