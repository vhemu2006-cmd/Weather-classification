# Lab 8 — Continuous Integration and Quality Gate

## What CI does

`.github/workflows/lab8_ci.yml` runs on pushes and pull requests that touch
`config.yaml`, `src/mlops_pipeline/**`, `requirements.txt`, the raw dataset or the workflow:

1. Checkout, Python 3.11, `pip install -r requirements.txt`
2. `python src/mlops_pipeline/weather_prediction_pipeline.py --force --skip-deploy`
3. Upload `artifacts/`, `reports/`, `logs/`, `outputs/` as a build artifact

`--force` is needed because a fresh checkout has no new batch; `--skip-deploy` is needed
because GitHub-hosted runners have no Kubernetes cluster.

## Quality gate

`07_quality_gate.py` reads `reports/evaluation_report.json` and requires

```yaml
quality_gate:
  metric: f1_score     # weighted F1
  minimum_value: 0.50
```

The baseline Random Forest scores about 0.57 weighted F1 on the fixed test set, so the gate
blocks clearly broken models (wrong features, corrupted labels, a mis-set hyper-parameter)
without failing normal runs. A failed gate exits 1, the CI job turns red and lifecycle /
deployment never run.

## How to demonstrate the gate

1. Temporarily set `model.max_depth: 2` in `config.yaml` and push → the gate fails.
2. Revert → the gate passes.

Record the two run URLs, the gate reports from the uploaded artifacts and the stage log
as your evidence.

## Persistent MLflow history

Without `MLFLOW_TRACKING_URI`, each CI run uses a throw-away `mlruns/`, so every run
registers version 1. See `MLFLOW_CI_SETUP.md` for a persistent server.
