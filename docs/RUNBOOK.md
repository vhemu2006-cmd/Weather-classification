# Runbook — Weather Daily Summary MLOps

## First-time setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
python src/mlops_pipeline/weather_prediction_pipeline.py --force --skip-deploy
python src/mlops_pipeline/11_prepare_deployment.py
docker build -t weather-summary-api:1.0 .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
python src/mlops_pipeline/12_create_evolved_dataset.py
python src/mlops_pipeline/12_create_concept_drift_dataset.py
```

After this, version 1 is in Production, the API runs in Kubernetes and the drift inputs exist.

## Before each automated run

1. Start Docker Desktop and wait until Kubernetes is ready.
2. Make sure the project checkout is on `main` and up to date.
3. Start the self-hosted runner in its own PowerShell window and keep it open:

   ```powershell
   cd C:\actions-runner
   .\run.cmd
   ```

4. Check the cluster:

   ```powershell
   docker info
   kubectl get nodes
   kubectl get deployment weather-api
   ```

## Start monitoring manually

GitHub → **Actions → Lab 13 - End-to-End MLOps → Run workflow** (branch `main`).
Or locally:

```bash
python src/mlops_pipeline/12_create_new_batch.py --days 30   # only if you want retraining
python src/mlops_pipeline/12_monitor_trigger.py
```

## Interpret the monitoring report

`outputs/monitoring/monitor_trigger_report.json`

| Status | Meaning |
|---|---|
| `no_drift` | Combined drift report shows no drift; the pipeline is not called. |
| `drift_waiting_for_data` | Drift detected but `data/raw/new_weather.csv` is missing. |
| `no_new_data_retraining_skipped` | The batch file exists but its hash was already ingested. |
| `retraining_completed` | New data ingested and the pipeline succeeded. Check `reports/lifecycle_report.json` for promotion. |
| `drift_analysis_failed` | A drift script failed (often: no deployment bundle, or evolved datasets not created). |

## Important reports

* `artifacts/data_load_report.json` — ingestion status, hash, row counts
* `data/raw/archive/dataset_lineage.json` — every ingested batch
* `outputs/drift/drift_analysis_report.csv` — combined drift
* `reports/evaluation_report.json`, `reports/per_class_report.csv` — metrics
* `reports/quality_gate_report.json` — gate result
* `reports/lifecycle_report.json` — promotion decision
* `reports/deployment_automation_report.json` — image and rollout
* `logs/weather_prediction_pipeline_run_log.json` — stage status

## Lifecycle behaviour

The gate requires weighted F1 ≥ 0.50. Passing the gate does not promote: the candidate
must beat the Production model's weighted F1. `deploy_if_promoted.py` skips deployment when
`promoted` is false.

## Troubleshooting

| Symptom | Action |
|---|---|
| Workflow queued forever | Runner offline. Check **Settings → Actions → Runners** and the `run.cmd` window. Check `PROJECT_DIR` in the workflow. |
| `kubectl` fails | Start Docker Desktop, wait for Kubernetes, run `kubectl get nodes`. |
| Retraining skipped | `already_ingested`: create a new batch with `12_create_new_batch.py`. Never edit lineage to force it. |
| Batch rejected for overlapping timestamps | The batch reuses times already in `weatherHistory.csv`; regenerate it. |
| Model not deployed | `reports/lifecycle_report.json` shows `promoted: false`; this is correct behaviour. |
| API 503 | Bundle files missing in `deployment/`; run `11_prepare_deployment.py` and rebuild the image. |
| Pod OOMKilled | Raise the memory limit in `k8s/deployment.yaml` or reduce model size in `config.yaml`. |
| Docker access denied from Python (Windows) | Compare the integrity level of the shell and `venv\Scripts\python.exe`. |

## Operational security

A self-hosted runner executes workflow code on your machine. Do not run untrusted
pull-request workflows on it; restrict who can trigger workflows.
