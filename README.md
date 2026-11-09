# Weather Daily Summary Classification — End-to-End MLOps Platform

Multiclass classification of the `Daily Summary` label (214 text classes, 96,453 hourly
rows from `weatherHistory.csv`), built with the same structure as the churn-prediction
MLOps project: Labs 3–6 (baseline, tracking, data pipeline, registry) plus Labs 7–13
(pipeline engineering, CI quality gate, failure handling, model serving, containers and
Kubernetes, monitoring/drift/retraining, and end-to-end automation).

## Table of Contents

1. [Project overview](#project-overview)
2. [How this differs from churn](#how-this-differs-from-churn)
3. [Architecture](#architecture)
4. [Repository layout](#repository-layout)
5. [Environment setup](#environment-setup)
6. [Labs 3–6](#labs-36)
7. [Lab 7 — ML pipeline](#lab-7--ml-pipeline)
8. [Lab 8 — CI and quality gate](#lab-8--ci-and-quality-gate)
9. [Lab 9 — Failure handling](#lab-9--failure-handling)
10. [Lab 10 — Model serving API](#lab-10--model-serving-api)
11. [Lab 11 — Docker and Kubernetes](#lab-11--docker-and-kubernetes)
12. [Lab 12 — Monitoring, drift and retraining](#lab-12--monitoring-drift-and-retraining)
13. [Lab 13 — End-to-end automation](#lab-13--end-to-end-automation)
14. [Reports and artifacts](#reports-and-artifacts)
15. [Sandbox verification](#sandbox-verification)
16. [Troubleshooting](#troubleshooting)

## Project overview

| Item | Value |
|---|---|
| Dataset | `data/raw/weatherHistory.csv` (Szeged, 2006–2016, hourly) |
| Target | `Daily Summary` — 214 classes, encoded with `LabelEncoder` |
| Features | 7 numeric readings, 5 time features from `Formatted Date` (UTC), one-hot `Summary` / `Precip Type` |
| Dropped | `Loud Cover` (constant 0.0) and 24 exact duplicate rows |
| Model | Random Forest, 50 trees, `max_depth=20`, `min_samples_leaf=5` (~35–40 MB) |
| Decision metric | Weighted F1 (`f1_score`); accuracy, weighted precision/recall and macro F1 also tracked |
| Quality gate | Weighted F1 ≥ 0.50 (baseline scores ≈ 0.57) |
| Registry name | `Weather_Daily_Summary_Production_Model` |
| API | FastAPI, `/predict` and `/predict/batch`, returns label + probability + top-3 |

## How this differs from churn

| Concern | Churn project | Weather project |
|---|---|---|
| Task | Binary (`Churn` yes/no) | 214-class multiclass |
| Gate / promotion metric | Recall ≥ 0.70 | Weighted F1 ≥ 0.50 |
| Row identity | `customerID` | `Formatted Date` timestamp (duplicates removed) |
| Fixed test set | `test_customer_ids.csv` | `test_row_keys.csv` (timestamps) |
| Deployment bundle | model + preprocessor | model + preprocessor + **label encoder** |
| API response | prediction + churn probability | class id, `daily_summary` text, probability, top-3 |
| Prediction drift | KS on 0/1 predictions | Jensen–Shannon on predicted classes + KS on confidence |
| Performance trend | 3 arbitrary windows | 4 **chronological** windows (2006→2016) |
| New batch | supplied `new_churn.csv` | `12_create_new_batch.py` generates `new_weather.csv` with new timestamps |

A few churn issues were fixed while porting:

* `11_prepare_deployment.py` no longer deletes the whole `deployment/` folder (which removed `deploy_production.py`); it only replaces bundle files.
* Retrained V2 is registered with its preprocessor/encoder under `preprocessing_pipeline/`, so it can actually be deployed if promoted.
* Stage 09 loads the preprocessor and encoder from the **same MLflow run** as the Production model.
* The pipeline accepts `--skip-deploy` so GitHub-hosted CI (no Kubernetes) can run the full ML path.
* Paths are resolved from the project root, so stages work from any working directory.

## Architecture

```text
GitHub Actions (manual / scheduled)  ──►  self-hosted runner
                                              │
                                   12_monitor_trigger.py
                     (data + prediction + concept drift → combined report)
                                              │ drift AND data/raw/new_weather.csv?
                                              ▼
                             weather_prediction_pipeline.py
01 Load (hash, lineage, archive) → 02 Schema validation → 03 Preprocess (fixed test set)
→ 04 Output validation → 05 Train + MLflow + Registry → 06 Evaluate → 07 Quality gate
→ 08 Lifecycle (Staging vs Production on weighted F1)
                                              │
                         promoted? ── no ──► skip deployment
                                              │ yes
                         11 Prepare bundle → Docker image model-vN → kubectl rollout
                                              │
                                    09 Production inference
```

See `docs/ARCHITECTURE.md` for component responsibilities.

## Repository layout

```text
weather-daily-summary-classification/
├── .github/workflows/          lab8_ci.yml, lab9_failure_tests.yml, lab13_monitoring.yml, lab13_runner_test.yml
├── api/                        FastAPI service (main, schemas, preprocessing, model_loader, monitoring_logger)
├── config.yaml                 Labs 7–13 configuration
├── data/raw/weatherHistory.csv
├── deployment/deploy_production.py   (bundle files are generated here)
├── docs/                       architecture, runbook, per-lab guides
├── k8s/                        deployment.yaml, service.yaml
├── lab11_commands/             Docker / Kubernetes command reference
├── notebooks/project_implementation.ipynb
├── pipelines/                  Lab 3–6 runners
├── src/
│   ├── config.py, preprocess*.py, train*.py, evaluate*.py, validate_*.py ...   Labs 3–6
│   ├── simulate_data_corruption.py                                           Lab 5
│   └── mlops_pipeline/                                                       Labs 7–13
│       ├── weather_common.py, validation.py, monitoring_common.py
│       ├── 01_load_data.py … 11_prepare_deployment.py, deploy_if_promoted.py
│       ├── weather_prediction_pipeline.py
│       └── 12_*.py              monitoring, drift, retraining, trigger
├── tests/                      test_api.py, test_performance.py, test_local_vs_container.py
├── Dockerfile, .dockerignore, requirements.txt, requirements-api.txt
└── check_mlflow.py, inspect_mlflow_run.py, inspect_model_version.py
```

Generated files (processed data, models, deployment bundle, reports, MLflow runs) are not
shipped; the pipeline creates them.

## Environment setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Run everything from the project root.

## Labs 3–6

```bash
python pipelines/run_lab3_baseline.py   # baseline RF, report, confusion matrix
python pipelines/run_lab4_tracking.py   # MLflow tracking, reproducibility
python pipelines/run_lab5_pipeline.py   # pandera validation + ColumnTransformer
python -m src.simulate_data_corruption  # Lab 5 fault injection demo
python pipelines/run_lab6_registry.py   # registry, champion/challenger
mlflow ui
```

## Lab 7 — ML pipeline

```bash
python src/mlops_pipeline/weather_prediction_pipeline.py --force
```

`--force` trains even when no new batch exists (first run, CI). Without it the pipeline
stops after Data Loading when the status is `no_new_data` or `already_ingested`.
Run log: `logs/weather_prediction_pipeline_run_log.json`. Details: `docs/LAB7_PIPELINE.md`.

On the very first run there is no Production model, so version 1 is promoted and the
deployment stage runs. If Docker/Kubernetes are not ready yet, use `--skip-deploy` and
deploy later (Lab 11).

## Lab 8 — CI and quality gate

`07_quality_gate.py` blocks promotion when weighted F1 < `quality_gate.minimum_value`.
`.github/workflows/lab8_ci.yml` runs the pipeline on push/PR with
`--force --skip-deploy`. Set the `MLFLOW_TRACKING_URI` secret for persistent history
(`docs/MLFLOW_CI_SETUP.md`).

## Lab 9 — Failure handling

```bash
python src/mlops_pipeline/10_failure_injection.py
```

Six scenarios × 3 repetitions: missing data, invalid datatype, out-of-range values,
schema mismatch, NaN in processed output (all must HALT), and a deleted model recovered
from the MLflow Production version. Report: `reports/failure_report.json`.

## Lab 10 — Model serving API

```bash
# needs the deployment bundle: run the pipeline (or 11_prepare_deployment.py) first
uvicorn api.main:app --reload
pytest tests/test_api.py -v
pytest tests/test_performance.py -v -s
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "formatted_date": "2016-09-09 14:00:00.000 +0200",
  "summary": "Partly Cloudy", "precip_type": "rain",
  "temperature_c": 24.5, "apparent_temperature_c": 24.5, "humidity": 0.45,
  "wind_speed_kmh": 11.3, "wind_bearing_degrees": 200, "visibility_km": 15.8,
  "pressure_millibars": 1014.2 }'
```

Response: `{"prediction": 197, "daily_summary": "...", "probability": 0.27, "top_3": [...]}`.
Every request is logged to `logs/inference/inference_log.jsonl`.

## Lab 11 — Docker and Kubernetes

```bash
python src/mlops_pipeline/11_prepare_deployment.py
docker build -t weather-summary-api:1.0 .
docker run -d -p 8000:8000 --name weather-api weather-summary-api:1.0
python -m tests.test_local_vs_container
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
python deployment/deploy_production.py      # builds weather-summary-api:model-vN and rolls it out
```

Full command list: `lab11_commands/lab11_docker_kubernetes_reference.txt`; guide:
`docs/LAB11_CONTAINERIZATION.md`.

## Lab 12 — Monitoring, drift and retraining

```bash
# drift simulation and detection
python src/mlops_pipeline/12_create_evolved_dataset.py
python src/mlops_pipeline/12_create_concept_drift_dataset.py
python src/mlops_pipeline/12_data_drift.py
python src/mlops_pipeline/12_prediction_drift.py
python src/mlops_pipeline/12_concept_drift.py
python src/mlops_pipeline/12_drift_analysis.py

# live monitoring from API logs / test set
python src/mlops_pipeline/12_prediction_monitoring.py
python src/mlops_pipeline/12_api_performance_monitoring.py
python src/mlops_pipeline/12_request_history.py
python src/mlops_pipeline/12_performance_monitoring.py
python src/mlops_pipeline/12_performance_trend.py

# continuous training
python src/mlops_pipeline/12_create_updated_dataset.py
python src/mlops_pipeline/12_retrain_model.py
python src/mlops_pipeline/12_model_comparison.py
python src/mlops_pipeline/12_update_mlflow_registry.py   # V2 -> Staging
python src/mlops_pipeline/08_automate_lifecycle.py        # V2 competes with Production
```

Details and thresholds: `docs/LAB12_MONITORING.md`.

## Lab 13 — End-to-end automation

```bash
python src/mlops_pipeline/12_create_new_batch.py --days 30   # simulate a new incoming batch
python src/mlops_pipeline/12_monitor_trigger.py
```

`.github/workflows/lab13_monitoring.yml` runs the trigger on a self-hosted Windows runner
(edit `PROJECT_DIR` in the workflow). Operating guide: `docs/RUNBOOK.md`,
`docs/LAB13_END_TO_END.md`.

## Reports and artifacts

| File | Purpose |
|---|---|
| `artifacts/data_load_report.json` | Ingestion status and batch hash |
| `data/raw/archive/dataset_lineage.json` | Ingested batches and dataset versions |
| `data/processed/test_row_keys.csv` | Fixed test-set timestamps |
| `reports/evaluation_report.json`, `reports/per_class_report.csv` | Metrics and per-class F1 |
| `reports/quality_gate_report.json` | Gate decision |
| `reports/lifecycle_report.json` | Promotion decision |
| `reports/deployment_automation_report.json` | Image, model version, rollout |
| `reports/failure_report.json` | Lab 9 results |
| `deployment/model_metadata.json` | Identity of the deployed model |
| `outputs/drift/*.csv` | Drift reports |
| `outputs/retraining/*` | Retraining and V1-vs-V2 comparison |
| `outputs/monitoring/monitor_trigger_report.json` | Trigger decision |
| `logs/weather_prediction_pipeline_run_log.json` | Stage-by-stage status |

## Sandbox verification

The non-MLflow parts were run end to end in a clean Python 3.13 / scikit-learn 1.9
sandbox before packaging (MLflow, pandera, FastAPI, Docker and Kubernetes were not
available there, so stages 01–02 schema checks, 05, 08, 09, 11, the HTTP tests and the
deployment were not executed and must be verified on your machine):

| Check | Result |
|---|---|
| Preprocessing | 24 duplicates removed; 77,143 train / 19,286 fixed test rows; 214 classes |
| Baseline RF on fixed test set | accuracy 0.588, weighted F1 0.567, macro F1 0.497 |
| API preprocessing vs training preprocessing | identical feature matrices |
| Data drift | 5 of 7 features flagged; Pressure (control) OK |
| Prediction drift | predicted-class JS 0.069 (OK); confidence KS 0.230 (DRIFT) |
| Concept drift | weighted F1 0.567 → 0.187 on evolved labelled data |
| Retrained V2 vs V1 | weighted F1 0.570 vs 0.567 (V2 improved, both pass gate) |
| New batch ingestion | 720 rows appended, archived as `weather_v01.csv`; same file re-run → `already_ingested`; overlapping timestamps rejected |

## Troubleshooting

* **`already_ingested`** — the batch hash is in `dataset_lineage.json`. Generate a new batch with `12_create_new_batch.py`; do not edit lineage.
* **Quality gate fails** — check `reports/evaluation_report.json`; loosen `model` params in `config.yaml` (bigger models score higher but grow quickly in size).
* **API returns 503** — the deployment bundle is missing; run `11_prepare_deployment.py`.
* **Model files are huge** — keep `max_depth`/`min_samples_leaf` limits; with 214 classes every leaf stores 214 probabilities.
* **Scores look optimistic** — `Daily Summary` is shared by the 24 hours of a day, so a random row split puts near-duplicates in train and test. For an honest estimate, split by calendar date.
* **Workflow stays queued** — the self-hosted runner must be online (`docs/RUNBOOK.md`).
* **`InconsistentVersionWarning` in the container** — the model was pickled with a different scikit-learn than `requirements-api.txt` pins. Train with the same version you serve with (or update the pin and rebuild).
