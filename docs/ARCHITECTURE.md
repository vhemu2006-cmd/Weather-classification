# Weather Daily Summary MLOps — Architecture

## Purpose

This repository classifies the hourly `Daily Summary` text (214 classes) of the Szeged
weather dataset and wraps the model in a full MLOps lifecycle: data versioning and
validation, preprocessing with a fixed test set, MLflow tracking and model registry, a
weighted-F1 quality gate, champion/challenger promotion, promotion-gated Docker/Kubernetes
deployment, an inference API with request logging, drift monitoring and retraining.

## End-to-end flow

```text
GitHub Actions (manual or scheduled)
              |
              v
Self-hosted runner  ──►  12_monitor_trigger.py
                           |  1. 12_data_drift / 12_prediction_drift / 12_concept_drift
                           |  2. 12_drift_analysis  -> outputs/drift/drift_analysis_report.csv
                           |  3. data/raw/new_weather.csv present?
                           v
            weather_prediction_pipeline.py   (only when drift AND a batch are present)
                           |
01 Data loading: schema check, SHA-256 hash, timestamp-overlap check, archive, lineage
                           |
02 Raw validation  ->  03 Preprocessing (dedupe, label encoder, fixed test set)  ->  04 Output validation
                           |
05 Training + MLflow run + registered version (model + preprocessor + label encoder)
                           |
06 Evaluation  ->  07 Quality gate (weighted F1 >= 0.50)
                           |
08 Lifecycle: None -> Staging; best Staging vs Production on weighted F1
                           |
          +----------------+----------------+
          |                                 |
     Not promoted                       Promoted
          |                                 |
   Skip deployment          11_prepare_deployment.py (bundle from MLflow)
                                            |
                            deploy_production.py: docker build weather-summary-api:model-vN
                                            |       kubectl set image + rollout status
                           |
09 Production inference (5-row demo from the Production model)
```

## Main components

| Component | Responsibility |
|---|---|
| `src/mlops_pipeline/weather_common.py` | Config loading, MLflow URI, shared cleaning (`clean_dataframe`), canonical `build_preprocessor`, multiclass metrics. |
| `src/mlops_pipeline/validation.py` | Pandera schema: value ranges, allowed `Summary`/`Precip Type` values, parsable timestamps, strict columns. |
| `01_load_data.py` | Validates and hashes `new_weather.csv`, rejects already-ingested hashes and overlapping timestamps, archives the previous dataset, appends rows and records lineage. |
| `03_preprocessing.py` | Removes duplicate rows, fits the label encoder, creates/reuses `test_row_keys.csv`, fits the ColumnTransformer on training rows only. |
| `05_train_registry.py` | Trains the Random Forest, logs params/metrics, logs `preprocessor.pkl` + `label_encoder.pkl` under `preprocessing_pipeline/`, registers a version. |
| `07_quality_gate.py` | Fails the pipeline if weighted F1 is below the minimum. Passing does not promote by itself. |
| `08_automate_lifecycle.py` | Promotes the best Staging version only if it beats Production; writes `reports/lifecycle_report.json`. |
| `deploy_if_promoted.py` | Runs packaging and deployment only when the lifecycle report says `promoted: true`. |
| `11_prepare_deployment.py` | Downloads the Production model and its matching preprocessor/encoder; verifies class ids decode; writes metadata. |
| `deployment/deploy_production.py` | Builds or reuses `weather-summary-api:model-vN`, updates the `weather-api` Deployment and waits for rollout. |
| `api/` | FastAPI service. Validates input with Pydantic, applies the same cleaning as training, returns label, probability and top-3, logs every request. |
| `12_*.py` | Drift simulation/detection, log-based monitoring, chronological performance trend, retraining V2, V1-vs-V2 comparison, V2 registry update, new-batch generator, monitoring trigger. |

## Design decisions specific to this dataset

* **Row identity.** The data has no ID column, so `Formatted Date` is the key. The raw
  file contains 24 exact duplicate rows; preprocessing drops them so the key is unique.
* **Label encoder in the bundle.** The model outputs integer class ids; the encoder that
  maps them back to text is versioned with each model run and shipped with the API.
* **Metrics on text labels.** Monitoring compares models in label-text space, so a V2
  whose encoder differs from V1's is still compared fairly.
* **Time features in UTC.** `Hour`, `Day` etc. are derived after converting the mixed
  `+0100`/`+0200` offsets to UTC, both in training and in the API.
* **Model size limits.** 214 classes make every leaf a 214-value vector; depth and leaf
  limits keep the model near 35–40 MB so it fits comfortably in a container.
* **Optimistic split.** All 24 hours of a day share one label, so a random split is
  optimistic. A date-grouped split is the honest alternative if you need it.

## Runtime dependencies and limits

* MLflow tracking (local `mlruns/` or a server via `MLFLOW_TRACKING_URI`).
* Docker and a Kubernetes cluster for deployment (Docker Desktop is enough).
* Drift datasets are simulated (`12_create_evolved_dataset.py`), and the incoming batch is
  generated from the latest month of real data with shifted timestamps.
