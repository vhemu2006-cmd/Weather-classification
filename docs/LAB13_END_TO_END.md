# Lab 13 — End-to-End MLOps Integration

## Objective

Connect data versioning, validation, preprocessing, training, MLflow registry, quality gate,
lifecycle decisions, deployment, monitoring and retraining triggers into one automated,
traceable loop.

## Flow

```text
lab13_monitoring.yml (manual / hourly)  ->  self-hosted runner
  -> 12_monitor_trigger.py
       -> 12_data_drift, 12_prediction_drift, 12_concept_drift, 12_drift_analysis
       -> drift? and data/raw/new_weather.csv?
            -> weather_prediction_pipeline.py
                 01 ingest (hash / overlap check / archive / lineage)
                 02-04 validate + preprocess
                 05 train + register, 06 evaluate, 07 gate
                 08 lifecycle -> deploy_if_promoted -> 11 bundle -> docker + kubectl rollout
                 09 production inference
```

## Scenarios to demonstrate

| # | Setup | Expected trigger status | Expected pipeline outcome |
|---|---|---|---|
| A | no `new_weather.csv` | `drift_waiting_for_data` | pipeline not called |
| B | `12_create_new_batch.py`, run trigger | `retraining_completed` | batch ingested (`weather_vNN.csv` archived, lineage updated), new version trained; promoted only if weighted F1 beats Production |
| C | run the trigger again with the same file | `no_new_data_retraining_skipped` | `already_ingested`, no retraining |
| D | batch with timestamps already in the dataset | pipeline fails at stage 01 | nothing appended |

For each scenario keep: the workflow run URL, `monitor_trigger_report.json`,
`data_load_report.json`, `dataset_lineage.json`, `lifecycle_report.json` and (if promoted)
`deployment_automation_report.json` plus `kubectl get pods`.

## Self-hosted runner

The Lab 13 workflows use `runs-on: [self-hosted, Windows, X64]` and a `PROJECT_DIR`
variable at the top of each file — set it to your project path. The runner must be online
(`C:\actions-runner\run.cmd`) with Docker Desktop and Kubernetes running.
`lab13_runner_test.yml` checks Python, Git, Docker, kubectl and the project path first.

## Known limitations

* The incoming batch is generated from the last month of real data with shifted
  timestamps, not from a live weather feed.
* Drift datasets are simulated.
* An interactive runner only works while its window is open and the PC is awake.
* Random row-level splitting is optimistic for this dataset (24 hourly rows share a label).
