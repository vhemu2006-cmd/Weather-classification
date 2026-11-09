# Lab 12 — Monitoring, Drift Detection and Continuous Training

All scripts read the deployed model from `deployment/` (the same bundle the API serves).

## 1. Simulated drift

| Script | Output | What changes |
|---|---|---|
| `12_create_evolved_dataset.py` | `data/monitoring/drift/evolved_data.csv` | fixed test rows: +~2.5 °C, +0.06 humidity, wind ×1.25, visibility ×0.85, more Overcast/Foggy, more snow; pressure unchanged (control) |
| `12_create_concept_drift_dataset.py` | `evolved_labeled_data.csv` | 25 % of the most humid hours → "Foggy throughout the day.", 15 % of the windiest → "Mostly cloudy throughout the day and breezy in the afternoon." |

## 2. Drift detection

| Script | Method | Threshold |
|---|---|---|
| `12_data_drift.py` | KS (Temperature, Humidity, Wind Speed, Visibility, Pressure), Jensen–Shannon (Summary, Precip Type) | KS ≥ 0.10, JS ≥ 0.10 |
| `12_prediction_drift.py` | JS on predicted classes; KS on max-probability (confidence) | 0.10 / 0.10 |
| `12_concept_drift.py` | Production metrics on baseline vs evolved labelled data | — |
| `12_drift_analysis.py` | Combines all three; a metric is "degraded" only if it drops > 0.01 | — |

Sandbox reference run (baseline model): 5/7 features drifted (Pressure OK, Precip Type just
below threshold), predicted-class JS 0.069 (OK), confidence KS 0.230 (DRIFT), weighted F1
0.567 → 0.187 on evolved labels.

## 3. Live monitoring from API logs (`logs/inference/inference_log.jsonl`)

| Script | Shows |
|---|---|
| `12_prediction_monitoring.py` | top predicted classes, confidence stats, share of low-confidence (< 0.30) predictions |
| `12_api_performance_monitoring.py` | success/failure rate, latency mean/median/P95, throughput |
| `12_request_history.py [request_id]` | recent requests, failures, one request in detail |

Generate traffic first, e.g. `pytest tests/test_performance.py -s`.

## 4. Model performance

| Script | Shows |
|---|---|
| `12_performance_monitoring.py` | Production metrics on the fixed test set + quality gate |
| `12_performance_trend.py` | metrics in 4 chronological windows (2006 → 2016) |

## 5. Continuous training

```bash
python src/mlops_pipeline/12_create_updated_dataset.py   # 77k original + 5,000 post-drift rows
python src/mlops_pipeline/12_retrain_model.py            # V2 (use --no-mlflow to skip logging)
python src/mlops_pipeline/12_model_comparison.py         # V1 vs V2 on the same fixed test set
python src/mlops_pipeline/12_update_mlflow_registry.py   # register V2 -> Staging (with preprocessor + encoder)
python src/mlops_pipeline/08_automate_lifecycle.py        # promote V2 only if it beats Production
python src/mlops_pipeline/deploy_if_promoted.py           # redeploy if promoted
```

Sandbox reference: V1 weighted F1 0.567, V2 0.570 — V2 passes the gate and slightly improves.
