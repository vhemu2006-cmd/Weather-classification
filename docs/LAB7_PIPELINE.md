# Lab 7 — ML Pipeline Engineering

Lab 7 turns the Lab 3–6 scripts into one ordered, restartable pipeline driven by `config.yaml`.

```text
01 Load Data                    src/mlops_pipeline/01_load_data.py
02 Raw Data Validation          src/mlops_pipeline/02_validate_data.py
03 Preprocessing                src/mlops_pipeline/03_preprocessing.py
04 Processed Output Validation  src/mlops_pipeline/04_validate_outputs.py
05 Training + MLflow + Registry src/mlops_pipeline/05_train_registry.py
06 Model Evaluation             src/mlops_pipeline/06_model_evaluation.py
08 Automated Model Lifecycle    src/mlops_pipeline/08_automate_lifecycle.py
09 Production Inference         src/mlops_pipeline/09_inference.py
```

Lab 8 inserts `07_quality_gate.py` between evaluation and lifecycle, and Lab 11/13 add the
promotion-gated `deploy_if_promoted.py`.

```bash
python src/mlops_pipeline/weather_prediction_pipeline.py --force
```

Each stage is a separate process; a non-zero exit code halts the pipeline and is recorded
in `logs/weather_prediction_pipeline_run_log.json`.

## Weather-specific choices

* **Fixed test set.** The first run stores the test-row timestamps in
  `data/processed/test_row_keys.csv`. Later runs (after new batches) keep exactly these test
  rows, so every model version is scored on the same data.
* **Deduplication.** 24 exact duplicate rows are removed so the timestamp key is unique.
* **Label encoder.** Fitted on all labels, saved to `models/label_encoder.pkl` and logged
  with the model in MLflow (`preprocessing_pipeline/label_encoder.pkl`).
* **Metrics.** Accuracy, weighted precision/recall/F1 and macro F1. Weighted F1 drives
  the lifecycle decision.

The registry uses the course stages `None → Staging → Production / Archived` under
`Weather_Daily_Summary_Production_Model`.
