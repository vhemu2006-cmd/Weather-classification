# Lab 9 — Failure Handling and Recovery

```bash
python src/mlops_pipeline/weather_prediction_pipeline.py --force --skip-deploy   # build normal state first
python src/mlops_pipeline/10_failure_injection.py
```

Every scenario corrupts one input, runs the stage that must catch it, checks the exit code
and **always restores** the original file (`try/finally`). The whole set repeats
`lab9.repetitions` times (default 3) to show the behaviour is deterministic.

| # | Scenario | Injection | Stage run | Expected |
|---|---|---|---|---|
| 1 | Missing data | `weatherHistory.csv` moved away | 02 validate | HALT |
| 2 | Invalid datatype | `Temperature (C)` row 0 = `"INVALID_NUMBER"` | 02 validate | HALT |
| 3 | Out-of-range values | `Humidity` = 1.75, `Wind Bearing` = 450 | 02 validate | HALT |
| 4 | Schema mismatch | `Humidity` column dropped | 02 validate | HALT |
| 5 | Preprocessing inconsistency | NaN written into `X_train_final.npy` | 04 validate outputs | HALT |
| 6 | Missing model artifact | `models/weather_model.pkl` deleted | 06 evaluate, then recover from `models:/…/Production`, 06 again | RECOVER |

Result: `reports/failure_report.json` with `overall_status` PASSED/FAILED, per-repetition
results and, for scenario 6, `failure_detected` and `recovery_succeeded`.

The workflow `.github/workflows/lab9_failure_tests.yml` runs the same steps on demand and
uploads the reports.
