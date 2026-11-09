# Lab 10 — ML Model Serving and API Engineering

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service banner |
| GET | `/health` | Liveness (used by Kubernetes probes) |
| GET | `/metadata` | Model name/version/run id, loaded flags, number of classes |
| POST | `/predict` | One observation |
| POST | `/predict/batch` | `{"observations": [...]}` (at least one) |

Interactive docs: `http://127.0.0.1:8000/docs`.

## Input schema (`api/schemas.py`)

| Field | Rule |
|---|---|
| `formatted_date` | any parsable timestamp with offset, e.g. `2016-09-09 14:00:00.000 +0200` |
| `summary` | one of the 27 hourly summaries in the dataset |
| `precip_type` | `rain`, `snow` or `null` (optional) |
| `temperature_c` | −50…60 |
| `apparent_temperature_c` | −60…60 |
| `humidity` | 0…1 |
| `wind_speed_kmh` | 0…150 |
| `wind_bearing_degrees` | 0…360 |
| `visibility_km`, `pressure_millibars` | ≥ 0 |

Invalid requests return 422 with `{"status": "error", "message": "Request validation failed", "details": [{"field", "message"}]}`.
A missing model bundle returns 503.

## Response

```json
{
  "prediction": 197,
  "daily_summary": "Partly cloudy throughout the day.",
  "probability": 0.27,
  "top_3": [
    {"daily_summary": "Partly cloudy throughout the day.", "probability": 0.27},
    {"daily_summary": "Partly cloudy starting in the morning.", "probability": 0.20},
    {"daily_summary": "...", "probability": 0.08}
  ]
}
```

With 214 classes the top-1 probability is usually low (≈0.3 on average); `top_3` shows the
close alternatives.

## Training/serving consistency

`api/preprocessing.py` mirrors `weather_common.clean_dataframe`: drop `Loud Cover`/target,
convert the timestamp to UTC time features, map `null` precipitation to the `unknown`
category. In the sandbox check both paths produced identical feature matrices on 2,000
test rows.

## Tests

```bash
pytest tests/test_api.py -v          # Stage 9: 27 functional/validation tests
pytest tests/test_performance.py -v -s   # Stage 11: latency, consistency, reliability, batch, throughput
```

Every request (success or failure) is appended to `logs/inference/inference_log.jsonl`
for the Lab 12 monitors.
