import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_PATH = Path("logs/inference/inference_log.jsonl")
_LOCK = threading.Lock()


def log_inference(
    input_data: dict[str, Any],
    prediction: int | None = None,
    daily_summary: str | None = None,
    probability: float | None = None,
    latency_ms: float | None = None,
    status: str = "success",
    error: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
) -> str:
    """Append one inference event as a JSON line (used by the Lab 12 monitors)."""
    request_id = request_id or uuid.uuid4().hex[:12]

    event = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": dict(input_data),
        "prediction": prediction,
        "daily_summary": daily_summary,
        "probability": probability,
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    return request_id
