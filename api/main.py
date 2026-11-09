import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_loader import ModelLoader
from api.monitoring_logger import log_inference
from api.preprocessing import clean_prediction_data
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    WeatherInput,
)

SERVICE_NAME = "Weather Daily Summary API"
SERVICE_VERSION = "1.0.0"

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        model_loader.load()
    except Exception as exc:
        # Keep the service up so /health and /metadata can report the problem;
        # prediction endpoints return 503 until a valid bundle is present.
        print(f"[ERROR] Model bundle could not be loaded: {exc}")
    yield
    # shutdown
    print("[INFO] Weather Daily Summary API shutting down...")


app = FastAPI(
    title=SERVICE_NAME,
    description="Lab 10 - ML Model Serving and API Engineering (214-class Daily Summary classifier)",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


def _log_failure(data_dict, start_time, error, request_id):
    latency_ms = None if start_time is None else round((time.perf_counter() - start_time) * 1000, 3)
    log_inference(
        input_data=data_dict,
        latency_ms=latency_ms,
        status="failure",
        error=error,
        model_name=model_loader.model_name,
        model_version=model_loader.model_version,
        run_id=model_loader.run_id,
        request_id=request_id,
    )


def _run_model(data_dicts):
    df = clean_prediction_data(pd.DataFrame(data_dicts))
    X = model_loader.preprocessor.transform(df)
    return model_loader.predict(X)


def _format(class_id, label, probability, top3):
    return {
        "prediction": int(class_id),
        "daily_summary": str(label),
        "probability": float(probability),
        "top_3": [{"daily_summary": name, "probability": p} for name, p in top3],
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        locations = [str(location) for location in error["loc"] if location != "body"]
        errors.append({"field": ".".join(locations), "message": error["msg"]})

    _log_failure({}, None, "Request validation failed", uuid.uuid4().hex[:12])

    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": "Request validation failed", "details": errors},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": "Prediction service unavailable",
            "details": [{"field": "prediction", "message": str(exc.detail)}],
        },
    )


@app.get("/")
def root():
    return {"message": f"{SERVICE_NAME} is running", "lab": "Lab 10", "status": "success"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "model_name": model_loader.model_name,
        "model_version": model_loader.model_version,
        "run_id": model_loader.run_id,
        "model_uri": model_loader.model_uri,
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
        "label_encoder_loaded": model_loader.label_encoder is not None,
        "n_classes": model_loader.n_classes,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: WeatherInput):
    request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()
    data_dict = data.model_dump()

    if not model_loader.is_ready():
        _log_failure(data_dict, start_time, "Model bundle is not available", request_id)
        raise HTTPException(status_code=503, detail="Model, preprocessor or label encoder is not available")

    try:
        class_ids, labels, probabilities, top3 = _run_model([data_dict])
        result = _format(class_ids[0], labels[0], probabilities[0], top3[0])
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        log_inference(
            input_data=data_dict,
            prediction=result["prediction"],
            daily_summary=result["daily_summary"],
            probability=result["probability"],
            latency_ms=latency_ms,
            status="success",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )
        return result

    except Exception as exc:
        _log_failure(data_dict, start_time, str(exc), request_id)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchPredictionRequest):
    start_time = time.perf_counter()
    data_dicts = [observation.model_dump() for observation in data.observations]

    if not model_loader.is_ready():
        batch_id = uuid.uuid4().hex[:12]
        for data_dict in data_dicts:
            _log_failure(data_dict, start_time, "Model bundle is not available", batch_id)
        raise HTTPException(status_code=503, detail="Model, preprocessor or label encoder is not available")

    try:
        class_ids, labels, probabilities, top3 = _run_model(data_dicts)
        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        results = []
        for data_dict, class_id, label, probability, top in zip(data_dicts, class_ids, labels, probabilities, top3):
            result = _format(class_id, label, probability, top)
            log_inference(
                input_data=data_dict,
                prediction=result["prediction"],
                daily_summary=result["daily_summary"],
                probability=result["probability"],
                latency_ms=total_latency_ms,
                status="success",
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=uuid.uuid4().hex[:12],
            )
            results.append(result)

        return {"count": len(results), "predictions": results}

    except Exception as exc:
        for data_dict in data_dicts:
            _log_failure(data_dict, start_time, str(exc), uuid.uuid4().hex[:12])
        raise HTTPException(status_code=500, detail="An unexpected error occurred during batch prediction")
