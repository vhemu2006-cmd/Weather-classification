"""Lab 10 Stage 9 - API tests. Run from the project root:  pytest tests/test_api.py -v

Requires the deployment bundle (deployment/model.pkl, preprocessor.pkl,
label_encoder.pkl, model_metadata.json).
"""
import pytest
from starlette.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


VALID_OBSERVATION = {
    "formatted_date": "2016-09-09 14:00:00.000 +0200",
    "summary": "Partly Cloudy",
    "precip_type": "rain",
    "temperature_c": 24.5,
    "apparent_temperature_c": 24.5,
    "humidity": 0.45,
    "wind_speed_kmh": 11.3,
    "wind_bearing_degrees": 200.0,
    "visibility_km": 15.8,
    "pressure_millibars": 1014.2,
}

SECOND_OBSERVATION = {
    "formatted_date": "2015-01-15 06:00:00.000 +0100",
    "summary": "Foggy",
    "precip_type": "snow",
    "temperature_c": -3.2,
    "apparent_temperature_c": -6.8,
    "humidity": 0.96,
    "wind_speed_kmh": 4.1,
    "wind_bearing_degrees": 30.0,
    "visibility_km": 1.2,
    "pressure_millibars": 1028.6,
}


def assert_valid_prediction(prediction):
    assert isinstance(prediction["prediction"], int)
    assert isinstance(prediction["daily_summary"], str) and prediction["daily_summary"]
    assert 0.0 <= prediction["probability"] <= 1.0
    assert len(prediction["top_3"]) == 3
    assert prediction["top_3"][0]["daily_summary"] == prediction["daily_summary"]
    probabilities = [item["probability"] for item in prediction["top_3"]]
    assert probabilities == sorted(probabilities, reverse=True)


def assert_validation_error(response, expected_fields):
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    fields = [d["field"] for d in data["details"]]
    for field in expected_fields:
        assert field in fields
    return data


# 9.1 - Health
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# 9.2 - Metadata
def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Weather Daily Summary API"
    assert data["version"] == "1.0.0"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert data["label_encoder_loaded"] is True
    assert data["n_classes"] > 1
    assert data["model_name"] is not None


# 9.3 - Valid single prediction
def test_valid_prediction(client):
    response = client.post("/predict", json=VALID_OBSERVATION)
    assert response.status_code == 200
    assert_valid_prediction(response.json())


def test_null_precip_type_is_accepted(client):
    payload = {**VALID_OBSERVATION, "precip_type": None}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert_valid_prediction(response.json())


def test_missing_optional_precip_type(client):
    payload = dict(VALID_OBSERVATION)
    del payload["precip_type"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# 9.4 - Invalid single predictions
def test_invalid_summary(client):
    payload = {**VALID_OBSERVATION, "summary": "Sunny with meatballs"}
    data = assert_validation_error(client.post("/predict", json=payload), ["summary"])
    assert len(data["details"]) == 1


def test_invalid_precip_type(client):
    payload = {**VALID_OBSERVATION, "precip_type": "hail"}
    assert_validation_error(client.post("/predict", json=payload), ["precip_type"])


def test_humidity_out_of_range(client):
    payload = {**VALID_OBSERVATION, "humidity": 1.5}
    assert_validation_error(client.post("/predict", json=payload), ["humidity"])


def test_wind_bearing_out_of_range(client):
    payload = {**VALID_OBSERVATION, "wind_bearing_degrees": 400}
    assert_validation_error(client.post("/predict", json=payload), ["wind_bearing_degrees"])


def test_invalid_timestamp(client):
    payload = {**VALID_OBSERVATION, "formatted_date": "not-a-date"}
    assert_validation_error(client.post("/predict", json=payload), ["formatted_date"])


def test_missing_required_field(client):
    payload = dict(VALID_OBSERVATION)
    del payload["temperature_c"]
    data = assert_validation_error(client.post("/predict", json=payload), ["temperature_c"])
    assert len(data["details"]) == 1


# 9.5 - Valid batch
def test_valid_batch_prediction(client):
    response = client.post("/predict/batch", json={"observations": [VALID_OBSERVATION, SECOND_OBSERVATION]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    for prediction in data["predictions"]:
        assert_valid_prediction(prediction)


def test_batch_with_mixed_timestamp_formats(client):
    iso = {**SECOND_OBSERVATION, "formatted_date": "2015-01-15T06:00:00+01:00"}
    response = client.post("/predict/batch", json={"observations": [VALID_OBSERVATION, iso]})
    assert response.status_code == 200
    assert response.json()["count"] == 2


# 9.6 - Invalid batch
def test_invalid_batch_prediction(client):
    bad = {**SECOND_OBSERVATION, "summary": "ABC"}
    response = client.post("/predict/batch", json={"observations": [VALID_OBSERVATION, bad]})
    assert_validation_error(response, ["observations.1.summary"])


# 9.7 - Multiple batch validation errors
def test_multiple_batch_validation_errors(client):
    bad_one = {**VALID_OBSERVATION, "humidity": -0.2}
    bad_two = {**SECOND_OBSERVATION, "pressure_millibars": -5}
    response = client.post("/predict/batch", json={"observations": [bad_one, bad_two]})
    data = assert_validation_error(response, ["observations.0.humidity", "observations.1.pressure_millibars"])
    assert len(data["details"]) == 2


# 9.8 - Empty batch
def test_empty_batch(client):
    assert_validation_error(client.post("/predict/batch", json={"observations": []}), ["observations"])


# 9.9 - Invalid datatypes
def test_invalid_datatype(client):
    payload = {**VALID_OBSERVATION, "temperature_c": "warm"}
    assert_validation_error(client.post("/predict", json=payload), ["temperature_c"])


def test_multiple_invalid_datatypes(client):
    payload = {**VALID_OBSERVATION, "temperature_c": "warm", "wind_speed_kmh": "fast"}
    data = assert_validation_error(client.post("/predict", json=payload), ["temperature_c", "wind_speed_kmh"])
    assert len(data["details"]) == 2


def test_null_numeric_value(client):
    payload = {**VALID_OBSERVATION, "visibility_km": None}
    assert_validation_error(client.post("/predict", json=payload), ["visibility_km"])


# 9.10 - Extreme but valid values
def test_extreme_valid_values(client):
    payload = {
        **VALID_OBSERVATION,
        "temperature_c": 59.9,
        "apparent_temperature_c": -59.9,
        "humidity": 0.0,
        "wind_speed_kmh": 150.0,
        "wind_bearing_degrees": 360.0,
        "visibility_km": 0.0,
        "pressure_millibars": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert_valid_prediction(response.json())


def test_extreme_negative_values(client):
    payload = {**VALID_OBSERVATION, "temperature_c": -80, "wind_speed_kmh": -1}
    assert_validation_error(client.post("/predict", json=payload), ["temperature_c", "wind_speed_kmh"])


# Request-structure errors
def test_malformed_json(client):
    response = client.post(
        "/predict", content='{"formatted_date": "2016-01-01 00:00:00.000 +0100",',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_wrong_json_structure(client):
    response = client.post("/predict", json=[VALID_OBSERVATION])
    assert response.status_code == 422


def test_invalid_batch_structure(client):
    response = client.post("/predict/batch", json={"observations": VALID_OBSERVATION})
    assert_validation_error(response, ["observations"])


def test_null_batch_observations(client):
    assert_validation_error(client.post("/predict/batch", json={"observations": None}), ["observations"])


def test_batch_with_null_observation(client):
    response = client.post("/predict/batch", json={"observations": [VALID_OBSERVATION, None]})
    assert_validation_error(response, ["observations.1"])


def test_large_batch(client):
    response = client.post("/predict/batch", json={"observations": [VALID_OBSERVATION] * 200})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 200
    # Identical inputs must give identical predictions.
    assert len({p["daily_summary"] for p in data["predictions"]}) == 1
