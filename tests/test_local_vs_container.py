"""Lab 11.8 - compare the local API with the running container.

Start the container first:
    docker run -d -p 8000:8000 --name weather-api weather-summary-api:1.0
Then run from the project root:
    python -m tests.test_local_vs_container
"""
import json
from urllib.request import Request, urlopen

from fastapi.testclient import TestClient

from api.main import app

CONTAINER_URL = "http://localhost:8000/predict"

PAYLOAD = {
    "formatted_date": "2016-06-21 13:00:00.000 +0200",
    "summary": "Mostly Cloudy",
    "precip_type": "rain",
    "temperature_c": 21.7,
    "apparent_temperature_c": 21.7,
    "humidity": 0.62,
    "wind_speed_kmh": 9.6,
    "wind_bearing_degrees": 160.0,
    "visibility_km": 11.2,
    "pressure_millibars": 1012.4,
}


def call_container():
    request = Request(
        CONTAINER_URL,
        data=json.dumps(PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    print("=" * 60)
    print("LOCAL VS CONTAINER PREDICTION CONSISTENCY")
    print("=" * 60)

    with TestClient(app) as client:
        local = client.post("/predict", json=PAYLOAD).json()
    container = call_container()

    print("\nLOCAL RESPONSE:")
    print(json.dumps(local, indent=2))
    print("\nCONTAINER RESPONSE:")
    print(json.dumps(container, indent=2))

    prediction_match = local["daily_summary"] == container["daily_summary"]
    probability_match = abs(local["probability"] - container["probability"]) < 1e-10

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Local prediction      : {local['daily_summary']}")
    print(f"Container prediction  : {container['daily_summary']}")
    print(f"Local probability     : {local['probability']}")
    print(f"Container probability : {container['probability']}")
    print(f"\nPrediction match      : {prediction_match}")
    print(f"Probability match     : {probability_match}")

    if prediction_match and probability_match:
        print("\n[SUCCESS] Local and container predictions are consistent.")
    else:
        print("\n[FAILURE] Local and container predictions differ.")
        raise AssertionError("Prediction consistency check failed.")


if __name__ == "__main__":
    main()
