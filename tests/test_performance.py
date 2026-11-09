"""Lab 10 Stage 11 - API performance tests. Run:  pytest tests/test_performance.py -v -s"""
import statistics
import time

from starlette.testclient import TestClient

from api.main import app

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


def create_client():
    return TestClient(app)


def timed_post(client, url, payload):
    start = time.perf_counter()
    response = client.post(url, json=payload)
    return response, (time.perf_counter() - start) * 1000


# 11.1 - Single request latency
def test_single_request_latency():
    with create_client() as client:
        response, latency_ms = timed_post(client, "/predict", VALID_OBSERVATION)
    assert response.status_code == 200
    print(f"\nSingle request latency: {latency_ms:.2f} ms")


# 11.2 - Repeated request latency
def test_repeated_request_latency():
    repetitions = 20
    with create_client() as client:
        latencies = []
        for _ in range(repetitions):
            response, latency_ms = timed_post(client, "/predict", VALID_OBSERVATION)
            assert response.status_code == 200
            latencies.append(latency_ms)

    print("\nRepeated request performance:")
    print(f"Requests: {repetitions}")
    print(f"Average latency: {statistics.mean(latencies):.2f} ms")
    print(f"Median latency: {statistics.median(latencies):.2f} ms")
    print(f"Minimum latency: {min(latencies):.2f} ms")
    print(f"Maximum latency: {max(latencies):.2f} ms")
    assert len(latencies) == repetitions


# 11.3 - Response consistency (Random Forest inference is deterministic)
def test_prediction_consistency():
    with create_client() as client:
        responses = [client.post("/predict", json=VALID_OBSERVATION).json() for _ in range(10)]

    labels = {r["daily_summary"] for r in responses}
    probabilities = {round(r["probability"], 12) for r in responses}
    print(f"\nUnique predictions: {labels}")
    assert len(labels) == 1
    assert len(probabilities) == 1


# 11.4 - Reliability / success rate
def test_api_reliability():
    total = 50
    with create_client() as client:
        successes = sum(client.post("/predict", json=VALID_OBSERVATION).status_code == 200 for _ in range(total))
    success_rate = successes / total * 100
    print(f"\nSuccess rate: {success_rate:.1f}% ({successes}/{total})")
    assert success_rate == 100.0


# 11.5 - Batch prediction performance
def test_batch_prediction_performance():
    with create_client() as client:
        for size in (1, 10, 50, 100):
            response, latency_ms = timed_post(client, "/predict/batch", {"observations": [VALID_OBSERVATION] * size})
            assert response.status_code == 200
            assert response.json()["count"] == size
            print(f"\nBatch size {size:>3}: {latency_ms:8.2f} ms total | {latency_ms / size:6.2f} ms per row")


# 11.6 - Throughput
def test_throughput():
    requests = 30
    with create_client() as client:
        start = time.perf_counter()
        for _ in range(requests):
            assert client.post("/predict", json=VALID_OBSERVATION).status_code == 200
        elapsed = time.perf_counter() - start
    throughput = requests / elapsed
    print(f"\nThroughput: {throughput:.2f} requests/sec over {requests} sequential requests")
    assert throughput > 0
