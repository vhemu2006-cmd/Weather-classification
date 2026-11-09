"""Lab 12 - API health from the inference log: success rate, latency, throughput."""
import pandas as pd

from monitoring_common import load_inference_logs


def main():
    df = load_inference_logs()
    total = len(df)
    success = int((df["status"] == "success").sum())
    failure = int((df["status"] == "failure").sum())

    print("\n" + "=" * 60)
    print("API PERFORMANCE MONITORING")
    print("=" * 60)
    print(f"\nTotal requests : {total}")
    print(f"Successful     : {success}")
    print(f"Failed         : {failure}")
    print(f"Success rate   : {success / total * 100:.2f}%")
    print(f"Failure rate   : {failure / total * 100:.2f}%")

    latency = df["latency_ms"].dropna()
    if not latency.empty:
        print("\nLatency:")
        print(f"Average        : {latency.mean():.3f} ms")
        print(f"Median         : {latency.median():.3f} ms")
        print(f"P95            : {latency.quantile(0.95):.3f} ms")
        print(f"Minimum        : {latency.min():.3f} ms")
        print(f"Maximum        : {latency.max():.3f} ms")

    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    elapsed = (timestamps.max() - timestamps.min()).total_seconds()
    if elapsed > 0:
        print(f"\nThroughput     : {total / elapsed:.3f} requests/sec")
    else:
        print("\nThroughput     : Cannot calculate from one timestamp")

    print("\nStatus distribution:")
    print(df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
