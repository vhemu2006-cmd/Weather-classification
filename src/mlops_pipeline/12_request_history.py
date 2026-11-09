"""Lab 12 - request/response history: recent requests, failures and one request in detail.

    python src/mlops_pipeline/12_request_history.py [request_id]
"""
import sys

import pandas as pd

from monitoring_common import load_inference_logs


def main():
    df = load_inference_logs()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print("\n" + "=" * 60)
    print("REQUEST-RESPONSE HISTORY")
    print("=" * 60)
    print(f"\nTotal records : {len(df)}")
    print("\nRecent requests:")
    columns = ["request_id", "timestamp", "daily_summary", "probability", "latency_ms", "status", "model_version"]
    print(df[columns].tail(10).to_string(index=False))

    failures = df[df["status"] == "failure"]
    print("\n" + "=" * 60)
    print("FAILED REQUESTS")
    print("=" * 60)
    if failures.empty:
        print("\nNo failed requests.")
    else:
        print(f"\nFailed requests: {len(failures)}")
        print(failures[["request_id", "timestamp", "status", "error", "model_version"]].to_string(index=False))

    request_id = sys.argv[1] if len(sys.argv) > 1 else df["request_id"].iloc[-1]
    match = df[df["request_id"] == request_id]
    print("\n" + "=" * 60)
    print(f"REQUEST DETAILS: {request_id}")
    print("=" * 60)
    if match.empty:
        print("Request not found.")
    else:
        for key, value in match.iloc[0].items():
            print(f"{key:14}: {value}")


if __name__ == "__main__":
    main()
