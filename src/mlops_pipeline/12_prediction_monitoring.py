"""Lab 12 - monitor the distribution of live API predictions from the inference log."""
from monitoring_common import load_inference_logs


def main():
    df = load_inference_logs()
    df = df[df["status"] == "success"]
    if df.empty:
        raise ValueError("No successful predictions in the inference log yet.")

    total = len(df)
    counts = df["daily_summary"].value_counts()

    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION MONITORING")
    print("=" * 60)
    print(f"\nSuccessful predictions : {total}")
    print(f"Distinct classes       : {counts.size}")

    print("\nTop 10 predicted Daily Summary classes:")
    for label, count in counts.head(10).items():
        print(f"  {count:6d} ({count / total * 100:5.2f}%)  {label}")

    print("\nPrediction confidence (max class probability):")
    print(df["probability"].describe().to_string())

    low_conf = (df["probability"] < 0.3).mean() * 100
    print(f"\nLow-confidence predictions (< 0.30): {low_conf:.2f}%")

    print("\nModel information:")
    print(f"Model name    : {df['model_name'].iloc[-1]}")
    print(f"Model version : {df['model_version'].iloc[-1]}")
    print(f"Run ID        : {df['run_id'].iloc[-1]}")


if __name__ == "__main__":
    main()
