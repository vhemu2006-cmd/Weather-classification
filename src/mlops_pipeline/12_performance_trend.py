"""Lab 12 - performance trend across chronological windows of the test set.

The test rows are sorted by timestamp and split into equal windows (oldest to
newest), so a downward trend means the model ages badly over the 2006-2016 span.
"""
import numpy as np
import pandas as pd

from monitoring_common import evaluate_with, load_production_loader, load_test_set

NUMBER_OF_WINDOWS = 4


def main():
    X_test, y_test = load_test_set()
    order = np.argsort(pd.to_datetime(X_test["Formatted Date"], utc=True).to_numpy())
    X_sorted = X_test.iloc[order].reset_index(drop=True)
    y_sorted = np.asarray(y_test)[order]

    loader = load_production_loader()
    bundle = (loader.model, loader.preprocessor, loader.label_encoder)

    results = []
    for index, positions in enumerate(np.array_split(np.arange(len(X_sorted)), NUMBER_OF_WINDOWS), start=1):
        X_window = X_sorted.iloc[positions]
        dates = pd.to_datetime(X_window["Formatted Date"], utc=True)
        metrics = evaluate_with(*bundle, X_window, y_sorted[positions])
        results.append({
            "window": f"Window {index}",
            "from": dates.min().date(),
            "to": dates.max().date(),
            "samples": len(X_window),
            **metrics,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("MODEL PERFORMANCE TREND ANALYSIS (chronological windows)")
    print("=" * 90)
    print(f"\nModel name    : {loader.model_name}")
    print(f"Model version : {loader.model_version}")
    print(f"Run ID        : {loader.run_id}\n")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))

    change = results_df["f1_score"].iloc[-1] - results_df["f1_score"].iloc[0]
    print(f"\nWeighted F1 change from first to last window: {change:+.4f}")
    if change < 0:
        print("Observation: Weighted F1 decreased over time.")
    elif change > 0:
        print("Observation: Weighted F1 increased over time.")
    else:
        print("Observation: Weighted F1 remained unchanged over time.")


if __name__ == "__main__":
    main()
