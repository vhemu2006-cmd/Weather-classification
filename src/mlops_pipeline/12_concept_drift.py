"""Lab 12 - concept drift: compare Production performance on baseline vs evolved labelled data."""
import pandas as pd

from monitoring_common import (DRIFT_DIR, EVOLVED_LABELED_PATH, TARGET, evaluate_with,
                               load_production_loader, load_test_set)

REPORT_PATH = DRIFT_DIR / "concept_drift_report.csv"


def main():
    print("=" * 70)
    print("CONCEPT DRIFT ANALYSIS")
    print("=" * 70)

    baseline_X, baseline_y = load_test_set()
    evolved = pd.read_csv(EVOLVED_LABELED_PATH)
    evolved_X = evolved.drop(columns=[TARGET])
    evolved_y = evolved[TARGET].to_numpy()

    loader = load_production_loader()
    print(f"\nModel name : {loader.model_name}")
    print(f"Version    : {loader.model_version}")
    print(f"Run ID     : {loader.run_id}")

    bundle = (loader.model, loader.preprocessor, loader.label_encoder)
    baseline_metrics = evaluate_with(*bundle, baseline_X, baseline_y)
    evolved_metrics = evaluate_with(*bundle, evolved_X, evolved_y)

    for title, metrics in (("BASELINE", baseline_metrics), ("EVOLVED", evolved_metrics)):
        print(f"\n{title} PERFORMANCE")
        print("-" * 60)
        for metric, value in metrics.items():
            print(f"{metric.upper():10s}: {value:.4f}")

    report = pd.DataFrame([
        {
            "metric": metric,
            "baseline": baseline_metrics[metric],
            "evolved": evolved_metrics[metric],
            "change": evolved_metrics[metric] - baseline_metrics[metric],
        }
        for metric in baseline_metrics
    ])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 70)
    print("PERFORMANCE CHANGE")
    print("=" * 70)
    print(report.to_string(index=False))
    print(f"\nReport saved : {REPORT_PATH}")


if __name__ == "__main__":
    main()
