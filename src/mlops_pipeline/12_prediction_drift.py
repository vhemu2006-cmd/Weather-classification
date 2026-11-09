"""Lab 12 - prediction drift for a multiclass model.

* Predicted-class distribution: Jensen-Shannon divergence between the classes
  predicted on the baseline and the evolved data (KS is meaningless for
  214 unordered class ids).
* Confidence distribution: KS test on the max class probability.
"""
import pandas as pd
from scipy.stats import ks_2samp

from monitoring_common import (DRIFT_DIR, EVOLVED_PATH, X_TEST_RAW, js_divergence,
                               load_production_loader, predict_loader)

REPORT_PATH = DRIFT_DIR / "prediction_drift_report.csv"
JS_THRESHOLD = 0.10
KS_THRESHOLD = 0.10


def summarize_predictions(name, labels, confidence):
    counts = pd.Series(labels).value_counts()
    print(f"\n{name} PREDICTIONS")
    print("-" * 60)
    print(f"Total predictions   : {len(labels)}")
    print(f"Distinct classes    : {counts.size}")
    print("Top 5 predicted classes:")
    for label, count in counts.head(5).items():
        print(f"  {count:6d} ({count / len(labels) * 100:5.2f}%)  {label}")
    print(f"Mean confidence     : {confidence.mean():.6f}")
    print(f"Median confidence   : {pd.Series(confidence).median():.6f}")
    print(f"Std confidence      : {confidence.std():.6f}")


def main():
    print("=" * 70)
    print("PREDICTION DRIFT ANALYSIS")
    print("=" * 70)

    baseline = pd.read_csv(X_TEST_RAW)
    evolved = pd.read_csv(EVOLVED_PATH)
    if baseline.shape != evolved.shape:
        raise ValueError(f"Dataset shape mismatch: baseline={baseline.shape}, evolved={evolved.shape}")

    loader = load_production_loader()
    print(f"\nModel name : {loader.model_name}")
    print(f"Version    : {loader.model_version}")
    print(f"Run ID     : {loader.run_id}")

    base_labels, base_conf = predict_loader(loader, baseline)
    evo_labels, evo_conf = predict_loader(loader, evolved)
    summarize_predictions("BASELINE", base_labels, base_conf)
    summarize_predictions("EVOLVED", evo_labels, evo_conf)

    class_js = js_divergence(base_labels, evo_labels)
    conf_ks, conf_p = ks_2samp(base_conf, evo_conf)

    results = {
        "prediction_distribution_metric": "JS",
        "prediction_distribution_score": float(class_js),
        "prediction_distribution_threshold": JS_THRESHOLD,
        "prediction_distribution_status": "DRIFT" if class_js >= JS_THRESHOLD else "OK",
        "probability_distribution_metric": "KS",
        "probability_distribution_score": float(conf_ks),
        "probability_distribution_p_value": float(conf_p),
        "probability_distribution_threshold": KS_THRESHOLD,
        "probability_distribution_status": "DRIFT" if conf_ks >= KS_THRESHOLD else "OK",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 70)
    print("PREDICTION DRIFT RESULTS")
    print("=" * 70)
    print(f"Predicted-class JS divergence : {class_js:.6f} -> {results['prediction_distribution_status']}")
    print(f"Confidence KS statistic       : {conf_ks:.6f} (p={conf_p:.6f}) -> "
          f"{results['probability_distribution_status']}")
    print(f"\nThresholds : JS >= {JS_THRESHOLD}, KS >= {KS_THRESHOLD}")
    print(f"Report     : {REPORT_PATH}")


if __name__ == "__main__":
    main()
