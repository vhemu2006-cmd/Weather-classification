"""Lab 12 - combine data, prediction and concept drift reports into one summary."""
import pandas as pd

from monitoring_common import DRIFT_DIR

DATA_DRIFT_PATH = DRIFT_DIR / "data_drift_report.csv"
PREDICTION_DRIFT_PATH = DRIFT_DIR / "prediction_drift_report.csv"
CONCEPT_DRIFT_PATH = DRIFT_DIR / "concept_drift_report.csv"
OUTPUT_PATH = DRIFT_DIR / "drift_analysis_report.csv"

# A metric only counts as degraded when it drops by more than this amount,
# so tiny random fluctuations do not trigger retraining.
DEGRADATION_TOLERANCE = 0.01


def main():
    data_drift = pd.read_csv(DATA_DRIFT_PATH)
    prediction_drift = pd.read_csv(PREDICTION_DRIFT_PATH).iloc[0]
    concept_drift = pd.read_csv(CONCEPT_DRIFT_PATH)

    drifted = data_drift[data_drift["status"] == "DRIFT"]
    degraded = concept_drift[concept_drift["change"] < -DEGRADATION_TOLERANCE]

    summary = pd.DataFrame([
        {
            "category": "Data Drift",
            "metric": "Features with drift",
            "value": len(drifted),
            "status": "DRIFT" if len(drifted) > 0 else "OK",
            "details": ", ".join(drifted["feature"].tolist()),
        },
        {
            "category": "Prediction Drift",
            "metric": "Predicted-class distribution",
            "value": prediction_drift["prediction_distribution_score"],
            "status": prediction_drift["prediction_distribution_status"],
            "details": f"{prediction_drift['prediction_distribution_metric']} divergence",
        },
        {
            "category": "Prediction Drift",
            "metric": "Confidence distribution",
            "value": prediction_drift["probability_distribution_score"],
            "status": prediction_drift["probability_distribution_status"],
            "details": f"{prediction_drift['probability_distribution_metric']} statistic",
        },
        {
            "category": "Concept Drift",
            "metric": "Degraded performance metrics",
            "value": len(degraded),
            "status": "DRIFT" if len(degraded) > 0 else "OK",
            "details": ", ".join(degraded["metric"].tolist()),
        },
    ])

    print("=" * 70)
    print("COMBINED DRIFT ANALYSIS")
    print("=" * 70)
    print("\n1. DATA DRIFT")
    print("-" * 70)
    print(f"Features analyzed : {len(data_drift)}")
    print(f"Features drifted  : {len(drifted)}")
    print(f"Drifted features  : {', '.join(drifted['feature']) or 'None'}")
    print("\n2. PREDICTION DRIFT")
    print("-" * 70)
    print(f"Predicted classes  : {prediction_drift['prediction_distribution_status']} "
          f"(JS={prediction_drift['prediction_distribution_score']:.6f})")
    print(f"Confidence         : {prediction_drift['probability_distribution_status']} "
          f"(KS={prediction_drift['probability_distribution_score']:.6f})")
    print("\n3. CONCEPT DRIFT")
    print("-" * 70)
    print(f"Metrics analyzed : {len(concept_drift)}")
    print(f"Metrics degraded : {len(degraded)} (drop > {DEGRADATION_TOLERANCE})")
    print(f"Degraded metrics : {', '.join(degraded['metric']) or 'None'}")
    print("\n" + "=" * 70)
    if (summary["status"] == "DRIFT").any():
        print("OVERALL STATUS : DRIFT / MODEL BEHAVIOR CHANGE OBSERVED")
    else:
        print("OVERALL STATUS : NO SIGNIFICANT DRIFT OBSERVED")
    print("=" * 70)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCombined report saved : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
