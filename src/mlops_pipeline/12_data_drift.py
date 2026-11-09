"""Lab 12 - data drift: KS test for numerical features, Jensen-Shannon for categorical."""
import pandas as pd
from scipy.stats import ks_2samp

from monitoring_common import DRIFT_DIR, EVOLVED_PATH, X_TEST_RAW, js_divergence

REPORT_PATH = DRIFT_DIR / "data_drift_report.csv"

NUMERICAL_COLUMNS = [
    "Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Visibility (km)",
    "Pressure (millibars)",
]
CATEGORICAL_COLUMNS = ["Summary", "Precip Type"]

KS_THRESHOLD = 0.10
JS_THRESHOLD = 0.10


def load_data():
    baseline = pd.read_csv(X_TEST_RAW)
    evolved = pd.read_csv(EVOLVED_PATH)
    if baseline.shape != evolved.shape:
        raise ValueError(f"Dataset shape mismatch: baseline={baseline.shape}, evolved={evolved.shape}")
    return baseline, evolved


def analyze_numerical_drift(baseline, evolved):
    results = []
    for column in NUMERICAL_COLUMNS:
        statistic, p_value = ks_2samp(baseline[column].dropna(), evolved[column].dropna())
        results.append({
            "feature": column,
            "type": "numerical",
            "metric": "KS",
            "score": round(float(statistic), 6),
            "p_value": round(float(p_value), 6),
            "threshold": KS_THRESHOLD,
            "status": "DRIFT" if statistic >= KS_THRESHOLD else "OK",
        })
    return results


def analyze_categorical_drift(baseline, evolved):
    results = []
    for column in CATEGORICAL_COLUMNS:
        # Missing Precip Type is a real category ('unknown') for this dataset.
        score = js_divergence(baseline[column].fillna("unknown"), evolved[column].fillna("unknown"))
        results.append({
            "feature": column,
            "type": "categorical",
            "metric": "JS",
            "score": round(score, 6),
            "p_value": None,
            "threshold": JS_THRESHOLD,
            "status": "DRIFT" if score >= JS_THRESHOLD else "OK",
        })
    return results


def main():
    baseline, evolved = load_data()
    print(f"Baseline shape : {baseline.shape}")
    print(f"Evolved shape  : {evolved.shape}")

    report = pd.DataFrame(analyze_numerical_drift(baseline, evolved) + analyze_categorical_drift(baseline, evolved))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("=" * 70)
    print("DATA DRIFT ANALYSIS")
    print("=" * 70)
    print(f"Baseline dataset : {X_TEST_RAW}")
    print(f"Evolved dataset  : {EVOLVED_PATH}\n")
    print(report.to_string(index=False))
    print()
    print("=" * 70)
    print(f"Features analyzed : {len(report)}")
    print(f"Drift detected    : {(report['status'] == 'DRIFT').sum()}")
    print(f"Report saved      : {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
