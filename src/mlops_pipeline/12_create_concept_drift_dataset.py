"""Lab 12 - attach labels to the evolved data and change the feature -> label relationship.

In the evolved environment:
* very humid hours are more often summarised as fog for the whole day;
* very windy hours are more often summarised as breezy days.
"""
import numpy as np
import pandas as pd

from monitoring_common import EVOLVED_LABELED_PATH, EVOLVED_PATH, TARGET, Y_TEST, labels_from_ids

RANDOM_SEED = 42
HUMID_LABEL = "Foggy throughout the day."
WINDY_LABEL = "Mostly cloudy throughout the day and breezy in the afternoon."


def apply_concept_drift(X, y, rng):
    """Return relabelled y. Shared with 12_create_updated_dataset.py."""
    y_evolved = np.asarray(y, dtype=object).copy()

    humid = np.where(X["Humidity"].to_numpy() >= X["Humidity"].quantile(0.75))[0]
    chosen = rng.choice(humid, size=int(len(humid) * 0.25), replace=False)
    y_evolved[chosen] = HUMID_LABEL

    windy = np.where(X["Wind Speed (km/h)"].to_numpy() >= X["Wind Speed (km/h)"].quantile(0.75))[0]
    chosen = rng.choice(windy, size=int(len(windy) * 0.15), replace=False)
    y_evolved[chosen] = WINDY_LABEL

    return y_evolved


def create_concept_drift_dataset():
    rng = np.random.default_rng(RANDOM_SEED)
    X = pd.read_csv(EVOLVED_PATH)
    y = labels_from_ids(np.load(Y_TEST))

    if len(X) != len(y):
        raise ValueError(f"Feature/label mismatch: X={len(X)}, y={len(y)}")

    y_evolved = apply_concept_drift(X, y, rng)
    result = X.copy()
    result[TARGET] = y_evolved

    EVOLVED_LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(EVOLVED_LABELED_PATH, index=False)

    print("=" * 70)
    print("CONCEPT DRIFT DATASET CREATED")
    print("=" * 70)
    print(f"Feature source : {EVOLVED_PATH}")
    print(f"Label source   : {Y_TEST}")
    print(f"Output         : {EVOLVED_LABELED_PATH}")
    print(f"Rows           : {len(result)}")
    print(f"Columns        : {len(result.columns)}")
    print()
    print("Top 5 baseline labels:")
    print(pd.Series(y).value_counts().head(5).to_string())
    print()
    print("Top 5 evolved labels:")
    print(pd.Series(y_evolved).value_counts().head(5).to_string())
    print()
    print(f"Changed labels : {(np.asarray(y, dtype=object) != y_evolved).sum()}")


if __name__ == "__main__":
    create_concept_drift_dataset()
