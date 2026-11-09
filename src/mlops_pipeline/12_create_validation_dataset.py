
"""Lab 12 - Generate a new synthetic weather stress-test dataset.

Note: This reuses previously evaluated test records with new
transformations. It is a robustness test, not independent validation.
"""

import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE_X = ROOT / "data/processed/X_test_raw.csv"
SOURCE_Y = ROOT / "data/processed/y_test.npy"

OUTPUT_DIR = ROOT / "data/monitoring/validation"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_validation_data.csv"

RANDOM_SEED = 20261107


def main():

    print("=" * 70)
    print("SYNTHETIC VALIDATION DATASET GENERATION")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_SEED)

    X = pd.read_csv(SOURCE_X)
    y = np.load(SOURCE_Y, allow_pickle=True)

    if len(X) != len(y):
        raise ValueError("Feature and label row counts do not match.")

    df = X.copy()

    # New stress scenario: warmer, windier and reduced visibility.
    df["Temperature (C)"] += rng.normal(2.5, 1.5, len(df))

    df["Apparent Temperature (C)"] += rng.normal(
        2.0, 1.5, len(df)
    )

    df["Humidity"] = np.clip(
        df["Humidity"] + rng.normal(0.06, 0.04, len(df)),
        0.0, 1.0
    )

    df["Wind Speed (km/h)"] = np.clip(
        df["Wind Speed (km/h)"] * rng.uniform(
            1.10, 1.35, len(df)
        ),
        0.0, 150.0
    )

    df["Wind Bearing (degrees)"] = (
        df["Wind Bearing (degrees)"]
        + rng.normal(20, 15, len(df))
    ) % 360

    df["Visibility (km)"] = np.clip(
        df["Visibility (km)"] * rng.uniform(
            0.65, 0.90, len(df)
        ),
        0.0, 16.1
    )

    # Keep labels and categorical columns unchanged.
    # This isolates numerical input sensitivity.
    df["target"] = y

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nGeneration successful.")
    print("Records :", len(df))
    print("Features:", len(X.columns))
    print("Classes :", len(np.unique(y)))
    print("Seed    :", RANDOM_SEED)
    print("Output  :", OUTPUT_FILE)

    print("\nValidation feature summary:")
    print(df.select_dtypes(include="number").describe().round(3))

    print("\nNOTE: Synthetic stress test, not independent real-world validation.")


if __name__ == "__main__":
    main()
