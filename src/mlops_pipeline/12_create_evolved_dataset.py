"""Lab 12 - simulate a changed climate/sensor environment on the fixed test rows.

Numerical drift : warmer (+~2.5 C), more humid, windier, lower visibility.
Categorical drift: many more Overcast/Foggy hours and a much higher share of snow.
Pressure is left unchanged as a control feature (should report OK).
"""
import numpy as np
import pandas as pd

from weather_common import project_path

SOURCE_PATH = project_path("data/processed/X_test_raw.csv")
OUTPUT_PATH = project_path("data/monitoring/drift/evolved_data.csv")
RANDOM_SEED = 42

SUMMARY_SHIFT = {
    "Partly Cloudy": 0.12,
    "Mostly Cloudy": 0.20,
    "Overcast": 0.35,
    "Clear": 0.08,
    "Foggy": 0.25,
}


def apply_environment_drift(df, rng):
    """Shift weather features. Shared with 12_create_updated_dataset.py."""
    df = df.copy()
    n = len(df)

    delta = rng.normal(2.5, 1.0, n)
    df["Temperature (C)"] = (df["Temperature (C)"] + delta).clip(-50, 60)
    df["Apparent Temperature (C)"] = (df["Apparent Temperature (C)"] + delta).clip(-60, 60)
    df["Humidity"] = (df["Humidity"] + rng.normal(0.06, 0.03, n)).clip(0.0, 1.0).round(2)
    df["Wind Speed (km/h)"] = (df["Wind Speed (km/h)"] * rng.normal(1.25, 0.10, n)).clip(0, 150)
    df["Visibility (km)"] = (df["Visibility (km)"] * rng.normal(0.85, 0.05, n)).clip(0, 16.1)

    df["Summary"] = rng.choice(list(SUMMARY_SHIFT), size=n, p=list(SUMMARY_SHIFT.values()))
    df["Precip Type"] = rng.choice(["rain", "snow"], size=n, p=[0.60, 0.40])
    return df


def create_evolved_dataset():
    rng = np.random.default_rng(RANDOM_SEED)
    df = apply_environment_drift(pd.read_csv(SOURCE_PATH), rng)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("EVOLVED DATASET CREATED")
    print("=" * 60)
    print(f"\nSource      : {SOURCE_PATH}")
    print(f"Output      : {OUTPUT_PATH}")
    print(f"Rows        : {len(df)}")
    print(f"Columns     : {len(df.columns)}")
    print("\nModified features:")
    for name in ["Temperature (C)", "Apparent Temperature (C)", "Humidity",
                 "Wind Speed (km/h)", "Visibility (km)", "Summary", "Precip Type"]:
        print(f"- {name}")
    print("Unchanged control feature: Pressure (millibars)")


if __name__ == "__main__":
    create_evolved_dataset()
