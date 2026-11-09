"""Lab 5 - inject realistic faults and show that schema validation catches them.

Run from the project root:  python -m src.simulate_data_corruption
"""
import pandas as pd

from src.config import RAW
from src.validate_data import validate_schema


def simulate_corrupted_data(df):
    """Simulates real-world data corruption scenarios across multiple columns."""
    print("[INFO] Injecting simulated faults into data...")
    corrupted_df = df.copy()

    # 1. Physically impossible value (humidity is a 0-1 fraction)
    corrupted_df.loc[0, "Humidity"] = 1.8

    # 2. Out-of-range wind bearing
    corrupted_df.loc[1, "Wind Bearing (degrees)"] = 400.0

    # 3. Invalid categorical value
    corrupted_df.loc[2, "Precip Type"] = "hail"

    # 4. Datatype mismatch (string in a float column)
    corrupted_df["Temperature (C)"] = corrupted_df["Temperature (C)"].astype(object)
    corrupted_df.loc[3, "Temperature (C)"] = "Twenty"

    # 5. Schema inconsistency (unexpected extra column)
    corrupted_df["unexpected_column"] = 123

    # 6. Missing required attribute (dropping a column)
    corrupted_df = corrupted_df.drop(columns=["Pressure (millibars)"])

    return corrupted_df


if __name__ == "__main__":
    try:
        raw_df = pd.read_csv(RAW)
        corrupted_df = simulate_corrupted_data(raw_df)
        print("\n--- Running Validation on Corrupted Data ---")
        validate_schema(corrupted_df, output_report_name="corrupted_simulation_errors.csv")
    except FileNotFoundError:
        print(f"[ERROR] Could not find data file at {RAW}.")
