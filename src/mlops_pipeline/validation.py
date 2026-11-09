"""Pandera schema for weatherHistory.csv (shared by stages 01 and 02)."""
import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

from weather_common import project_path

SUMMARY_VALUES = [
    "Breezy", "Breezy and Dry", "Breezy and Foggy", "Breezy and Mostly Cloudy",
    "Breezy and Overcast", "Breezy and Partly Cloudy", "Clear",
    "Dangerously Windy and Partly Cloudy", "Drizzle", "Dry", "Dry and Mostly Cloudy",
    "Dry and Partly Cloudy", "Foggy", "Humid and Mostly Cloudy", "Humid and Overcast",
    "Humid and Partly Cloudy", "Light Rain", "Mostly Cloudy", "Overcast", "Partly Cloudy",
    "Rain", "Windy", "Windy and Dry", "Windy and Foggy", "Windy and Mostly Cloudy",
    "Windy and Overcast", "Windy and Partly Cloudy",
]


def validate_timestamps(series):
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.notna()


def get_weather_schema():
    return pa.DataFrameSchema(
        {
            "Formatted Date": Column(pa.String, Check(validate_timestamps)),
            "Summary": Column(pa.String, Check.isin(SUMMARY_VALUES)),
            "Precip Type": Column(pa.String, Check.isin(["rain", "snow"]), nullable=True),
            "Temperature (C)": Column(pa.Float, Check.in_range(-50, 60)),
            "Apparent Temperature (C)": Column(pa.Float, Check.in_range(-60, 60)),
            "Humidity": Column(pa.Float, Check.in_range(0.0, 1.0)),
            "Wind Speed (km/h)": Column(pa.Float, Check.in_range(0, 150)),
            "Wind Bearing (degrees)": Column(pa.Float, Check.in_range(0, 360)),
            "Visibility (km)": Column(pa.Float, Check.ge(0.0)),
            "Loud Cover": Column(pa.Float),
            "Pressure (millibars)": Column(pa.Float, Check.ge(0.0)),
            "Daily Summary": Column(pa.String),
        },
        strict=True,
    )


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_weather_schema()

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.head(50).to_string())

        artifacts_dir = project_path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        report_path = artifacts_dir / output_report_name
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False
