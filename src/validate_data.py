"""Lab 5 - schema validation of the raw weather data (pandera)."""
import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

from src.config import ARTIFACTS_DIR, RAW


def get_weather_schema():
    """Strict schema specification for weatherHistory.csv."""
    return pa.DataFrameSchema({
        "Formatted Date": Column(pa.String),
        "Summary": Column(pa.String),
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
    }, strict=True)


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
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        report_path = ARTIFACTS_DIR / output_report_name
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    if RAW.exists():
        ok = validate_schema(pd.read_csv(RAW), output_report_name="baseline_validation.csv")
        if not ok:
            raise SystemExit(1)
    else:
        print(f"[ERROR] Target file not found at: {RAW}")
        raise SystemExit(1)
