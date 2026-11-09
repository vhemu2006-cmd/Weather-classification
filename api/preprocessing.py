import pandas as pd

# API field name -> column name used in weatherHistory.csv and by the preprocessor.
FIELD_TO_COLUMN = {
    "formatted_date": "Formatted Date",
    "summary": "Summary",
    "precip_type": "Precip Type",
    "temperature_c": "Temperature (C)",
    "apparent_temperature_c": "Apparent Temperature (C)",
    "humidity": "Humidity",
    "wind_speed_kmh": "Wind Speed (km/h)",
    "wind_bearing_degrees": "Wind Bearing (degrees)",
    "visibility_km": "Visibility (km)",
    "pressure_millibars": "Pressure (millibars)",
}

DATE_COLUMN = "Formatted Date"
NON_FEATURE_COLUMNS = ["Daily Summary", "Loud Cover"]


def parse_timestamps(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, utc=True)
    except (ValueError, TypeError):
        # A batch may mix timestamp formats; parse each element on its own.
        return pd.to_datetime(values, utc=True, format="mixed")


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw observations (API field names or CSV column names) into the
    feature frame expected by the saved ColumnTransformer.

    This mirrors weather_common.clean_dataframe used during training:
    drop the target and the constant 'Loud Cover' column, and replace the
    timestamp with Year/Month/Day/Hour/DayOfWeek (UTC).
    """
    df = df.rename(columns=FIELD_TO_COLUMN).copy()

    df = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])

    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {DATE_COLUMN}")

    dt = parse_timestamps(df[DATE_COLUMN])
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Hour"] = dt.dt.hour
    df["DayOfWeek"] = dt.dt.dayofweek
    df = df.drop(columns=[DATE_COLUMN])

    # None -> NaN so the categorical imputer maps it to 'unknown' as in training.
    if "Precip Type" in df.columns:
        df["Precip Type"] = df["Precip Type"].astype(object).where(df["Precip Type"].notna(), float("nan"))

    return df
