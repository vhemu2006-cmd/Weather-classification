"""Lab 13 - simulate a genuinely new incoming batch: data/raw/new_weather.csv.

weatherHistory.csv ends on 2016-12-31, so a "new" batch is made by copying the
most recent N days of observations and moving their timestamps forward so the
batch starts right after the latest timestamp in the current dataset. Every run
therefore produces new, non-overlapping timestamps (and a new file hash), which
01_load_data.py will accept and append.

    python src/mlops_pipeline/12_create_new_batch.py --days 30
"""
import argparse

import pandas as pd

from weather_common import load_config, project_path


def format_timestamp(ts):
    """Match the dataset format: '2016-12-31 23:00:00.000 +0000'."""
    return ts.strftime("%Y-%m-%d %H:%M:%S.000 +0000")


def create_new_batch(days):
    config = load_config()
    raw_path = project_path(config["data"]["raw_path"])
    output_path = project_path(config["data"]["new_data_path"])
    date_col = config["data"]["date_column"]

    current = pd.read_csv(raw_path).drop_duplicates(subset=[date_col])
    timestamps = pd.to_datetime(current[date_col], utc=True)
    latest = timestamps.max()

    window_start = latest - pd.Timedelta(days=days)
    batch = current.loc[timestamps > window_start].copy()
    batch_ts = timestamps.loc[batch.index]

    # Shift by whole days so each hour keeps its time of day (and its Daily Summary
    # still describes a coherent day). Every new timestamp is later than `latest`.
    shift = pd.Timedelta(days=days)
    batch[date_col] = (batch_ts + shift).map(format_timestamp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(output_path, index=False)

    new_ts = pd.to_datetime(batch[date_col], utc=True)
    print("=" * 70)
    print("NEW INCOMING BATCH CREATED")
    print("=" * 70)
    print(f"Source dataset    : {raw_path} (latest {latest})")
    print(f"Output            : {output_path}")
    print(f"Rows              : {len(batch)}")
    print(f"New time range    : {new_ts.min()} -> {new_ts.max()}")
    print("Next step         : python src/mlops_pipeline/weather_prediction_pipeline.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30, help="Number of most recent days to copy (default 30).")
    create_new_batch(parser.parse_args().days)
