"""Stage 01 - Data loading with batch ingestion, hashing, archiving and lineage.

If data/raw/new_weather.csv exists it is validated, hashed and (if it was not
ingested before) appended to weatherHistory.csv. The previous dataset is
archived as data/raw/archive/weather_vNN.csv and the batch is recorded in
data/raw/archive/dataset_lineage.json.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil

import pandas as pd

from validation import validate_schema
from weather_common import load_config, project_path


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_lineage(lineage_path):
    if not lineage_path.exists():
        return []
    try:
        with lineage_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_lineage(lineage_path, lineage):
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with lineage_path.open("w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2)


def get_next_archive_version(archive_dir):
    versions = []
    for file_path in archive_dir.glob("weather_v*.csv"):
        try:
            versions.append(int(file_path.stem.replace("weather_v", "")))
        except ValueError:
            continue
    return max(versions, default=0) + 1


def validate_new_batch_identity(new_df, current_df, key_column):
    """A new batch must contain unique timestamps that are not already in the dataset."""
    if new_df.empty:
        raise ValueError("new_weather.csv is empty.")
    if current_df.empty:
        raise ValueError("Current weatherHistory.csv is empty.")
    if not new_df[key_column].is_unique:
        raise ValueError(f"new_weather.csv contains duplicate '{key_column}' values.")

    # The original Kaggle file already contains 24 exact duplicate rows; they are
    # removed during preprocessing, so they are reported here but not fatal.
    existing_dupes = int(current_df[key_column].duplicated().sum())
    if existing_dupes:
        print(f"[WARNING] Current dataset has {existing_dupes} duplicate '{key_column}' values "
              "(dropped during preprocessing).")

    current_keys = set(pd.to_datetime(current_df[key_column], utc=True))
    new_keys = set(pd.to_datetime(new_df[key_column], utc=True))
    overlapping = current_keys.intersection(new_keys)
    if overlapping:
        raise ValueError(
            f"new_weather.csv contains {len(overlapping)} timestamps already present in weatherHistory.csv."
        )
    print("[SUCCESS] New batch identity validation passed.")


def create_data_load_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def ingest_new_data(raw_path, new_data_path, archive_dir, lineage_path, report_path, key_column):
    current_df = pd.read_csv(raw_path)
    new_df = pd.read_csv(new_data_path)

    print(f"[INFO] Current dataset rows: {len(current_df)}")
    print(f"[INFO] New dataset rows: {len(new_df)}")

    print("[INFO] Validating new dataset schema...")
    if not validate_schema(new_df, output_report_name="new_data_validation.csv"):
        raise ValueError("New dataset failed schema validation. Ingestion stopped.")

    new_data_hash = calculate_file_hash(new_data_path)
    print(f"[INFO] New dataset SHA256: {new_data_hash}")

    lineage = load_lineage(lineage_path)
    if any(record.get("new_data_hash") == new_data_hash for record in lineage):
        print("[INFO] This new dataset batch has already been ingested.")
        print("[INFO] No archive or retraining action will be performed.")
        create_data_load_report(report_path, {
            "status": "already_ingested",
            "ingestion_performed": False,
            "new_data_hash": new_data_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_rows": len(current_df),
            "new_rows": len(new_df),
        })
        return False

    validate_new_batch_identity(new_df, current_df, key_column)

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_version = get_next_archive_version(archive_dir)
    archive_path = archive_dir / f"weather_v{archive_version:02d}.csv"
    shutil.copy2(raw_path, archive_path)
    print(f"[SUCCESS] Current dataset archived as: {archive_path.name}")

    previous_rows = len(current_df)
    updated_df = pd.concat([current_df, new_df[current_df.columns]], ignore_index=True)

    temp_path = raw_path.with_suffix(".tmp.csv")
    updated_df.to_csv(temp_path, index=False)
    os.replace(temp_path, raw_path)

    print(f"[SUCCESS] New data appended to {raw_path}")
    print(f"[INFO] Previous rows: {previous_rows}")
    print(f"[INFO] Added rows: {len(new_df)}")
    print(f"[INFO] Current rows: {len(updated_df)}")

    lineage_record = {
        "dataset_version": f"v{archive_version + 1:02d}",
        "archive_file": archive_path.relative_to(project_path(".")).as_posix(),
        "source_file": new_data_path.relative_to(project_path(".")).as_posix(),
        "new_data_hash": new_data_hash,
        "previous_rows": previous_rows,
        "new_rows": len(new_df),
        "current_rows": len(updated_df),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    lineage.append(lineage_record)
    save_lineage(lineage_path, lineage)

    create_data_load_report(report_path, {"status": "ingested", "ingestion_performed": True, **lineage_record})
    print("[SUCCESS] Dataset ingestion completed.")
    return True


def load_current_data(raw_path):
    if not raw_path.exists():
        raise FileNotFoundError(f"Current dataset not found: {raw_path}")
    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError(f"Current dataset is empty: {raw_path}")
    print(f"[INFO] Loaded dataset: {raw_path}")
    print(f"[INFO] Dataset shape: {df.shape}")
    return df


def main():
    config = load_config()
    raw_path = project_path(config["data"]["raw_path"])
    new_data_path = project_path(config["data"]["new_data_path"])
    archive_dir = project_path(config["data"]["archive_dir"])
    lineage_path = archive_dir / "dataset_lineage.json"
    report_path = project_path("artifacts/data_load_report.json")
    key_column = config["data"]["id_column"]

    print("=" * 70)
    print("DATA LOADING")
    print("=" * 70)

    if not raw_path.exists():
        raise FileNotFoundError(f"Current dataset not found: {raw_path}")

    if new_data_path.exists():
        print("[INFO] New dataset detected.")
        ingest_new_data(raw_path, new_data_path, archive_dir, lineage_path, report_path, key_column)
    else:
        print("[INFO] No new dataset detected.")
        print("[INFO] Continuing with current dataset.")
        current_df = load_current_data(raw_path)
        create_data_load_report(report_path, {
            "status": "no_new_data",
            "ingestion_performed": False,
            "current_rows": len(current_df),
            "current_columns": len(current_df.columns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    final_df = load_current_data(raw_path)
    print(f"[SUCCESS] Final dataset shape: {final_df.shape}")
    print("[SUCCESS] Data loading stage completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
