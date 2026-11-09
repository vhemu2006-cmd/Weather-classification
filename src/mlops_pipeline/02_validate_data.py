"""Stage 02 - Raw data schema validation. Exit code 1 halts the pipeline."""
import sys

import pandas as pd

from validation import validate_schema
from weather_common import load_config, project_path


def main():
    config = load_config()
    data_path = project_path(config["data"]["raw_path"])

    if not data_path.exists():
        print(f"[ERROR] Target file not found at: {data_path}")
        sys.exit(1)

    raw_df = pd.read_csv(data_path)
    passed = validate_schema(raw_df, output_report_name="baseline_validation.csv")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
