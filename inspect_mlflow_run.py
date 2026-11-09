"""List the artifacts of one MLflow run.

    python inspect_mlflow_run.py <run_id>
(default: the run behind deployment/model_metadata.json)
"""
import json
import sys
from pathlib import Path

from mlflow import MlflowClient

if len(sys.argv) > 1:
    run_id = sys.argv[1]
else:
    run_id = json.loads(Path("deployment/model_metadata.json").read_text(encoding="utf-8"))["run_id"]

client = MlflowClient()
run = client.get_run(run_id)

print("RUN ID:", run.info.run_id)
print("STATUS:", run.info.status)
print("ARTIFACT URI:", run.info.artifact_uri)
print("METRICS:", {k: round(v, 4) for k, v in run.data.metrics.items()})

print("\nARTIFACTS:")
for artifact in client.list_artifacts(run_id):
    print(f"{artifact.path} | is_dir={artifact.is_dir}")
    if artifact.is_dir:
        for child in client.list_artifacts(run_id, artifact.path):
            print(f"  └── {child.path} | is_dir={child.is_dir}")
