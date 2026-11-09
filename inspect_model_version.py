"""Show one registered model version.   python inspect_model_version.py [version]"""
import sys

from mlflow import MlflowClient

MODEL_NAME = "Weather_Daily_Summary_Production_Model"
version = sys.argv[1] if len(sys.argv) > 1 else "1"

mv = MlflowClient().get_model_version(MODEL_NAME, version)

print("MODEL NAME:", mv.name)
print("VERSION:", mv.version)
print("STAGE:", mv.current_stage)
print("RUN ID:", mv.run_id)
print("SOURCE:", mv.source)
print("STATUS:", mv.status)
print("TAGS:", mv.tags)
print("CREATION TIMESTAMP:", mv.creation_timestamp)
