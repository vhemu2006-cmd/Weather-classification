"""Quick look at the registered weather model versions (run from the project root)."""
from mlflow import MlflowClient

MODEL_NAME = "Weather_Daily_Summary_Production_Model"

client = MlflowClient()
model = client.get_registered_model(MODEL_NAME)

print("MODEL:", model.name)
print("VERSIONS:")
for version in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version)):
    print(f"version={version.version} stage={version.current_stage} run_id={version.run_id}")
