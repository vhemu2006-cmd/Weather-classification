"""Build (or reuse) a model-versioned Docker image and roll it out to Kubernetes.

Requires Docker and a reachable Kubernetes cluster (e.g. Docker Desktop
Kubernetes) and an existing 'weather-api' Deployment (kubectl apply -f k8s/).
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
K8S_DEPLOYMENT_PATH = PROJECT_ROOT / "k8s" / "deployment.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "deployment_automation_report.json"

IMAGE_NAME = "weather-summary-api"
BUNDLE = ["model.pkl", "preprocessor.pkl", "label_encoder.pkl"]


def run_command(command):
    print(f"[INFO] Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Deployment metadata not found: {METADATA_PATH}")
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_kubernetes_config():
    if not K8S_DEPLOYMENT_PATH.exists():
        raise FileNotFoundError(f"Kubernetes deployment file not found: {K8S_DEPLOYMENT_PATH}")
    with K8S_DEPLOYMENT_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return (
        config["metadata"]["name"],
        config["metadata"]["labels"]["app"],
        config["spec"]["template"]["spec"]["containers"][0]["name"],
    )


def current_kube_context():
    result = subprocess.run(["kubectl", "config", "current-context"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def save_report(report):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


def deploy_production():
    print("=" * 70)
    print("PRODUCTION DEPLOYMENT")
    print("=" * 70)

    metadata = load_metadata()
    deployment_name, app_label, container_name = load_kubernetes_config()

    if metadata["model_stage"] != "Production":
        raise RuntimeError(f"Deployment blocked. Model stage is '{metadata['model_stage']}', not 'Production'.")

    for name in BUNDLE:
        if not (DEPLOYMENT_DIR / name).exists():
            raise FileNotFoundError(f"Deployment artifact not found: {DEPLOYMENT_DIR / name}")

    model_version = metadata["model_version"]
    image_tag = f"{IMAGE_NAME}:model-v{model_version}"

    print(f"[INFO] Model        : {metadata['model_name']}")
    print(f"[INFO] Model version: {model_version}")
    print(f"[INFO] Run ID       : {metadata['run_id']}")
    print(f"[INFO] Docker image : {image_tag}")
    print(f"[INFO] Kubernetes   : {deployment_name}")
    print(f"[INFO] Container    : {container_name}")

    image_exists = subprocess.run(
        ["docker", "image", "inspect", image_tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0

    if image_exists:
        print(f"[INFO] Docker image already exists: {image_tag}. Reusing existing image.")
    else:
        print("[INFO] Docker image not found. Building...")
        run_command(["docker", "build", "-t", image_tag, "."])

    print("[INFO] Updating Kubernetes deployment...")
    run_command(["kubectl", "set", "image", f"deployment/{deployment_name}", f"{container_name}={image_tag}"])

    print("[INFO] Waiting for Kubernetes rollout...")
    run_command(["kubectl", "rollout", "status", f"deployment/{deployment_name}", "--timeout=180s"])

    save_report({
        "status": "success",
        "model_name": metadata["model_name"],
        "model_version": str(model_version),
        "run_id": metadata["run_id"],
        "docker_image": image_tag,
        "image_reused": image_exists,
        "kubernetes_context": current_kube_context(),
        "kubernetes_deployment": deployment_name,
        "kubernetes_container": container_name,
        "app_label": app_label,
        "deployment_bundle": {name: f"deployment/{name}" for name in BUNDLE + ["model_metadata.json"]},
    })

    print("[SUCCESS] Kubernetes deployment updated and rollout completed.")
    print(f"[SUCCESS] Deployment report saved to {REPORT_PATH}")
    return True


if __name__ == "__main__":
    try:
        sys.exit(0 if deploy_production() else 1)
    except Exception as exc:
        print(f"[ERROR] Production deployment failed: {exc}")
        sys.exit(1)
