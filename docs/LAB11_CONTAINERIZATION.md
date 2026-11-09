# Lab 11 — Containerization and Deployment

## Stage order

| Stage | Task | Evidence to capture |
|---|---|---|
| 11.1 | Architecture: API + deployment bundle + config in one image | diagram below |
| 11.2 | Prepare bundle: `python src/mlops_pipeline/11_prepare_deployment.py` | `deployment/model_metadata.json` |
| 11.3 | `Dockerfile` (python:3.11-slim, `requirements-api.txt`, uvicorn) | file |
| 11.4 | `.dockerignore` (venv, data, mlruns, models, tests, notebooks…) | small build context |
| 11.5 | `docker build -t weather-summary-api:1.0 .` | `docker images` |
| 11.6 | `docker run -d -p 8000:8000 --name weather-api weather-summary-api:1.0` | `docker ps`, logs |
| 11.7 | Call `/health`, `/metadata`, `/predict`, `/predict/batch` on the container | responses |
| 11.8 | `python -m tests.test_local_vs_container` | identical label and probability |
| 11.9 | Pinned `requirements-api.txt`; rebuild gives same predictions | `pip freeze` inside container |
| 11.10 | Failure simulation (below) | error logs |
| 11.11 | Debug with `docker logs`, `docker exec`, `kubectl describe/logs` | notes |
| 11.12 | Recovery: fix, rebuild, redeploy | healthy container |
| 11.13 | `kubectl apply -f k8s/deployment.yaml` | `kubectl get pods` |
| 11.14 | `kubectl apply -f k8s/service.yaml`; port-forward or NodePort | `/health` via service |
| 11.15 | `kubectl scale deployment weather-api --replicas=3` | 3/3 ready |
| 11.16 | Delete a pod; Kubernetes recreates it | new pod name |
| 11.17 | Automated rollout: `python deployment/deploy_production.py` | `reports/deployment_automation_report.json` |

```text
image weather-summary-api:model-vN
└── /app
    ├── api/                    FastAPI app
    ├── deployment/
    │   ├── model.pkl           Production model (from MLflow)
    │   ├── preprocessor.pkl    same run
    │   ├── label_encoder.pkl   same run
    │   └── model_metadata.json
    └── config.yaml
```

## Failure scenarios for 11.10

| Scenario | How to cause it | Symptom | Fix |
|---|---|---|---|
| Dependency mismatch | set `scikit-learn==1.3.0` in `requirements-api.txt`, rebuild | unpickling error / InconsistentVersionWarning at startup; 503 on `/predict` | restore pinned version |
| Missing model artifact | remove `label_encoder.pkl` from the bundle, rebuild | build fails at `COPY` (or 503 if mounted) | rerun `11_prepare_deployment.py` |
| Incorrect configuration | wrong `containerPort`/`targetPort` | service unreachable | match port 8000 |
| Wrong image tag | `image: weather-summary-api:9.9` | `ErrImagePull` / `ImagePullBackOff` | correct tag, `imagePullPolicy: IfNotPresent` |
| Not enough memory | `limits.memory: 64Mi` | `OOMKilled` | raise to ≥ 512Mi |

## Notes

* The model is ~35–40 MB; the image is mostly Python + scikit-learn.
* Readiness/liveness probes call `/health`; the API stays up even if the bundle is missing
  so you can inspect `/metadata`, but `/predict` returns 503.
* `deploy_production.py` tags images by model version, reuses an existing tag, and reads the
  deployment/container names from `k8s/deployment.yaml`.
