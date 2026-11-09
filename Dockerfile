FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
# Deployment bundle produced by src/mlops_pipeline/11_prepare_deployment.py
COPY deployment/model.pkl deployment/preprocessor.pkl deployment/label_encoder.pkl deployment/model_metadata.json ./deployment/
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
