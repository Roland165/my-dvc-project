# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/artifacts/model

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/

# Fetch the latest trained model from DagsHub through MLflow at build time.
# Credentials are passed as BuildKit secrets, so they never end up in an image layer.
ARG MLFLOW_TRACKING_URI=https://dagshub.com/Roland165/my-dvc-project.mlflow
ARG MLFLOW_EXPERIMENT_NAME=""
# Changing this value (the CI passes the run id) forces a fresh download instead of a cached layer
ARG CACHE_BUST=manual
RUN --mount=type=secret,id=dagshub_user,env=MLFLOW_TRACKING_USERNAME \
    --mount=type=secret,id=dagshub_token,env=MLFLOW_TRACKING_PASSWORD \
    echo "cache bust: ${CACHE_BUST}" && \
    MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
    MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME}" \
    python scripts/download_model.py

COPY app/ app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
