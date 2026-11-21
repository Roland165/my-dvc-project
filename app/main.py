"""FastAPI app serving the latest RandomForest digits model fetched from DagsHub."""

import json
import os
from pathlib import Path

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/artifacts/model"))
N_FEATURES = 64  # load_digits: 8x8 images flattened

model = mlflow.pyfunc.load_model(str(MODEL_DIR))

info_file = MODEL_DIR.parent / "model_info.json"
MODEL_RUN_ID = json.loads(info_file.read_text())["run_id"] if info_file.exists() else None

# Docs routes disabled so the app exposes a single endpoint: /predict
app = FastAPI(title="Digits classifier", docs_url=None, redoc_url=None, openapi_url=None)


class PredictRequest(BaseModel):
    inputs: list[list[float]] = Field(..., min_length=1, description="Rows of 64 pixel values (0-16)")


class PredictResponse(BaseModel):
    predictions: list[int]
    model_run_id: str | None = None


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    X = np.asarray(request.inputs, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"each row must contain exactly {N_FEATURES} values, got shape {list(X.shape)}",
        )
    predictions = model.predict(X)
    return PredictResponse(predictions=[int(p) for p in predictions], model_run_id=MODEL_RUN_ID)
