"""FastAPI inference service for the Bati Bank credit risk model."""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.model_loader import ModelStore
from src.api.pydantic_models import (
    CustomerFeatures,
    PredictionResponse,
    risk_category_from_probability,
)

model_store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the best available model once at startup."""
    try:
        model_store.load()
    except FileNotFoundError:
        model_store.model = None
        model_store.source = None
    yield


app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Inference service for proxy high-risk probability scoring",
    version="0.2.0",
    lifespan=lifespan,
)


def get_model():
    if not model_store.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model or configure MODEL_URI / MLflow registry.",
        )
    return model_store.model


@app.get("/health")
def health():
    """Service health check."""
    return {
        "status": "ok",
        "model_loaded": model_store.is_loaded,
        "model_source": model_store.source,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    """Score a customer and return high-risk probability."""
    model = get_model()
    feature_row = features.to_feature_row()
    input_df = pd.DataFrame([feature_row])

    try:
        probability = float(model.predict_proba(input_df)[0][1])
    except Exception as exc:  # noqa: BLE001 - return clean API error for feature mismatch
        raise HTTPException(
            status_code=422,
            detail=f"Model prediction failed: {exc}",
        ) from exc

    return PredictionResponse(
        risk_probability=probability,
        risk_category=risk_category_from_probability(probability),
    )
