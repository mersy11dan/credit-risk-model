"""FastAPI service for Bati Bank probability-of-default scoring."""

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import (
    CreditApplication,
    PredictionResponse,
    risk_category_from_probability,
)
from src.config import DEFAULT_MODEL_PATH

_model = None


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    """Load model from disk; raises if artifact is missing."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    global _model
    try:
        _model = load_model()
    except FileNotFoundError:
        _model = None
    yield


app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Probability-of-default scoring for loan applications",
    version="0.1.0",
    lifespan=lifespan,
)


def get_model():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and deploy first.")
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):
    """Score a single credit application."""
    model = get_model()
    features = pd.DataFrame([application.model_dump()])
    probability = float(model.predict_proba(features)[0][1])
    return PredictionResponse(
        probability_of_default=probability,
        risk_category=risk_category_from_probability(probability),
    )
