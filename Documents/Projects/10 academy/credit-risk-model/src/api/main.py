"""FastAPI application for credit risk scoring."""

import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import CreditApplication, PredictionResponse

app = FastAPI(title="Credit Risk API", version="1.0.0")

MODEL_PATH = Path("models/model.pkl")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not loaded")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):
    model = get_model()
    features = [list(application.model_dump().values())]
    probability = float(model.predict_proba(features)[0][1])
    return PredictionResponse(default_probability=probability)
