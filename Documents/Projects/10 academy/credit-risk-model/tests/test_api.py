"""Tests for FastAPI credit risk inference service."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, model_store


class FakeModel:
    """Minimal stand-in for a fitted sklearn pipeline."""

    def predict_proba(self, X):
        return [[0.75, 0.25]]


@pytest.fixture
def client():
    model_store.model = FakeModel()
    model_store.source = "test://fake-model"
    yield TestClient(app)
    model_store.model = None
    model_store.source = None


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_source"] == "test://fake-model"


def test_predict_returns_risk_probability(client):
    payload = {
        "total_transaction_amount": 1000.0,
        "avg_transaction_amount": 200.0,
        "transaction_count": 5,
        "std_transaction_amount": 25.0,
        "txn_hour": 10,
        "txn_day": 12,
        "txn_month": 6,
        "txn_year": 2019,
        "mode_ProductCategory": "airtime",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["risk_probability"] == 0.25
    assert body["risk_category"] == "Medium"


def test_predict_returns_503_when_model_not_loaded():
    model_store.model = None
    model_store.source = None
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "total_transaction_amount": 100.0,
            "avg_transaction_amount": 50.0,
            "transaction_count": 2,
            "std_transaction_amount": 5.0,
            "txn_hour": 9,
            "txn_day": 1,
            "txn_month": 1,
            "txn_year": 2019,
            "mode_ProductCategory": "airtime",
        },
    )
    assert response.status_code == 503


def test_predict_validates_request_payload(client):
    response = client.post("/predict", json={"transaction_count": -1})
    assert response.status_code == 422
