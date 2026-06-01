"""Tests for model training pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import HIGH_RISK_TARGET_COLUMN
from src.train import (
    compute_classification_metrics,
    infer_feature_columns,
    load_processed_dataset,
    split_train_test,
    train,
)


@pytest.fixture
def sample_modeling_dataset(tmp_path: Path) -> Path:
    rng = pd.Series([0, 1] * 50)
    df = pd.DataFrame(
        {
            "CustomerId": [f"C{i}" for i in range(100)],
            "total_transaction_amount": rng * 1000 + 500,
            "avg_transaction_amount": rng * 100 + 50,
            "transaction_count": rng * 5 + 3,
            "std_transaction_amount": rng * 20 + 5,
            "txn_hour": 10,
            "txn_day": 15,
            "txn_month": 6,
            "txn_year": 2019,
            "mode_ProductCategory": ["airtime", "utility_bill"] * 50,
            HIGH_RISK_TARGET_COLUMN: rng.values,
        }
    )
    path = tmp_path / "modeling_dataset.csv"
    df.to_csv(path, index=False)
    return path


def test_load_processed_dataset_requires_target(tmp_path):
    path = tmp_path / "missing_target.csv"
    pd.DataFrame({"CustomerId": ["C1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing target column"):
        load_processed_dataset(path)


def test_infer_feature_columns(sample_modeling_dataset):
    df = load_processed_dataset(sample_modeling_dataset)
    numeric, categorical = infer_feature_columns(df)
    assert "total_transaction_amount" in numeric
    assert "mode_ProductCategory" in categorical
    assert HIGH_RISK_TARGET_COLUMN not in numeric + categorical


def test_split_train_test_is_stratified(sample_modeling_dataset):
    df = load_processed_dataset(sample_modeling_dataset)
    _, _, y_train, y_test = split_train_test(df, test_size=0.2, random_state=42)
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert set(y_train.unique()) == {0, 1}


def test_compute_classification_metrics():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = [0, 1, 1, 0]
    y_proba = [0.1, 0.6, 0.8, 0.4]
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(metrics.keys())


def test_train_runs_two_models(sample_modeling_dataset, tmp_path, monkeypatch):
    mlruns_dir = (tmp_path / "mlruns").resolve()
    models_dir = tmp_path / "models"
    monkeypatch.setattr("src.train.MLFLOW_TRACKING_URI", mlruns_dir)
    monkeypatch.setattr("src.train.MODELS_DIR", models_dir)

    model = train(
        data_path=sample_modeling_dataset,
        model_path=models_dir / "model.pkl",
        model_names=["logistic_regression", "decision_tree"],
        n_iter=2,
        cv=2,
        register_model=False,
    )

    assert model is not None
    assert (models_dir / "model.pkl").exists()
    assert (models_dir / "best_model.json").exists()
