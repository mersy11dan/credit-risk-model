"""Tests for model loading utilities."""

import pickle
from unittest.mock import patch

from sklearn.linear_model import LogisticRegression

from src.api.model_loader import ModelStore, load_best_model


def test_load_best_model_from_pickle(tmp_path):
    model = LogisticRegression()
    model.fit([[1, 2], [3, 4]], [0, 1])
    model_path = tmp_path / "model.pkl"

    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)

    loaded, source = load_best_model(
        model_path=model_path,
        registry_name="missing-model",
        registry_stage=None,
    )

    assert isinstance(loaded, LogisticRegression)
    assert str(model_path) in source


def test_model_store_loads_pickle(tmp_path):
    model = LogisticRegression()
    model.fit([[1], [2]], [0, 1])
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)

    store = ModelStore(model_path=model_path, registry_name="unused-model")
    with patch("src.api.model_loader._load_from_registry", side_effect=FileNotFoundError):
        store.load()

    assert store.is_loaded
    assert store.model is not None
