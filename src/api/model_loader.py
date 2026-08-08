"""Load the best trained model from MLflow registry or local artifact."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import (
    DEFAULT_MODEL_PATH,
    MLFLOW_REGISTERED_MODEL_NAME,
)
from src.train import get_mlflow_tracking_uri


class ModelStore:
    """Lazy loader for the production credit risk model."""

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        registry_name: str | None = None,
        registry_stage: str | None = None,
    ) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.registry_name = registry_name or MLFLOW_REGISTERED_MODEL_NAME
        self.registry_stage = registry_stage or os.getenv("MLFLOW_MODEL_STAGE")
        self.model = None
        self.source: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load model using MODEL_URI, registry, or local pickle fallback."""
        self.model, self.source = load_best_model(
            model_path=self.model_path,
            registry_name=self.registry_name,
            registry_stage=self.registry_stage,
        )


def _load_from_uri(model_uri: str):
    return mlflow.sklearn.load_model(model_uri), model_uri


def _load_from_registry(registry_name: str, registry_stage: str | None) -> tuple[object, str]:
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    client = MlflowClient()

    if registry_stage:
        model_uri = f"models:/{registry_name}/{registry_stage}"
        return _load_from_uri(model_uri)

    versions = client.search_model_versions(f"name='{registry_name}'")
    if not versions:
        raise FileNotFoundError(f"No registered versions found for model '{registry_name}'.")

    latest = max(versions, key=lambda version: int(version.version))
    model_uri = f"models:/{registry_name}/{latest.version}"
    return _load_from_uri(model_uri)


def _load_from_pickle(model_path: Path) -> tuple[object, str]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    with open(model_path, "rb") as handle:
        return pickle.load(handle), str(model_path)


def load_best_model(
    *,
    model_path: Path | None = None,
    registry_name: str | None = None,
    registry_stage: str | None = None,
) -> tuple[object, str]:
    """Load the best model from MLflow registry or a saved local artifact.

    Resolution order:
    1. ``MODEL_URI`` environment variable
    2. MLflow Model Registry (stage or latest version)
    3. Local pickle at ``models/model.pkl``
    """
    artifact_path = Path(model_path or DEFAULT_MODEL_PATH)
    registry = registry_name or MLFLOW_REGISTERED_MODEL_NAME
    stage = registry_stage or os.getenv("MLFLOW_MODEL_STAGE")

    model_uri = os.getenv("MODEL_URI")
    if model_uri:
        return _load_from_uri(model_uri)

    try:
        return _load_from_registry(registry, stage)
    except Exception:
        return _load_from_pickle(artifact_path)
