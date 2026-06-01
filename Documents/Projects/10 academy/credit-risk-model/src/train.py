"""Model training with MLflow experiment tracking."""

import argparse
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    ID_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_processing import load_raw_data, preprocess, save_processed
from src.features import build_preprocessor


def build_model_pipeline(preprocessor) -> Pipeline:
    """Combine preprocessing and classifier into a single sklearn Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute classification metrics for probability-of-default model."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
    }


def train(
    data_path: str,
    model_path: str | Path | None = None,
    target_column: str = TARGET_COLUMN,
) -> Pipeline:
    """Train credit risk model and log run to MLflow."""
    df = preprocess(load_raw_data(data_path))
    save_processed(df)

    feature_cols = [c for c in df.columns if c not in {target_column, *ID_COLUMNS}]
    X = df[feature_cols]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    model = build_model_pipeline(preprocessor)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_params({"model_type": "RandomForestClassifier", "test_size": TEST_SIZE})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

    output_path = Path(model_path or MODELS_DIR / "model.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bati Bank credit risk model")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--model", default=str(MODELS_DIR / "model.pkl"), help="Output model path")
    parser.add_argument("--target", default=TARGET_COLUMN, help="Target column name")
    args = parser.parse_args()
    train(args.data, args.model, args.target)
