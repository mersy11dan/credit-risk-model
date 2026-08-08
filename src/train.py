"""Model training with hyperparameter search and MLflow experiment tracking."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    DEFAULT_MODELING_DATASET,
    HIGH_RISK_TARGET_COLUMN,
    ID_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_processing import build_preprocessing_pipeline
from src.results import save_model_comparison_results


def get_mlflow_tracking_uri() -> str:
    """Return a file URI suitable for MLflow on all platforms."""
    return Path(MLFLOW_TRACKING_URI).resolve().as_uri()


def load_processed_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the customer-level processed dataset containing ``is_high_risk``."""
    dataset_path = Path(path or DEFAULT_MODELING_DATASET)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_path}. "
            "Run build_modeling_dataset_with_proxy_target() first."
        )

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError(f"Processed dataset is empty: {dataset_path}")
    if HIGH_RISK_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{HIGH_RISK_TARGET_COLUMN}' in {dataset_path}")

    return df


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    target_column: str = HIGH_RISK_TARGET_COLUMN,
    id_columns: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns from the processed dataset."""
    exclude = {target_column, *(id_columns or ID_COLUMNS)}
    feature_cols = [col for col in df.columns if col not in exclude]

    numeric = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical = df[feature_cols].select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def split_train_test(
    df: pd.DataFrame,
    *,
    target_column: str = HIGH_RISK_TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split processed data into stratified train and test sets."""
    id_cols = [c for c in ID_COLUMNS if c in df.columns]
    numeric, categorical = infer_feature_columns(
        df, target_column=target_column, id_columns=id_cols
    )
    feature_cols = numeric + categorical

    X = df[feature_cols]
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_classifier_pipeline(
    model_name: str,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Build a sklearn Pipeline with preprocessing and a named classifier."""
    preprocessor = build_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=True,
    )

    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    if model_name not in classifiers:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(classifiers)}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifiers[model_name]),
        ]
    )


def get_hyperparameter_search_space(model_name: str) -> dict[str, list[Any]]:
    """Return a RandomizedSearchCV parameter space for the given model."""
    spaces: dict[str, dict[str, list[Any]]] = {
        "logistic_regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "decision_tree": {
            "classifier__max_depth": [3, 5, 8, 12, None],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__min_samples_split": [2, 5, 10],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [5, 10, 15, None],
            "classifier__min_samples_leaf": [1, 2, 5],
        },
        "gradient_boosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__max_depth": [2, 3, 4],
        },
    }
    return spaces[model_name]


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred,
    y_proba,
) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a fitted pipeline on the hold-out test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return compute_classification_metrics(y_test, y_pred, y_proba)


def train_model_with_search(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    n_iter: int = 12,
    cv: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, RandomizedSearchCV, dict[str, Any]]:
    """Tune hyperparameters with RandomizedSearchCV and return the best pipeline."""
    pipeline = build_classifier_pipeline(
        model_name,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    param_distributions = get_hyperparameter_search_space(model_name)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params = {k: v for k, v in search.best_params_.items()}
    return search.best_estimator_, search, best_params


def register_model_if_configured(
    model_uri: str,
    model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
) -> None:
    """Register the model in MLflow Model Registry when a registry backend is available."""
    try:
        mlflow.register_model(model_uri=model_uri, name=model_name)
    except Exception as exc:  # noqa: BLE001 - registry may not be configured locally
        mlflow.log_param("model_registry_status", f"skipped: {exc}")


def log_model_run(
    model_name: str,
    pipeline: Pipeline,
    metrics: dict[str, float],
    best_params: dict[str, Any],
    *,
    test_size: float,
) -> str:
    """Log parameters, metrics, and model artifact to the active MLflow run."""
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipeline, artifact_path="model")
    run_id = mlflow.active_run().info.run_id
    return f"runs:/{run_id}/model"


def train_all_models(
    data_path: str | Path | None = None,
    *,
    model_names: list[str] | None = None,
    model_path: str | Path | None = None,
    n_iter: int = 12,
    cv: int = 3,
    register_model: bool = True,
) -> tuple[Pipeline, dict[str, dict[str, float]]]:
    """Train and compare multiple models; save and optionally register the best by ROC-AUC."""
    df = load_processed_dataset(data_path)
    X_train, X_test, y_train, y_test = split_train_test(df)

    numeric, categorical = infer_feature_columns(df)
    candidates = model_names or [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ]

    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_metrics: dict[str, dict[str, float]] = {}
    best_model_name = None
    best_model = None
    best_roc_auc = -1.0
    best_model_uri = None

    for model_name in candidates:
        with mlflow.start_run(run_name=model_name):
            pipeline, search, best_params = train_model_with_search(
                model_name,
                X_train,
                y_train,
                numeric_features=numeric,
                categorical_features=categorical,
                n_iter=n_iter,
                cv=cv,
            )
            metrics = evaluate_model(pipeline, X_test, y_test)
            all_metrics[model_name] = metrics

            mlflow.log_metric("cv_best_score", float(search.best_score_))
            model_uri = log_model_run(
                model_name,
                pipeline,
                metrics,
                best_params,
                test_size=TEST_SIZE,
            )

            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model_name = model_name
                best_model = pipeline
                best_model_uri = model_uri

    if best_model is None or best_model_name is None:
        raise RuntimeError("No models were trained successfully.")

    output_path = Path(model_path or MODELS_DIR / "model.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(best_model, f)

    save_model_comparison_results(all_metrics, best_model_name, output_dir=output_path.parent)

    if register_model and best_model_uri:
        with mlflow.start_run(run_name="best_model_registration"):
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_metrics(all_metrics[best_model_name])
            register_model_if_configured(best_model_uri)

    return best_model, all_metrics


def train(
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    *,
    model_names: list[str] | None = None,
    n_iter: int = 12,
    cv: int = 3,
    register_model: bool = True,
) -> Pipeline:
    """Train credit risk models on the processed dataset with ``is_high_risk`` target."""
    best_model, _ = train_all_models(
        data_path=data_path,
        model_names=model_names,
        model_path=model_path,
        n_iter=n_iter,
        cv=cv,
        register_model=register_model,
    )
    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bati Bank credit risk models")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_MODELING_DATASET),
        help="Path to processed modeling dataset CSV",
    )
    parser.add_argument("--model", default=str(MODELS_DIR / "model.pkl"), help="Output model path")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"],
        help="Models to train",
    )
    parser.add_argument("--n-iter", type=int, default=12, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip MLflow Model Registry registration",
    )
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_path=args.model,
        model_names=args.models,
        n_iter=args.n_iter,
        cv=args.cv,
        register_model=not args.no_register,
    )
