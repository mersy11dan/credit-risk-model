"""Batch inference for probability-of-default scoring."""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.config import ID_COLUMNS, MODELS_DIR, TARGET_COLUMN


def load_model(model_path: str | Path):
    """Load serialized sklearn pipeline from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_proba(model, df: pd.DataFrame) -> pd.Series:
    """Return probability of default for each row."""
    feature_cols = [c for c in df.columns if c not in {TARGET_COLUMN, *ID_COLUMNS}]
    return pd.Series(model.predict_proba(df[feature_cols])[:, 1], name="probability_of_default")


def run_batch_inference(
    model_path: str | Path,
    data_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Score a batch of applications and write results to CSV."""
    model = load_model(model_path)
    df = pd.read_csv(data_path)
    result = df.copy()
    result["probability_of_default"] = predict_proba(model, df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch credit risk predictions")
    parser.add_argument(
        "--model",
        default=str(MODELS_DIR / "model.pkl"),
        help="Path to trained model",
    )
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to save predictions")
    args = parser.parse_args()
    run_batch_inference(args.model, args.data, args.output)
