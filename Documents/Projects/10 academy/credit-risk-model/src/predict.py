"""Generate predictions using a trained credit risk model."""

import argparse
import pickle

import pandas as pd


def predict(model_path: str, data_path: str, output_path: str) -> None:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(data_path)
    predictions = model.predict_proba(df)[:, 1]

    result = df.copy()
    result["default_probability"] = predictions
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run credit risk predictions")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to save predictions")
    args = parser.parse_args()
    predict(args.model, args.data, args.output)
