"""Train a credit risk classification model."""

import argparse
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data_processing import load_raw_data, preprocess


def train(data_path: str, model_path: str, target_column: str = "default") -> None:
    df = preprocess(load_raw_data(data_path))
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit risk model")
    parser.add_argument("--data", required=True, help="Path to training data CSV")
    parser.add_argument("--model", default="models/model.pkl", help="Output model path")
    parser.add_argument("--target", default="default", help="Target column name")
    args = parser.parse_args()
    train(args.data, args.model, args.target)
