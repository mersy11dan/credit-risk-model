"""Data loading, validation, and persistence utilities."""

from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED_DIR, TARGET_COLUMN


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load raw credit application data from CSV."""
    return pd.read_csv(path)


def validate_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Validate required columns and basic data quality."""
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline cleaning: drop duplicates and rows with missing target."""
    cleaned = df.drop_duplicates().copy()

    if TARGET_COLUMN in cleaned.columns:
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])

    return cleaned


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: validate then clean."""
    validated = validate_dataframe(df, require_target=TARGET_COLUMN in df.columns)
    return clean_data(validated)


def save_processed(df: pd.DataFrame, filename: str = "processed.csv") -> Path:
    """Persist processed dataset to data/processed/."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path
