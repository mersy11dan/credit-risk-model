"""Data loading and preprocessing utilities for credit risk modeling."""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw credit data from a CSV file."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic preprocessing steps to the raw dataframe."""
    processed = df.copy()
    processed = processed.dropna()
    return processed
