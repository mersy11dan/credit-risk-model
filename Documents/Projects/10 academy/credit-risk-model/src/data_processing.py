"""Data loading, validation, and persistence utilities."""

from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    TARGET_COLUMN,
    TRANSACTION_DATA_PATH,
    TRANSACTION_DATETIME_COLUMN,
)


def load_raw_data(
    path: str | Path,
    *,
    parse_dates: bool = False,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a CSV dataset with basic path validation.

    Parameters
    ----------
    path:
        Path to the CSV file.
    parse_dates:
        When True, attempt to parse ``TRANSACTION_DATETIME_COLUMN`` if present.
    nrows:
        Optional row limit for quick sampling during development.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty after loading.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path, low_memory=False, nrows=nrows)
    if df.empty:
        raise ValueError(f"Data file is empty: {file_path}")

    if parse_dates and TRANSACTION_DATETIME_COLUMN in df.columns:
        df[TRANSACTION_DATETIME_COLUMN] = pd.to_datetime(
            df[TRANSACTION_DATETIME_COLUMN],
            errors="coerce",
            utc=True,
        )

    return df


def load_transactions(
    path: str | Path | None = None,
    *,
    parse_dates: bool = True,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load the Bati Bank / Xente transaction dataset safely.

    Defaults to ``data/raw/data.csv`` and parses transaction timestamps when
    ``TransactionStartTime`` is available.
    """
    return load_raw_data(
        path or TRANSACTION_DATA_PATH,
        parse_dates=parse_dates,
        nrows=nrows,
    )


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
