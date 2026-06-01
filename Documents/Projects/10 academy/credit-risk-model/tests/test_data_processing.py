"""Tests for data processing utilities."""

import pandas as pd
import pytest

from src.config import TRANSACTION_DATA_PATH
from src.data_processing import (
    clean_data,
    load_raw_data,
    load_transactions,
    preprocess,
    validate_dataframe,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "income": [50000, 60000, None],
            "default": [0, 1, 0],
        }
    )


def test_load_raw_data_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "missing.csv")


def test_load_raw_data_raises_for_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("col\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_raw_data(empty_file)


def test_load_transactions_parses_datetime():
    if not TRANSACTION_DATA_PATH.exists():
        pytest.skip("Transaction dataset not available")

    df = load_transactions(nrows=100)
    assert "TransactionStartTime" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionStartTime"])


def test_validate_dataframe_requires_target():
    df = pd.DataFrame({"income": [1, 2]})
    with pytest.raises(ValueError, match="Missing target column"):
        validate_dataframe(df)


def test_clean_data_drops_duplicate_rows(sample_df):
    duplicated = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    result = clean_data(duplicated)
    assert len(result) == len(sample_df)


def test_preprocess_drops_rows_with_missing_target():
    df = pd.DataFrame({"default": [0, None, 1], "income": [1, 2, 3]})
    result = preprocess(df)
    assert len(result) == 2
