"""Tests for data processing utilities."""

import pandas as pd
import pytest

from src.data_processing import clean_data, preprocess, validate_dataframe


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "income": [50000, 60000, None],
            "default": [0, 1, 0],
        }
    )


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
