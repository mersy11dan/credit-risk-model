"""Tests for data processing utilities."""

import pandas as pd

from src.data_processing import preprocess


def test_preprocess_drops_missing_values():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    result = preprocess(df)
    assert len(result) == 2
    assert result.isna().sum().sum() == 0
