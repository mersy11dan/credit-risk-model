import pandas as pd
import pytest

from src.woe_iv import bin_numeric, iv_summary, woe_iv_table


def test_woe_iv_table_computes_iv_and_table_columns():
    df = pd.DataFrame(
        {
            "feature": ["A", "A", "B", "B", "B", None],
            "target": [1, 0, 0, 0, 1, 0],
        }
    )

    result = woe_iv_table(df, feature="feature", target="target", event=1, smoothing=0.5)

    assert result.iv >= 0
    assert {"category", "count", "event_count", "non_event_count", "woe", "iv_component"} <= set(
        result.table.columns
    )


def test_iv_summary_orders_by_iv_and_handles_missing_feature():
    df = pd.DataFrame(
        {
            "x": ["A", "A", "B", "B", "C", "C"],
            "y": [1, 0, 0, 0, 1, 1],
        }
    )

    summary = iv_summary(df, features=["x", "missing"], target="y")
    assert summary.iloc[0]["feature"] == "x"
    assert summary.loc[summary["feature"] == "missing", "error"].iloc[0] == "missing_feature"


def test_bin_numeric_returns_categorical_bins():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    binned = bin_numeric(s, q=4)
    assert binned.isna().sum() == 0
    assert binned.dtype.name in {"category", "CategoricalDtype"}


def test_woe_iv_table_rejects_non_binary_target():
    df = pd.DataFrame({"feature": ["A", "B", "C"], "target": [0, 1, 2]})
    with pytest.raises(ValueError, match="must be binary"):
        woe_iv_table(df, feature="feature", target="target")
