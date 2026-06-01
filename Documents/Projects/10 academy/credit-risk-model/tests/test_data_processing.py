"""Unit tests for data processing and feature engineering."""

import pandas as pd
import pytest

from src.config import HIGH_RISK_TARGET_COLUMN, TRANSACTION_DATA_PATH
from src.data_processing import (
    build_customer_dataset,
    clean_data,
    load_raw_data,
    load_transactions,
    preprocess,
    rfm_kmeans_proxy_target,
    validate_dataframe,
)


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Small deterministic transaction dataset for feature and proxy tests."""
    return pd.DataFrame(
        {
            "CustomerId": ["A", "A", "A", "B", "B", "C"],
            "Amount": [100, 120, 90, 10, 15, 5],
            "Value": [100, 120, 90, 10, 15, 5],
            "TransactionStartTime": [
                "2019-01-10T00:00:00Z",
                "2019-01-11T00:00:00Z",
                "2019-01-12T00:00:00Z",
                "2019-01-05T00:00:00Z",
                "2019-01-06T00:00:00Z",
                "2018-12-01T00:00:00Z",
            ],
            "ProductCategory": [
                "airtime",
                "airtime",
                "airtime",
                "utility_bill",
                "utility_bill",
                "transport",
            ],
            "ChannelId": [
                "ChannelId_1",
                "ChannelId_1",
                "ChannelId_1",
                "ChannelId_2",
                "ChannelId_2",
                "ChannelId_3",
            ],
        }
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


def test_build_customer_dataset_returns_expected_feature_columns(sample_transactions):
    """Feature engineering should produce stable aggregate, time, and mode columns."""
    customer_df = build_customer_dataset(
        sample_transactions,
        categorical_cols=["ProductCategory", "ChannelId"],
        label_col=None,
    )

    expected_numeric = {
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "txn_hour",
        "txn_day",
        "txn_month",
        "txn_year",
    }
    expected_categorical = {"mode_ProductCategory", "mode_ChannelId"}

    assert len(customer_df) == 3
    assert expected_numeric <= set(customer_df.columns)
    assert expected_categorical <= set(customer_df.columns)

    # Customer A: three transactions summing to 310
    customer_a = customer_df.loc[customer_df["CustomerId"] == "A"].iloc[0]
    assert customer_a["transaction_count"] == 3
    assert customer_a["total_transaction_amount"] == 310


def test_rfm_proxy_target_marks_least_engaged_customer_as_high_risk(sample_transactions):
    """Proxy target should flag the least engaged customer cluster as high risk."""
    snapshot = pd.Timestamp("2019-01-13", tz="UTC")

    proxy = rfm_kmeans_proxy_target(
        sample_transactions,
        snapshot_date=snapshot,
        random_state=42,
    )

    assert {HIGH_RISK_TARGET_COLUMN, "rfm_cluster", "recency_days", "frequency", "monetary"} <= set(
        proxy.columns
    )
    assert set(proxy[HIGH_RISK_TARGET_COLUMN].unique()).issubset({0, 1})
    assert proxy[HIGH_RISK_TARGET_COLUMN].sum() >= 1

    # Customer C is oldest, lowest frequency, and lowest monetary spend.
    customer_c = proxy.loc[proxy["CustomerId"] == "C"].iloc[0]
    assert int(customer_c[HIGH_RISK_TARGET_COLUMN]) == 1


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
