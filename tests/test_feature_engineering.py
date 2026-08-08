import pandas as pd

from src.data_processing import (
    build_customer_aggregates,
    build_customer_dataset,
    build_time_features,
    make_model_ready_dataset,
)


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2", "C3", "C3", "C3"],
            "Amount": [100, 300, -50, 10, 10, 0],
            "TransactionStartTime": [
                "2018-11-15T02:18:49Z",
                "2018-11-16T02:18:49Z",
                "2018-11-15T03:00:00Z",
                "2018-12-01T04:00:00Z",
                "2019-01-01T04:00:00Z",
                None,
            ],
            "ProductCategory": [
                "airtime",
                "airtime",
                "transport",
                "airtime",
                "utility_bill",
                "airtime",
            ],
            "ChannelId": [
                "ChannelId_1",
                "ChannelId_1",
                "ChannelId_2",
                "ChannelId_3",
                "ChannelId_3",
                "ChannelId_3",
            ],
            "FraudResult": [0, 1, 0, 0, 0, 0],
        }
    )


def test_build_customer_aggregates_shape_and_columns():
    agg = build_customer_aggregates(_sample_transactions())
    assert len(agg) == 3
    expected = {
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
    }
    assert expected <= set(agg.columns)


def test_build_time_features_last_strategy_extracts_parts():
    time_df = build_time_features(_sample_transactions(), strategy="last")
    assert len(time_df) == 3
    assert {"txn_hour", "txn_day", "txn_month", "txn_year"} <= set(time_df.columns)

    # C1 last txn is 2018-11-16T02:18:49Z
    c1 = time_df.loc[time_df["CustomerId"] == "C1"].iloc[0]
    assert int(c1["txn_hour"]) == 2
    assert int(c1["txn_month"]) == 11
    assert int(c1["txn_year"]) == 2018


def test_build_customer_dataset_includes_label_and_modes():
    customer_df = build_customer_dataset(
        _sample_transactions(),
        categorical_cols=["ProductCategory", "ChannelId"],
        label_col="FraudResult",
    )
    assert len(customer_df) == 3
    assert "FraudResult" in customer_df.columns
    assert "mode_ProductCategory" in customer_df.columns
    assert "mode_ChannelId" in customer_df.columns

    # Label is max over transactions -> C1 has fraud=1
    c1 = customer_df.loc[customer_df["CustomerId"] == "C1"].iloc[0]
    assert int(c1["FraudResult"]) == 1


def test_make_model_ready_dataset_returns_features_and_pipeline():
    X, y, pipeline = make_model_ready_dataset(
        _sample_transactions(),
        categorical_cols=["ProductCategory", "ChannelId"],
        label_col="FraudResult",
    )
    assert X.shape[0] == 3
    assert y is not None and len(y) == 3
    assert hasattr(pipeline, "fit") and hasattr(pipeline, "transform")
