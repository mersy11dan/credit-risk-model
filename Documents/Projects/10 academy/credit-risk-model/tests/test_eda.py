"""Tests for EDA helper functions."""

import pandas as pd

from src.eda import (
    compute_risk_metrics,
    customer_risk_profile,
    dataset_overview,
    describe_categorical,
    describe_numeric,
    eda_summary,
    group_summary,
    missingness_report,
)


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2", "C2", "C3"],
            "Amount": [1000, -50, 500, 200, 0],
            "Value": [1000, 50, 500, 200, 0],
            "FraudResult": [0, 0, 1, 0, 0],
            "ProductCategory": [
                "airtime",
                "financial_services",
                "airtime",
                "utility_bill",
                "airtime",
            ],
        }
    )


def test_dataset_overview():
    df = _sample_transactions()
    overview = dataset_overview(df)

    assert overview["rows"] == 5
    assert overview["columns"] == 5
    assert "Amount" in overview["numeric_columns"]
    assert "ProductCategory" in overview["categorical_columns"]


def test_missingness_report():
    df = _sample_transactions()
    df.loc[0, "Amount"] = None
    report = missingness_report(df)

    assert report.iloc[0]["column"] == "Amount"
    assert report.iloc[0]["missing_count"] == 1


def test_describe_numeric_and_categorical():
    df = _sample_transactions()
    numeric = describe_numeric(df)
    categorical = describe_categorical(df)

    assert "Amount" in numeric.index
    assert "ProductCategory" in categorical
    assert len(categorical["ProductCategory"]) <= 10


def test_group_summary():
    df = _sample_transactions()
    summary = group_summary(
        df,
        by="ProductCategory",
        value_cols=["Value"],
        agg_funcs=["count", "mean"],
    )

    assert "ProductCategory" in summary.columns
    assert any("Value" in col for col in summary.columns)


def test_compute_risk_metrics():
    metrics = compute_risk_metrics(_sample_transactions())

    assert metrics["transaction_count"] == 5
    assert metrics["unique_customers"] == 3
    assert metrics["proxy_event_count"] == 1
    assert metrics["customers_with_proxy_event"] == 1


def test_customer_risk_profile():
    profile = customer_risk_profile(_sample_transactions())

    assert len(profile) == 3
    assert "transaction_count" in profile.columns
    assert profile.loc[profile["CustomerId"] == "C2", "has_proxy_event"].iloc[0] == 1


def test_eda_summary_runs_end_to_end():
    summary = eda_summary(_sample_transactions())

    assert "overview" in summary
    assert "risk_metrics" in summary
    assert summary["overview"]["rows"] == 5
