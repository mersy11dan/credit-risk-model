"""Notebook-friendly EDA helpers for the Bati Bank transaction dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    AMOUNT_COLUMN,
    CUSTOMER_ID_COLUMN,
    PROXY_TARGET_COLUMN,
    VALUE_COLUMN,
)


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return high-level dataset metadata for quick inspection.

    Useful as the first cell output in an EDA notebook.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "duplicate_rows": int(df.duplicated().sum()),
    }


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values per column, sorted by severity."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    report = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values,
            "non_null_count": (len(df) - missing_count).values,
        }
    )
    return report.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def describe_numeric(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Extended descriptive statistics for numerical features."""
    numeric_df = df.select_dtypes(include="number")
    cols = columns or numeric_df.columns.tolist()
    cols = [col for col in cols if col in numeric_df.columns]

    if not cols:
        return pd.DataFrame()

    stats = numeric_df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats["missing_count"] = df[cols].isna().sum()
    stats["missing_pct"] = (stats["missing_count"] / len(df) * 100).round(2)
    stats["zeros"] = (numeric_df[cols] == 0).sum()
    stats["unique"] = numeric_df[cols].nunique()
    return stats


def describe_categorical(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Summarize categorical features with cardinality and top values.

    Returns a mapping of column name to a small summary dataframe.
    """
    categorical_df = df.select_dtypes(exclude="number")
    cols = columns or categorical_df.columns.tolist()
    cols = [col for col in cols if col in categorical_df.columns]

    summaries: dict[str, pd.DataFrame] = {}
    for col in cols:
        value_counts = df[col].value_counts(dropna=False).head(top_n)
        summaries[col] = pd.DataFrame(
            {
                "value": value_counts.index.astype(str),
                "count": value_counts.values,
                "pct": (value_counts.values / len(df) * 100).round(2),
            }
        )
        summaries[col].attrs["unique_count"] = df[col].nunique(dropna=False)
        summaries[col].attrs["missing_count"] = int(df[col].isna().sum())

    return summaries


def group_summary(
    df: pd.DataFrame,
    by: str | list[str],
    value_cols: list[str],
    agg_funcs: list[str] | dict[str, str | list[str]] | None = None,
) -> pd.DataFrame:
    """Aggregate numeric columns by one or more grouping keys.

    Parameters
    ----------
    df:
        Input dataframe.
    by:
        Column(s) to group by.
    value_cols:
        Numeric columns to aggregate.
    agg_funcs:
        Aggregation functions passed to ``DataFrameGroupBy.agg``.
        Defaults to count, mean, sum, and median.
    """
    if agg_funcs is None:
        agg_funcs = ["count", "mean", "sum", "median"]

    grouped = df.groupby(by, dropna=False)[value_cols].agg(agg_funcs)
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = ["_".join(map(str, col)).strip("_") for col in grouped.columns]

    return grouped.reset_index()


def compute_risk_metrics(
    df: pd.DataFrame,
    *,
    customer_id: str = CUSTOMER_ID_COLUMN,
    proxy_col: str = PROXY_TARGET_COLUMN,
    amount_col: str = AMOUNT_COLUMN,
    value_col: str = VALUE_COLUMN,
) -> dict[str, Any]:
    """Compute simple transaction- and customer-level risk indicators.

    When a direct default label is unavailable, ``proxy_col`` (e.g. fraud flag)
    can be used as a behavioral proxy for adverse customer outcomes.
    """
    metrics: dict[str, Any] = {
        "transaction_count": len(df),
        "unique_customers": df[customer_id].nunique() if customer_id in df.columns else None,
    }

    if proxy_col in df.columns:
        proxy_rate = df[proxy_col].mean()
        metrics["proxy_event_rate"] = round(float(proxy_rate), 4)
        metrics["proxy_event_count"] = int(df[proxy_col].sum())

    if amount_col in df.columns:
        metrics["total_amount"] = float(df[amount_col].sum())
        metrics["avg_amount"] = round(float(df[amount_col].mean()), 2)
        metrics["debit_count"] = int((df[amount_col] > 0).sum())
        metrics["credit_count"] = int((df[amount_col] < 0).sum())

    if value_col in df.columns:
        metrics["avg_transaction_value"] = round(float(df[value_col].mean()), 2)
        metrics["max_transaction_value"] = float(df[value_col].max())

    if customer_id in df.columns and proxy_col in df.columns:
        customer_proxy = (
            df.groupby(customer_id)[proxy_col].max().rename("has_proxy_event").reset_index()
        )
        metrics["customers_with_proxy_event"] = int(customer_proxy["has_proxy_event"].sum())
        metrics["customer_proxy_rate"] = round(float(customer_proxy["has_proxy_event"].mean()), 4)

    if customer_id in df.columns:
        txn_per_customer = df.groupby(customer_id).size()
        metrics["avg_transactions_per_customer"] = round(float(txn_per_customer.mean()), 2)
        metrics["median_transactions_per_customer"] = float(txn_per_customer.median())

    if customer_id in df.columns and value_col in df.columns:
        customer_value = df.groupby(customer_id)[value_col].sum()
        metrics["avg_total_value_per_customer"] = round(float(customer_value.mean()), 2)

    return metrics


def customer_risk_profile(
    df: pd.DataFrame,
    *,
    customer_id: str = CUSTOMER_ID_COLUMN,
    proxy_col: str = PROXY_TARGET_COLUMN,
    amount_col: str = AMOUNT_COLUMN,
    value_col: str = VALUE_COLUMN,
) -> pd.DataFrame:
    """Build a customer-level table for proxy-based risk analysis."""
    if customer_id not in df.columns:
        raise ValueError(f"Missing customer id column: {customer_id}")

    agg_map: dict[str, Any] = {"transaction_count": (customer_id, "size")}

    if value_col in df.columns:
        agg_map["total_value"] = (value_col, "sum")
        agg_map["avg_value"] = (value_col, "mean")
        agg_map["max_value"] = (value_col, "max")

    if amount_col in df.columns:
        agg_map["total_amount"] = (amount_col, "sum")
        agg_map["debit_count"] = (amount_col, lambda s: int((s > 0).sum()))

    if proxy_col in df.columns:
        agg_map["proxy_event_count"] = (proxy_col, "sum")
        agg_map["has_proxy_event"] = (proxy_col, "max")

    profile = df.groupby(customer_id).agg(**agg_map).reset_index()

    if "total_value" in profile.columns and "transaction_count" in profile.columns:
        profile["value_per_transaction"] = (
            profile["total_value"] / profile["transaction_count"]
        ).replace([np.inf, -np.inf], np.nan)

    return profile


def eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Run a one-call EDA summary suitable for notebook display."""
    overview = dataset_overview(df)
    missing = missingness_report(df)
    numeric = describe_numeric(df)
    categorical = describe_categorical(df)
    risk = compute_risk_metrics(df)

    return {
        "overview": overview,
        "missingness": missing,
        "numeric_summary": numeric,
        "categorical_summary": categorical,
        "risk_metrics": risk,
    }
