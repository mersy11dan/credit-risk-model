"""Weight of Evidence (WoE) and Information Value (IV) helpers.

These utilities are intended for **analysis and scorecard-style interpretation**.
They are useful when you want transparent, regulator-friendly evidence of whether
categorical (or discretized) variables have predictive strength for a binary target.

Notes
-----
- WoE/IV are most commonly applied to categorical variables or *binned* numeric variables.
- This module deliberately keeps the implementation simple and reproducible.
- The smoothing parameter prevents divide-by-zero when an event/non-event count is 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WoeIvResult:
    """Container for WoE table and overall IV."""

    table: pd.DataFrame
    iv: float


def _validate_binary_target(series: pd.Series, *, target_name: str) -> None:
    """Ensure the target is binary (two unique values excluding NA)."""
    unique = set(series.dropna().unique().tolist())
    if len(unique) != 2:
        raise ValueError(f"Target '{target_name}' must be binary; got values: {sorted(unique)}")


def woe_iv_table(
    df: pd.DataFrame,
    feature: str,
    target: str,
    *,
    event: int | str = 1,
    smoothing: float = 0.5,
    include_missing_as_category: bool = True,
) -> WoeIvResult:
    """Compute a WoE table and IV for a single (categorical) feature.

    Parameters
    ----------
    df:
        Input dataframe.
    feature:
        Feature column name. Typically categorical or already binned.
    target:
        Binary target column name.
    event:
        Which target value is treated as the "event" (e.g., default=1).
    smoothing:
        Additive smoothing applied to event/non-event distributions to avoid
        infinite WoE when a category has 0 events or 0 non-events.
    include_missing_as_category:
        If True, missing feature values are treated as their own category.

    Returns
    -------
    WoeIvResult
        Contains a per-category table and overall IV (sum of IV components).
    """
    if feature not in df.columns:
        raise ValueError(f"Feature not found: {feature}")
    if target not in df.columns:
        raise ValueError(f"Target not found: {target}")

    _validate_binary_target(df[target], target_name=target)

    x = df[feature]
    if include_missing_as_category:
        x = x.astype("object").where(~x.isna(), "__MISSING__")

    y = df[target]
    is_event = y == event
    if is_event.sum() == 0 or (~is_event).sum() == 0:
        raise ValueError("Both event and non-event classes must be present to compute WoE/IV.")

    grouped = pd.DataFrame({"x": x, "is_event": is_event}).groupby("x", dropna=False)
    counts = grouped["is_event"].agg(["count", "sum"]).rename(columns={"sum": "event_count"})
    counts["non_event_count"] = counts["count"] - counts["event_count"]

    total_event = float(counts["event_count"].sum())
    total_non_event = float(counts["non_event_count"].sum())

    # Smoothed distributions (avoid log(0) and division by zero).
    counts["dist_event"] = (counts["event_count"] + smoothing) / (
        total_event + smoothing * len(counts)
    )
    counts["dist_non_event"] = (counts["non_event_count"] + smoothing) / (
        total_non_event + smoothing * len(counts)
    )

    counts["woe"] = np.log(counts["dist_event"] / counts["dist_non_event"])
    counts["iv_component"] = (counts["dist_event"] - counts["dist_non_event"]) * counts["woe"]

    counts["event_rate"] = counts["event_count"] / counts["count"]
    counts = counts.reset_index().rename(columns={"x": "category"})

    iv = float(counts["iv_component"].sum())
    counts = counts.sort_values("woe").reset_index(drop=True)

    return WoeIvResult(table=counts, iv=iv)


def iv_summary(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    event: int | str = 1,
    smoothing: float = 0.5,
    min_unique: int = 2,
) -> pd.DataFrame:
    """Compute IV across many features for quick screening.

    This is designed to make it easy to inspect predictive strength of
    categorical variables. For numeric variables, discretize/bin first.
    """
    rows: list[dict[str, object]] = []
    for feature in features:
        if feature not in df.columns:
            rows.append(
                {"feature": feature, "iv": np.nan, "unique": np.nan, "error": "missing_feature"}
            )
            continue

        unique = int(df[feature].nunique(dropna=True))
        if unique < min_unique:
            rows.append(
                {
                    "feature": feature,
                    "iv": np.nan,
                    "unique": unique,
                    "error": "too_few_unique_values",
                }
            )
            continue

        try:
            result = woe_iv_table(
                df,
                feature=feature,
                target=target,
                event=event,
                smoothing=smoothing,
                include_missing_as_category=True,
            )
            rows.append({"feature": feature, "iv": result.iv, "unique": unique, "error": None})
        except Exception as e:  # noqa: BLE001 - analysis helper; surface failure reason
            rows.append({"feature": feature, "iv": np.nan, "unique": unique, "error": str(e)})

    summary = pd.DataFrame(rows)
    return summary.sort_values("iv", ascending=False, na_position="last").reset_index(drop=True)


def bin_numeric(
    series: pd.Series,
    *,
    q: int = 10,
    duplicates: str = "drop",
) -> pd.Series:
    """Quantile-bin a numeric series into ordered categories for WoE/IV.

    This helper exists to keep analysis reproducible with a simple default.
    """
    return pd.qcut(series, q=q, duplicates=duplicates)
