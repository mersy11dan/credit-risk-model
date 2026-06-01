"""Data loading and feature engineering for the Bati Bank project.

This module intentionally supports two use cases:
- Notebook-friendly data loading / basic validation.
- A reusable, testable feature engineering pipeline that produces model-ready data.
"""

from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    AMOUNT_COLUMN,
    CUSTOMER_ID_COLUMN,
    DATA_PROCESSED_DIR,
    PROXY_TARGET_COLUMN,
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


# -----------------------------
# Feature engineering (modeling)
# -----------------------------


def _coerce_transaction_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the transaction time column is a parsed datetime (UTC)."""
    if TRANSACTION_DATETIME_COLUMN in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df[TRANSACTION_DATETIME_COLUMN]
    ):
        df = df.copy()
        df[TRANSACTION_DATETIME_COLUMN] = pd.to_datetime(
            df[TRANSACTION_DATETIME_COLUMN], errors="coerce", utc=True
        )
    return df


def build_customer_aggregates(
    transactions: pd.DataFrame,
    *,
    customer_id_col: str = CUSTOMER_ID_COLUMN,
    amount_col: str = AMOUNT_COLUMN,
) -> pd.DataFrame:
    """Aggregate transaction history into customer-level numeric features.

    Produces one row per customer. Feature names are stable and suitable for ML pipelines.
    """
    if customer_id_col not in transactions.columns:
        raise ValueError(f"Missing customer id column: {customer_id_col}")
    if amount_col not in transactions.columns:
        raise ValueError(f"Missing amount column: {amount_col}")

    tx = transactions[[customer_id_col, amount_col]].copy()
    agg = (
        tx.groupby(customer_id_col, dropna=False)[amount_col]
        .agg(["sum", "mean", "count", "std"])
        .rename(
            columns={
                "sum": "total_transaction_amount",
                "mean": "avg_transaction_amount",
                "count": "transaction_count",
                "std": "std_transaction_amount",
            }
        )
        .reset_index()
    )
    return agg


def build_time_features(
    transactions: pd.DataFrame,
    *,
    customer_id_col: str = CUSTOMER_ID_COLUMN,
    datetime_col: str = TRANSACTION_DATETIME_COLUMN,
    strategy: str = "last",
) -> pd.DataFrame:
    """Create customer-level time features from transaction timestamps.

    Parameters
    ----------
    strategy:
        How to reduce multiple transaction timestamps per customer.
        - ``last``: use the most recent timestamp.
        - ``first``: use the earliest timestamp.
    """
    if customer_id_col not in transactions.columns:
        raise ValueError(f"Missing customer id column: {customer_id_col}")
    if datetime_col not in transactions.columns:
        raise ValueError(f"Missing datetime column: {datetime_col}")

    tx = _coerce_transaction_datetime(transactions[[customer_id_col, datetime_col]].copy())

    if strategy not in {"last", "first"}:
        raise ValueError("strategy must be one of {'last', 'first'}")

    reduce_fn = "max" if strategy == "last" else "min"
    reduced = (
        tx.groupby(customer_id_col, dropna=False)[datetime_col].agg(reduce_fn).rename("txn_time")
    )
    reduced = reduced.reset_index()

    # Use pandas nullable integers so missing datetimes remain <NA>
    ts = reduced["txn_time"]
    reduced["txn_hour"] = ts.dt.hour.astype("Int64")
    reduced["txn_day"] = ts.dt.day.astype("Int64")
    reduced["txn_month"] = ts.dt.month.astype("Int64")
    reduced["txn_year"] = ts.dt.year.astype("Int64")
    return reduced.drop(columns=["txn_time"])


def build_customer_categorical_features(
    transactions: pd.DataFrame,
    *,
    customer_id_col: str = CUSTOMER_ID_COLUMN,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build customer-level categorical features using the most frequent value (mode).

    This is a pragmatic baseline for customer-level modeling when we only have transaction data.
    """
    if customer_id_col not in transactions.columns:
        raise ValueError(f"Missing customer id column: {customer_id_col}")

    if not categorical_cols:
        return transactions[[customer_id_col]].drop_duplicates().reset_index(drop=True)

    present = [c for c in categorical_cols if c in transactions.columns]
    if not present:
        return transactions[[customer_id_col]].drop_duplicates().reset_index(drop=True)

    def _mode(series: pd.Series):
        m = series.mode(dropna=True)
        return m.iloc[0] if not m.empty else pd.NA

    grouped = transactions.groupby(customer_id_col, dropna=False)[present].agg(_mode).reset_index()
    # Prefix to make lineage clear
    grouped = grouped.rename(columns={c: f"mode_{c}" for c in present})
    return grouped


def build_customer_dataset(
    transactions: pd.DataFrame,
    *,
    customer_id_col: str = CUSTOMER_ID_COLUMN,
    amount_col: str = AMOUNT_COLUMN,
    datetime_col: str = TRANSACTION_DATETIME_COLUMN,
    categorical_cols: list[str] | None = None,
    time_strategy: str = "last",
    label_col: str | None = PROXY_TARGET_COLUMN,
) -> pd.DataFrame:
    """Create a customer-level dataset from transaction-level records.

    Includes:
    - Aggregates: total/avg/count/std of transaction amounts
    - Time features: hour/day/month/year from reduced timestamp
    - Categorical modes: most frequent category per customer (optional)
    - Label: customer-level proxy label (max over transactions) when available
    """
    aggregates = build_customer_aggregates(
        transactions, customer_id_col=customer_id_col, amount_col=amount_col
    )
    time_features = build_time_features(
        transactions,
        customer_id_col=customer_id_col,
        datetime_col=datetime_col,
        strategy=time_strategy,
    )
    categorical = build_customer_categorical_features(
        transactions, customer_id_col=customer_id_col, categorical_cols=categorical_cols
    )

    customer_df = aggregates.merge(time_features, on=customer_id_col, how="left").merge(
        categorical, on=customer_id_col, how="left"
    )

    if label_col and label_col in transactions.columns:
        labels = (
            transactions.groupby(customer_id_col, dropna=False)[label_col]
            .max()
            .rename(label_col)
            .reset_index()
        )
        customer_df = customer_df.merge(labels, on=customer_id_col, how="left")

    return customer_df


class CustomerFeatureBuilder(BaseEstimator, TransformerMixin):
    """Sklearn transformer to build customer-level features from transactions."""

    def __init__(
        self,
        *,
        customer_id_col: str = CUSTOMER_ID_COLUMN,
        amount_col: str = AMOUNT_COLUMN,
        datetime_col: str = TRANSACTION_DATETIME_COLUMN,
        categorical_cols: list[str] | None = None,
        time_strategy: str = "last",
        label_col: str | None = PROXY_TARGET_COLUMN,
    ) -> None:
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.datetime_col = datetime_col
        self.categorical_cols = categorical_cols
        self.time_strategy = time_strategy
        self.label_col = label_col

    def fit(self, X: pd.DataFrame, y=None):  # noqa: ANN001
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return build_customer_dataset(
            X,
            customer_id_col=self.customer_id_col,
            amount_col=self.amount_col,
            datetime_col=self.datetime_col,
            categorical_cols=self.categorical_cols,
            time_strategy=self.time_strategy,
            label_col=self.label_col,
        )


def build_preprocessing_pipeline(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create a ColumnTransformer that imputes and encodes model features."""
    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=num_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_model_ready_dataset(
    transactions: pd.DataFrame,
    *,
    categorical_cols: list[str] | None = None,
    label_col: str | None = PROXY_TARGET_COLUMN,
    include_customer_id: bool = False,
    scale_numeric: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None, Pipeline]:
    """Create a model-ready customer dataset and fitted preprocessing pipeline.

    Returns
    -------
    X:
        A 2D feature matrix as a pandas DataFrame (one row per customer).
    y:
        Optional label series if ``label_col`` exists in the dataset.
    pipeline:
        A fitted sklearn Pipeline that includes customer feature building and preprocessing.
    """
    # Build customer-level dataset first (pandas), then learn encodings/scalers (sklearn).
    cat_default = categorical_cols or [
        "ProductCategory",
        "ChannelId",
        "CurrencyCode",
        "ProviderId",
        "PricingStrategy",
        "CountryCode",
    ]

    feature_builder = CustomerFeatureBuilder(
        categorical_cols=cat_default,
        label_col=label_col,
    )

    customer_df = feature_builder.transform(transactions)
    y = customer_df[label_col] if (label_col and label_col in customer_df.columns) else None

    drop_cols = [label_col] if (label_col and label_col in customer_df.columns) else []
    if not include_customer_id:
        drop_cols = [*drop_cols, CUSTOMER_ID_COLUMN]

    features_df = customer_df.drop(columns=[c for c in drop_cols if c in customer_df.columns])

    numeric_features = features_df.select_dtypes(include="number").columns.tolist()
    categorical_features = features_df.select_dtypes(exclude="number").columns.tolist()

    preprocessor = build_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )
    X_arr = pipeline.fit_transform(features_df)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    X = pd.DataFrame(X_arr, columns=feature_names, index=features_df.index)

    return X, y, pipeline
