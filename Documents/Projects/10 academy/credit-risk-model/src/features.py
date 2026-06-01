"""Feature engineering and sklearn preprocessing pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if NUMERIC_COLUMNS:
        transformers.append(("numeric", numeric_pipeline, NUMERIC_COLUMNS))
    if CATEGORICAL_COLUMNS:
        transformers.append(("categorical", categorical_pipeline, CATEGORICAL_COLUMNS))

    if not transformers:
        raise ValueError(
            "Define NUMERIC_COLUMNS and/or CATEGORICAL_COLUMNS in src/config.py after EDA."
        )

    return ColumnTransformer(transformers=transformers)


def get_feature_columns(
    df, target_column: str, id_columns: list[str]
) -> tuple[list[str], list[str]]:
    """Infer feature column lists from dataframe, excluding target and ID columns."""
    exclude = {target_column, *id_columns}
    feature_cols = [col for col in df.columns if col not in exclude]

    numeric = [col for col in feature_cols if col in NUMERIC_COLUMNS or df[col].dtype != "object"]
    categorical = [
        col for col in feature_cols if col in CATEGORICAL_COLUMNS or df[col].dtype == "object"
    ]

    return numeric, categorical
