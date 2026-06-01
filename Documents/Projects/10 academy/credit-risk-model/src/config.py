"""Central configuration for paths, columns, and training defaults."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"

# Transaction dataset (Xente)
TRANSACTION_DATA_PATH = DATA_RAW_DIR / "data.csv"
VARIABLE_DEFINITIONS_PATH = DATA_RAW_DIR / "Xente_Variable_Definitions.csv"
TRANSACTION_DATETIME_COLUMN = "TransactionStartTime"

# Entity columns
CUSTOMER_ID_COLUMN = "CustomerId"
ACCOUNT_ID_COLUMN = "AccountId"
ID_COLUMNS = [CUSTOMER_ID_COLUMN, ACCOUNT_ID_COLUMN, "TransactionId"]

# Proxy / risk columns (adjust after EDA)
TARGET_COLUMN = "default"
PROXY_TARGET_COLUMN = "FraudResult"
AMOUNT_COLUMN = "Amount"
VALUE_COLUMN = "Value"

# Feature lists (populate after EDA)
CATEGORICAL_COLUMNS: list[str] = []
NUMERIC_COLUMNS: list[str] = []

# Training defaults
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT_NAME = "bati-bank-credit-risk"

# API
DEFAULT_MODEL_PATH = MODELS_DIR / "model.pkl"
