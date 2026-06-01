"""Central configuration for paths, columns, and training defaults."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"

# Data columns (adjust after EDA)
TARGET_COLUMN = "default"
ID_COLUMNS = ["customer_id"]
CATEGORICAL_COLUMNS: list[str] = []
NUMERIC_COLUMNS: list[str] = []

# Training defaults
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT_NAME = "bati-bank-credit-risk"

# API
DEFAULT_MODEL_PATH = MODELS_DIR / "model.pkl"
