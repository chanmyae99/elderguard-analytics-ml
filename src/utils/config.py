"""
config.py
---------
Loads configuration values from config/config.yaml and exposes
them as module-level constants.

All pipeline modules import from here — no hardcoded values elsewhere.


"""

import yaml
import os

# Load config.yaml relative to the project root
_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),  # src/utils/
    "..", "..",                  # up to project root
    "config", "config.yaml"
)

with open(os.path.normpath(_CONFIG_PATH), "r") as f:
    _cfg = yaml.safe_load(f)

# ── Data paths ────────────────────────────────────────────────
PROCESSED_DATA_PATH = _cfg["data"]["csv_path"]
DB_PATH             = _cfg["data"].get("db_path", "data/gas_monitoring.db")
DB_TABLE            = _cfg["data"].get("db_table", "gas_monitoring")

# ── Columns ───────────────────────────────────────────────────
TARGET_COL       = _cfg["preprocessing"]["target_col"]
DROP_COLS        = _cfg["preprocessing"]["drop_cols"]
CATEGORICAL_COLS = _cfg["preprocessing"]["categorical_cols"]

# ── Split ─────────────────────────────────────────────────────
TEST_SIZE    = _cfg["data"]["test_size"]
RANDOM_STATE = _cfg["data"]["random_state"]

# ── Cross-validation ──────────────────────────────────────────
CV_FOLDS = _cfg["evaluation"]["cv_folds"]

# ── Output directories ────────────────────────────────────────
MODEL_DIR  = _cfg["output"]["model_dir"]
REPORT_DIR = _cfg["output"]["report_dir"]

# ── Model hyperparameters ─────────────────────────────────────
LR_PARAMS  = _cfg["models"]["logistic_regression"]
RF_PARAMS  = _cfg["models"]["random_forest"]
XGB_PARAMS = _cfg["models"]["xgboost"]


TUNING_CONFIG = _cfg["tuning"]