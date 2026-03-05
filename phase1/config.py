from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "oes_ml_dataset_1nm.csv"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# --- Column definitions ---
META_COLS = ["sheet", "condition"]
DISCHARGE_COLS = ["frequency_hz", "pulse_width_ns", "rise_time_ns", "flow_rate_sccm"]
TARGET_COL = "h2o2_rate"
OES_PREFIX = "I_"

# --- PCA ---
PCA_VARIANCE_THRESHOLD = 0.95

# --- Model hyperparameters ---
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

PLS_MAX_COMPONENTS = 10

SVR_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0],
    "epsilon": [0.01, 0.1, 0.5],
    "gamma": ["scale", "auto"],
}

XGBOOST_PARAMS = {
    "max_depth": 2,
    "n_estimators": 50,
    "learning_rate": 0.05,
    "reg_alpha": 10.0,
    "reg_lambda": 10.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

MLP_CONFIG = {
    "hidden_sizes": [32, 16],
    "dropout": 0.4,
    "weight_decay": 1e-2,
    "lr": 1e-3,
    "max_epochs": 500,
    "patience": 50,
}

CNN_CONFIG = {
    "conv_channels": [16, 32],
    "kernel_size": 7,
    "dropout": 0.4,
    "weight_decay": 1e-2,
    "lr": 1e-3,
    "max_epochs": 500,
    "patience": 50,
}

# --- Input configurations ---
CONFIG_NAMES = ["A", "B", "C"]

# --- Random seed ---
RANDOM_SEED = 42
