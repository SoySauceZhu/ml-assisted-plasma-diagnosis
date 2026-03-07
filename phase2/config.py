from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

PHASE1_RESULTS_PATH = PROJECT_ROOT / "phase1" / "results" / "tables" / "loocv_results_summary.csv"

# --- PCA ---
PCA_K = 11  # From Phase 1 analysis (95% variance threshold)

# --- Tuning settings ---
N_TRIALS_RF = 200
N_TRIALS_MLP = 100
N_TRIALS_CNN = 100
RANDOM_SEED = 42

# --- Search space references (actual suggest calls in tuner files) ---
RF_SEARCH_SPACE = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_split": [2, 3, 4, 5],
    "min_samples_leaf": [1, 2, 3],
    "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
    "bootstrap": [True, False],
}

MLP_SEARCH_SPACE = {
    "hidden_sizes": [[8], [16], [32], [8, 4], [16, 8], [32, 16]],
    "dropout": (0.3, 0.7),
    "weight_decay": (1e-3, 1e-1),
    "lr": (1e-4, 5e-3),
    "max_epochs": [200, 500, 1000],
    "patience": [30, 50, 100],
    "batch_norm": [True, False],
}

CNN_SEARCH_SPACE = {
    "conv_channels": [[8], [16], [8, 16], [16, 32], [32, 64]],
    "kernel_size": [3, 5, 7, 11, 15, 21],
    "dropout": (0.3, 0.6),
    "weight_decay": (1e-3, 5e-2),
    "lr": (1e-4, 1e-3),
    "max_epochs": [300, 500, 1000],
    "patience": [30, 50, 100],
    "pool_type": ["avg", "max"],
    "fc_hidden": [None, 8, 16],
}

# --- Model x Config applicability ---
MODEL_CONFIGS = {
    "RF": ["A", "B", "C"],
    "MLP": ["A", "B", "C"],
    "CNN": ["A", "C"],
}
