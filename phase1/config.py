"""
Phase 1 Configuration
=====================
Central configuration file for the baseline ML pipeline. All hyperparameters,
file paths, column names, and constants are defined here so that changes
propagate consistently across the entire Phase 1 codebase.
"""

from pathlib import Path

# --- Paths ---
# PROJECT_ROOT points to the parent of phase1/, i.e., the main project directory
PROJECT_ROOT = Path(__file__).parent.parent
# Path to the raw OES + discharge + H2O2 dataset (20 samples, 701 wavelengths)
DATA_PATH = PROJECT_ROOT / "oes_ml_dataset_1nm.csv"
# Output directories for results (figures and CSV tables)
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# --- Column definitions ---
# Metadata columns (not used as features): sheet ID and experimental condition label
META_COLS = ["sheet", "condition"]
# The 4 discharge parameters that control the plasma: frequency, pulse width, rise time, gas flow rate
DISCHARGE_COLS = ["frequency_hz", "pulse_width_ns", "rise_time_ns", "flow_rate_sccm"]
# Prediction target: H2O2 yield rate (range: 0.02 to 0.83)
TARGET_COL = "h2o2_rate"
# All OES wavelength columns start with this prefix (I_200, I_201, ..., I_900)
OES_PREFIX = "I_"

# --- PCA ---
# Cumulative explained variance threshold for selecting the number of PCA components.
# 0.95 means we keep enough components to explain 95% of total OES variance.
# Result: k=11 components needed (out of 19 possible with n=20 samples).
PCA_VARIANCE_THRESHOLD = 0.95

# --- Model hyperparameters ---
# Ridge regression: candidate regularisation strengths for RidgeCV's internal LOOCV alpha selection
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# PLS: maximum number of latent components to try (optimal k selected via inner LOOCV)
PLS_MAX_COMPONENTS = 10

# SVR: grid search space for RBF kernel SVR (C=regularisation, epsilon=tube width, gamma=kernel width)
SVR_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0],
    "epsilon": [0.01, 0.1, 0.5],
    "gamma": ["scale", "auto"],
}

# XGBoost: conservative parameters to mitigate overfitting on n=20
# (shallow trees, heavy regularisation, subsampling)
XGBOOST_PARAMS = {
    "max_depth": 2,            # very shallow trees to prevent overfitting
    "n_estimators": 50,        # few trees
    "learning_rate": 0.05,     # slow learning rate
    "reg_alpha": 10.0,         # L1 regularisation
    "reg_lambda": 10.0,        # L2 regularisation
    "subsample": 0.8,          # row subsampling per tree
    "colsample_bytree": 0.8,   # feature subsampling per tree
}

# Random Forest: moderate complexity to avoid overfitting
RF_PARAMS = {
    "n_estimators": 100,       # number of trees in the forest
    "max_depth": 3,            # shallow trees
    "min_samples_split": 5,    # need at least 5 samples to split a node (25% of data)
    "min_samples_leaf": 2,     # each leaf must contain at least 2 samples
}

# MLP (PyTorch): fully-connected neural network configuration
MLP_CONFIG = {
    "hidden_sizes": [32, 16],  # two hidden layers with 32 and 16 neurons
    "dropout": 0.4,            # 40% dropout for regularisation
    "weight_decay": 1e-2,      # L2 weight penalty in Adam optimiser
    "lr": 1e-3,                # learning rate
    "max_epochs": 500,         # maximum training epochs
    "patience": 50,            # early stopping: stop if no improvement for 50 epochs
}

# CNN (PyTorch): 1D convolutional network for processing raw 701-point OES spectra
CNN_CONFIG = {
    "conv_channels": [16, 32], # two conv layers with 16 and 32 output channels
    "kernel_size": 7,          # convolution kernel width (7 nm spectral window)
    "dropout": 0.4,            # dropout rate
    "weight_decay": 1e-2,      # L2 weight penalty
    "lr": 1e-3,                # learning rate
    "max_epochs": 500,         # maximum training epochs
    "patience": 50,            # early stopping patience
}

# --- Input configurations ---
# A = OES features only, B = discharge parameters only, C = OES + discharge combined
CONFIG_NAMES = ["A", "B", "C"]

# --- Random seed ---
# Fixed seed for reproducibility across numpy, torch, and sklearn
RANDOM_SEED = 42
