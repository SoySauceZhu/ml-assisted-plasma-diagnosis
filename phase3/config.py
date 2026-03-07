from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

PHASE1_RESULTS_PATH = PROJECT_ROOT / "phase1" / "results" / "tables" / "loocv_results_summary.csv"
PHASE2_RESULTS_PATH = PROJECT_ROOT / "phase2" / "results" / "tables" / "phase2_loocv_results_summary.csv"

# --- Feature definitions ---
# Single wavelengths: (feature_name, wavelength_nm)
SINGLE_WAVELENGTHS = [
    ("I_309_OH", 309),
    ("I_777_O", 777),
    ("I_656_Ha", 656),
    ("I_486_Hb", 486),
    ("I_337_N2", 337),
    ("I_406_CO2p", 406),
    ("I_516_C2", 516),
]

# Band integrals: (feature_name, start_nm, end_nm)
BAND_INTEGRALS = [
    ("band_OH_306_312", 306, 312),
    ("band_CO2p_398_412", 398, 412),
    ("band_CO_Hb_460_500", 460, 500),
]

# Intensity ratios: (feature_name, numerator_nm, denominator_nm)
INTENSITY_RATIOS = [
    ("ratio_309_656", 309, 656),
    ("ratio_777_309", 777, 309),
    ("ratio_656_486", 656, 486),
]

RATIO_EPSILON = 1e-10

# --- Random seed ---
RANDOM_SEED = 42

# --- Model hyperparameters (initial) ---
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
PLS_MAX_COMPONENTS = 10

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_split": 3,
    "min_samples_leaf": 2,
    "max_features": 0.8,
    "bootstrap": False,
}

MLP_CONFIG = {
    "hidden_sizes": [16],
    "dropout": 0.4,
    "weight_decay": 0.01,
    "lr": 0.004,
    "max_epochs": 500,
    "patience": 50,
    "batch_norm": True,
}

# --- Tuning settings ---
N_TRIALS_RF = 200
N_TRIALS_MLP = 100

# --- Model x Config applicability ---
MODEL_CONFIGS = {
    "Ridge": ["A", "B", "C"],
    "PLS": ["A", "B", "C"],
    "RF": ["A", "B", "C"],
    "MLP": ["A", "B", "C"],
}

MODEL_NAMES = ["Ridge", "PLS", "RF", "MLP"]
CONFIG_NAMES = ["A", "B", "C"]
