from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE4_DIR = PROJECT_ROOT / "phase4"
RESULTS_DIR = PHASE4_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

PHASE3_TUNED_PARAMS_PATH = (
    PROJECT_ROOT / "phase3" / "results" / "tables" / "tuned_hyperparameters.json"
)
PHASE3_PREDICTIONS_PATH = (
    PROJECT_ROOT / "phase3" / "results" / "tables" / "phase3_predictions_detail.csv"
)

# --- Random seed ---
RANDOM_SEED = 42

# --- Bootstrap ---
BOOTSTRAP_N_ITER = 500
BOOTSTRAP_CI_LEVEL = 0.95

# --- Model hyperparameters (same as phase3) ---
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
PLS_MAX_COMPONENTS = 10

# --- Feature names (must match phase3.feature_engineer output order) ---
OES_FEATURE_NAMES = [
    "I_309_OH", "I_777_O", "I_656_Ha", "I_486_Hb", "I_337_N2",
    "I_406_CO2p", "I_516_C2",
    "band_OH_306_312", "band_CO2p_398_412", "band_CO_Hb_460_500",
    "ratio_309_656", "ratio_777_309", "ratio_656_486",
]
DISCHARGE_FEATURE_NAMES = [
    "frequency_hz", "pulse_width_ns", "rise_time_ns", "flow_rate_sccm",
]
ALL_FEATURE_NAMES_C = OES_FEATURE_NAMES + DISCHARGE_FEATURE_NAMES  # 17 total

# --- Analysis focus ---
MODELS_FOR_IMPORTANCE = ["Ridge", "PLS", "RF", "MLP"]
FOCUS_CONFIG = "C"

# --- Plotting ---
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
