"""
Data Loader
============
Loads the raw CSV dataset and preprocesses it into separate feature groups:
  - OES (701 wavelength intensities, 200-900 nm at 1 nm resolution)
  - Discharge parameters (4 experimental control variables)
  - Target (H2O2 yield rate)
Applies baseline correction by subtracting the spectrum at pulse_width=0 ns.
"""

import numpy as np
import pandas as pd
from .config import DATA_PATH, META_COLS, DISCHARGE_COLS, TARGET_COL, OES_PREFIX


def load_dataset(csv_path=None):
    """Load the CSV dataset from disk.

    Args:
        csv_path: Path to CSV file. If None, uses the default DATA_PATH from config.

    Returns:
        pd.DataFrame: Full dataset with all columns (metadata, OES, discharge, target).
    """
    df = pd.read_csv(csv_path or DATA_PATH)
    return df


def separate_features(df):
    """Split the DataFrame into three feature groups.

    Identifies OES columns by their "I_" prefix (I_200, I_201, ..., I_900),
    selects the 4 discharge parameter columns, and extracts the H2O2 target.

    Args:
        df: Full DataFrame from load_dataset().

    Returns:
        tuple: (oes_df [n x 701], discharge_df [n x 4], target [n,])
    """
    oes_cols = [c for c in df.columns if c.startswith(OES_PREFIX)]
    oes_df = df[oes_cols]
    discharge_df = df[DISCHARGE_COLS]
    target = df[TARGET_COL]
    return oes_df, discharge_df, target


def baseline_correction(oes_df, df):
    """Subtract the baseline spectrum to remove background emission.

    pulse_width=0 ns means no plasma discharge, so the OES spectrum at that condition represents the background emission. 

    Args:
        oes_df: OES intensity DataFrame (n_samples x 701 wavelengths).
        df: Full DataFrame (needed to access pulse_width_ns column for baseline identification).

    Returns:
        pd.DataFrame: Baseline-corrected OES intensities. Returns original if no baseline found.
    """
    baseline_mask = df["pulse_width_ns"] == 0.0
    if baseline_mask.sum() == 0:
        print("Warning: no pulse_width=0 baseline sample found, skipping baseline correction")
        return oes_df
    baseline_spectrum = oes_df.loc[baseline_mask].values.mean(axis=0)
    corrected = oes_df.values - baseline_spectrum
    return pd.DataFrame(corrected, columns=oes_df.columns, index=oes_df.index)


def prepare_data(csv_path=None):
    """General data preparation function. Loads, separates, and preprocesses all data.

    Args:
        csv_path: Optional path to CSV file (default: DATA_PATH from config).

    Returns:
        dict with keys:
            - "oes_raw": np.ndarray (n_samples, 701) — baseline-corrected OES intensities
            - "discharge_raw": np.ndarray (n_samples, 4) — discharge parameters
            - "target": np.ndarray (n_samples,) — H2O2 yield rates
            - "wavelengths": np.ndarray (701,) — integer wavelengths [200, 201, ..., 900]
            - "sample_info": pd.DataFrame — metadata columns (sheet, condition)
    """
    df = load_dataset(csv_path)
    oes_df, discharge_df, target = separate_features(df)
    oes_df = baseline_correction(oes_df, df)

    oes_cols = [c for c in oes_df.columns]
    # Extract integer wavelengths from column names: "I_309" -> 309
    wavelengths = np.array([int(c.replace(OES_PREFIX, "")) for c in oes_cols])

    return {
        "oes_raw": oes_df.values.astype(np.float64),
        "discharge_raw": discharge_df.values.astype(np.float64),
        "target": target.values.astype(np.float64),
        "wavelengths": wavelengths,
        "sample_info": df[META_COLS].copy(),
    }
