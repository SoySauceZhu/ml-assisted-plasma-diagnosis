import numpy as np
import pandas as pd
from .config import DATA_PATH, META_COLS, DISCHARGE_COLS, TARGET_COL, OES_PREFIX


def load_dataset(csv_path=None):
    df = pd.read_csv(csv_path or DATA_PATH)
    return df


def separate_features(df):
    oes_cols = [c for c in df.columns if c.startswith(OES_PREFIX)]
    oes_df = df[oes_cols]
    discharge_df = df[DISCHARGE_COLS]
    target = df[TARGET_COL]
    return oes_df, discharge_df, target


def baseline_correction(oes_df, df):
    """Subtract the spectrum of the pulse_width=0 ns sample (near-zero H2O2 production)."""
    baseline_mask = df["pulse_width_ns"] == 0.0
    if baseline_mask.sum() == 0:
        print("Warning: no pulse_width=0 baseline sample found, skipping baseline correction")
        return oes_df
    baseline_spectrum = oes_df.loc[baseline_mask].values.mean(axis=0)
    corrected = oes_df.values - baseline_spectrum
    return pd.DataFrame(corrected, columns=oes_df.columns, index=oes_df.index)


def prepare_data(csv_path=None):
    df = load_dataset(csv_path)
    oes_df, discharge_df, target = separate_features(df)
    oes_df = baseline_correction(oes_df, df)

    oes_cols = [c for c in oes_df.columns]
    wavelengths = np.array([int(c.replace(OES_PREFIX, "")) for c in oes_cols])

    return {
        "oes_raw": oes_df.values.astype(np.float64),
        "discharge_raw": discharge_df.values.astype(np.float64),
        "target": target.values.astype(np.float64),
        "wavelengths": wavelengths,
        "sample_info": df[META_COLS].copy(),
    }
