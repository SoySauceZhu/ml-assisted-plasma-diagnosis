import numpy as np

from .config import SINGLE_WAVELENGTHS, BAND_INTEGRALS, INTENSITY_RATIOS, RATIO_EPSILON


def extract_oes_features(oes_raw, wavelengths):
    """Extract 13 domain-knowledge features from baseline-corrected OES spectra.

    Args:
        oes_raw: (n_samples, 701) baseline-corrected OES intensities
        wavelengths: (701,) array of wavelength values (200-900 nm, integers)

    Returns:
        features: (n_samples, 13) engineered feature matrix
        feature_names: list of 13 feature name strings
    """
    n_samples = oes_raw.shape[0]
    features = []
    feature_names = []

    # Build wavelength-to-index mapping
    wl_to_idx = {int(w): i for i, w in enumerate(wavelengths)}

    # Category 1: Single-wavelength intensities
    single_values = {}
    for name, wl in SINGLE_WAVELENGTHS:
        idx = wl_to_idx[wl]
        col = oes_raw[:, idx]
        features.append(col)
        feature_names.append(name)
        single_values[wl] = col

    # Category 2: Band integrals (trapezoidal integration)
    for name, start_nm, end_nm in BAND_INTEGRALS:
        start_idx = wl_to_idx[start_nm]
        end_idx = wl_to_idx[end_nm]
        band_wl = wavelengths[start_idx:end_idx + 1]
        band_data = oes_raw[:, start_idx:end_idx + 1]
        integral = np.trapezoid(band_data, band_wl, axis=1)
        features.append(integral)
        feature_names.append(name)

    # Category 3: Intensity ratios
    for name, num_wl, den_wl in INTENSITY_RATIOS:
        numerator = single_values[num_wl]
        denominator = single_values[den_wl]
        ratio = numerator / (denominator + RATIO_EPSILON)
        features.append(ratio)
        feature_names.append(name)

    features = np.column_stack(features)
    return features, feature_names
