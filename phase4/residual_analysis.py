import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .config import ALL_FEATURE_NAMES_C


def analyse_residuals(predictions_path, data):
    """Analyse residual patterns for Config C models (Ridge, MLP).

    Args:
        predictions_path: Path to phase3_predictions_detail.csv
        data: dict from prepare_data() (for sample_info)

    Returns:
        DataFrame: Model, Sample, y_true, y_pred, residual, abs_residual,
                   condition, sheet, is_outlier
    """
    df = pd.read_csv(predictions_path)
    sample_info = data["sample_info"]

    # Filter to Config C, Ridge and MLP
    mask = (df["Config"] == "C") & (df["Model"].isin(["Ridge", "MLP"]))
    sub = df[mask].copy()

    sub["residual"] = sub["y_pred"] - sub["y_true"]
    sub["abs_residual"] = sub["residual"].abs()

    # Merge sample_info by Sample index
    sub["condition"] = sub["Sample"].map(
        dict(enumerate(sample_info["condition"].values))
    )
    sub["sheet"] = sub["Sample"].map(
        dict(enumerate(sample_info["sheet"].values))
    )

    # Flag outliers per model (|residual| > 2 * std)
    sub["is_outlier"] = False
    for model_name in ["Ridge", "MLP"]:
        model_mask = sub["Model"] == model_name
        std_resid = sub.loc[model_mask, "residual"].std()
        outlier_mask = model_mask & (sub["abs_residual"] > 2 * std_resid)
        sub.loc[outlier_mask, "is_outlier"] = True

    return sub.reset_index(drop=True)


def residual_feature_correlation(residual_df, oes_features, discharge, feature_names):
    """Correlate |residuals| with each feature to detect model weaknesses.

    Args:
        residual_df: output from analyse_residuals()
        oes_features: (20, 13) OES feature matrix
        discharge: (20, 4) discharge parameter matrix
        feature_names: list of 17 feature names (Config C order)

    Returns:
        DataFrame: Model, feature, correlation, p_value
    """
    all_features = np.hstack([oes_features, discharge])  # (20, 17)
    rows = []

    for model_name in ["Ridge", "MLP"]:
        model_mask = residual_df["Model"] == model_name
        abs_resid = residual_df.loc[model_mask, "abs_residual"].values

        for j, feat_name in enumerate(feature_names):
            feat_vals = all_features[:, j]
            corr, pval = pearsonr(abs_resid, feat_vals)
            rows.append({
                "Model": model_name,
                "feature": feat_name,
                "correlation": corr,
                "p_value": pval,
            })

    return pd.DataFrame(rows)


def condition_grouped_summary(residual_df):
    """Group residuals by experimental condition.

    Returns:
        DataFrame: Model, condition, n_samples, mean_residual,
                   std_residual, mean_abs_residual
    """
    grouped = residual_df.groupby(["Model", "condition"]).agg(
        n_samples=("residual", "count"),
        mean_residual=("residual", "mean"),
        std_residual=("residual", "std"),
        mean_abs_residual=("abs_residual", "mean"),
    ).reset_index()

    return grouped
