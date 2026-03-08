import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from .config import (
    RANDOM_SEED, BOOTSTRAP_N_ITER, BOOTSTRAP_CI_LEVEL,
    MODELS_FOR_IMPORTANCE, ALL_FEATURE_NAMES_C,
)


def bootstrap_metrics(y_true, y_pred, n_iter=BOOTSTRAP_N_ITER,
                      ci=BOOTSTRAP_CI_LEVEL, seed=RANDOM_SEED):
    """Bootstrap resample (y_true, y_pred) pairs, compute R2 and RMSE.

    Skips resamples where y_true has near-zero variance (degenerate R2).

    Returns:
        dict with R2_mean, R2_lo, R2_hi, R2_std,
        RMSE_mean, RMSE_lo, RMSE_hi, RMSE_std, R2_distribution, n_skipped
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    r2_boot, rmse_boot = [], []
    n_skipped = 0

    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]

        if np.var(yt) < 1e-10:
            n_skipped += 1
            continue

        r2_boot.append(r2_score(yt, yp))
        rmse_boot.append(np.sqrt(mean_squared_error(yt, yp)))

    r2_boot = np.array(r2_boot)
    rmse_boot = np.array(rmse_boot)
    alpha = (1 - ci) / 2

    return {
        "R2_mean": np.mean(r2_boot),
        "R2_lo": np.percentile(r2_boot, 100 * alpha),
        "R2_hi": np.percentile(r2_boot, 100 * (1 - alpha)),
        "R2_std": np.std(r2_boot),
        "RMSE_mean": np.mean(rmse_boot),
        "RMSE_lo": np.percentile(rmse_boot, 100 * alpha),
        "RMSE_hi": np.percentile(rmse_boot, 100 * (1 - alpha)),
        "RMSE_std": np.std(rmse_boot),
        "R2_distribution": r2_boot.tolist(),
        "n_skipped": n_skipped,
    }


def bootstrap_all_models(predictions_path):
    """Run bootstrap for all model x config pairs (B and C).

    Args:
        predictions_path: Path to phase3_predictions_detail.csv

    Returns:
        DataFrame: Model, Config, R2_point, R2_lo95, R2_hi95,
                   RMSE_point, RMSE_lo95, RMSE_hi95
    """
    df = pd.read_csv(predictions_path)
    rows = []
    bootstrap_distributions = {}

    for model in MODELS_FOR_IMPORTANCE:
        for config in ["B", "C"]:
            mask = (df["Model"] == model) & (df["Config"] == config)
            sub = df[mask]
            if len(sub) == 0:
                continue

            y_true = sub["y_true"].values
            y_pred = sub["y_pred"].values
            point_r2 = r2_score(y_true, y_pred)
            point_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            result = bootstrap_metrics(y_true, y_pred)

            rows.append({
                "Model": model,
                "Config": config,
                "R2_point": point_r2,
                "R2_lo95": result["R2_lo"],
                "R2_hi95": result["R2_hi"],
                "R2_std": result["R2_std"],
                "RMSE_point": point_rmse,
                "RMSE_lo95": result["RMSE_lo"],
                "RMSE_hi95": result["RMSE_hi"],
                "RMSE_std": result["RMSE_std"],
                "n_skipped": result["n_skipped"],
            })
            bootstrap_distributions[(model, config)] = result["R2_distribution"]

    result_df = pd.DataFrame(rows)
    result_df.attrs["distributions"] = bootstrap_distributions
    return result_df


def fold_importance_stability(ridge_imp, pls_imp, rf_imp, shap_vals, feature_names):
    """Compute per-feature importance stability across 20 LOOCV folds.

    Args:
        ridge_imp: (20, 17) absolute Ridge coefficients
        pls_imp: (20, 17) PLS VIP scores
        rf_imp: (20, 17) RF permutation importance
        shap_vals: (20, 17) raw SHAP values (uses |SHAP| per fold)
        feature_names: list of 17 feature names

    Returns:
        DataFrame: feature, model, mean_importance, std_importance, cv, is_stable
    """
    model_data = {
        "Ridge": ridge_imp,
        "PLS": pls_imp,
        "RF": rf_imp,
        "MLP": np.abs(shap_vals),
    }

    rows = []
    for model_name, imp_array in model_data.items():
        for j, feat_name in enumerate(feature_names):
            col = imp_array[:, j]
            mean_val = col.mean()
            std_val = col.std()
            cv = std_val / mean_val if mean_val > 1e-12 else np.inf

            rows.append({
                "feature": feat_name,
                "model": model_name,
                "mean_importance": mean_val,
                "std_importance": std_val,
                "cv": cv,
                "is_stable": cv <= 1.0,
            })

    return pd.DataFrame(rows)
