import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from phase1.models.ridge import RidgeModel
from phase1.evaluation import compute_metrics
from .config import RIDGE_ALPHAS, OES_FEATURE_NAMES


def compute_correlation_vif(oes_features, feature_names):
    """Pearson correlation matrix + VIF for 13 OES features.

    Args:
        oes_features: (20, 13) OES feature matrix
        feature_names: list of 13 OES feature names

    Returns:
        (corr_df, vif_df)
        - corr_df: 13x13 correlation DataFrame
        - vif_df: DataFrame with columns [feature, VIF, is_high_vif]
    """
    df = pd.DataFrame(oes_features, columns=feature_names)
    corr_df = df.corr()

    # VIF on standardised features
    X_scaled = StandardScaler().fit_transform(oes_features)
    vif_values = []
    for i in range(X_scaled.shape[1]):
        vif_values.append(variance_inflation_factor(X_scaled, i))

    vif_df = pd.DataFrame({
        "feature": feature_names,
        "VIF": vif_values,
        "is_high_vif": [v > 10 for v in vif_values],
    })

    return corr_df, vif_df


def _run_ridge_loocv_subset(oes_subset, data):
    """Run Ridge LOOCV on Config C with a subset of OES features.

    Manually constructs X = hstack([oes_subset_scaled, discharge_scaled])
    since get_input_config expects the full 13-feature OES matrix.

    Args:
        oes_subset: (20, k) where k <= 13, the OES feature subset
        data: dict from prepare_data()

    Returns:
        dict with R2, RMSE, MAE
    """
    discharge = data["discharge_raw"]
    target = data["target"]
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(oes_subset):
        # Scale OES subset
        oes_scaler = StandardScaler()
        oes_tr_s = oes_scaler.fit_transform(oes_subset[train_idx])
        oes_te_s = oes_scaler.transform(oes_subset[test_idx])

        # Scale discharge (always all 4)
        dis_scaler = StandardScaler()
        dis_tr_s = dis_scaler.fit_transform(discharge[train_idx])
        dis_te_s = dis_scaler.transform(discharge[test_idx])

        # Config C: hstack
        X_train = np.hstack([oes_tr_s, dis_tr_s])
        X_test = np.hstack([oes_te_s, dis_te_s])

        model = RidgeModel(alphas=RIDGE_ALPHAS)
        model.fit(X_train, target[train_idx])
        pred = model.predict(X_test)

        y_true_all.append(target[test_idx][0])
        y_pred_all.append(pred[0])

    return compute_metrics(np.array(y_true_all), np.array(y_pred_all))


def ablation_backward_elimination(oes_features, data, feature_names):
    """Backward elimination: remove least important OES feature one at a time.

    All 4 discharge params are always included. Starting from 13 OES features,
    removes one per step (least important by mean |Ridge coef| on the OES portion),
    re-evaluates, repeats until 3 OES features remain.

    Args:
        oes_features: (20, 13) OES feature matrix
        data: dict from prepare_data()
        feature_names: list of 13 OES feature names

    Returns:
        DataFrame: n_oes_features, removed_feature, remaining_features, R2, RMSE, MAE
    """
    remaining_idx = list(range(len(feature_names)))
    results = []

    # Baseline with all 13
    baseline = _run_ridge_loocv_subset(oes_features, data)
    results.append({
        "n_oes_features": len(remaining_idx),
        "removed_feature": None,
        "remaining_features": ",".join([feature_names[i] for i in remaining_idx]),
        "R2": baseline["R2"],
        "RMSE": baseline["RMSE"],
        "MAE": baseline["MAE"],
    })

    while len(remaining_idx) > 3:
        # Get Ridge coefficients on current subset to find least important OES feature
        oes_subset = oes_features[:, remaining_idx]
        discharge = data["discharge_raw"]
        target = data["target"]

        # Fit Ridge on full dataset to get coefficient magnitudes
        oes_scaler = StandardScaler()
        dis_scaler = StandardScaler()
        oes_s = oes_scaler.fit_transform(oes_subset)
        dis_s = dis_scaler.fit_transform(discharge)
        X_full = np.hstack([oes_s, dis_s])
        model = RidgeModel(alphas=RIDGE_ALPHAS)
        model.fit(X_full, target)

        # OES coefficients are the first k entries
        n_oes = len(remaining_idx)
        oes_coefs = np.abs(model.model.coef_[:n_oes])

        # Remove the least important OES feature
        least_idx_local = np.argmin(oes_coefs)
        removed_global = remaining_idx[least_idx_local]
        remaining_idx.remove(removed_global)

        # Evaluate with reduced subset
        oes_subset_new = oes_features[:, remaining_idx]
        metrics = _run_ridge_loocv_subset(oes_subset_new, data)

        results.append({
            "n_oes_features": len(remaining_idx),
            "removed_feature": feature_names[removed_global],
            "remaining_features": ",".join([feature_names[i] for i in remaining_idx]),
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
        })

    return pd.DataFrame(results)


def ablation_category(oes_features, data, feature_names):
    """Category-based ablation: evaluate Config C with OES feature subsets by type.

    Categories:
    - All 13 OES (baseline)
    - Single-wavelength only (indices 0:7)
    - Band integrals only (indices 7:10)
    - Ratios only (indices 10:13)

    Args:
        oes_features: (20, 13) OES feature matrix
        data: dict from prepare_data()
        feature_names: list of 13 OES feature names

    Returns:
        DataFrame: category, n_oes_features, n_total_features, R2, RMSE, MAE
    """
    categories = {
        "All OES (13)": list(range(13)),
        "Single-wavelength (7)": list(range(7)),
        "Band integrals (3)": list(range(7, 10)),
        "Ratios (3)": list(range(10, 13)),
    }

    rows = []
    for cat_name, indices in categories.items():
        oes_subset = oes_features[:, indices]
        metrics = _run_ridge_loocv_subset(oes_subset, data)
        rows.append({
            "category": cat_name,
            "n_oes_features": len(indices),
            "n_total_features": len(indices) + 4,  # + 4 discharge
            "features": ",".join([feature_names[i] for i in indices]),
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
        })

    return pd.DataFrame(rows)
