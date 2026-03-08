import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import LeaveOneOut
from sklearn.inspection import permutation_importance

from phase1.models.ridge import RidgeModel
from phase1.models.pls import PLSModel
from phase1.models.rf import RFModel
from phase3.evaluation import _scale_features, get_input_config
from .config import RANDOM_SEED, RIDGE_ALPHAS, PLS_MAX_COMPONENTS, ALL_FEATURE_NAMES_C


def _loocv_importance_loop(oes_features, data, model_factory_fn, importance_extract_fn):
    """Generic LOOCV loop: fit model per fold, extract importance vector.

    Args:
        oes_features: (n, 13) engineered OES features
        data: dict from prepare_data()
        model_factory_fn: callable(X_train, y_train) -> fitted model wrapper
        importance_extract_fn: callable(fitted_model, X_train, y_train) -> (17,)

    Returns:
        np.ndarray of shape (n, 17) — importance per fold
    """
    discharge = data["discharge_raw"]
    target = data["target"]
    loo = LeaveOneOut()
    importances = []

    for train_idx, test_idx in loo.split(oes_features):
        oes_tr_s, oes_te_s, dis_tr_s, dis_te_s = _scale_features(
            oes_features[train_idx], oes_features[test_idx],
            discharge[train_idx], discharge[test_idx],
        )
        X_train, _ = get_input_config("C", oes_tr_s, oes_te_s, dis_tr_s, dis_te_s)

        model = model_factory_fn(X_train, target[train_idx])
        imp = importance_extract_fn(model, X_train, target[train_idx])
        importances.append(imp)

    return np.array(importances)


def _compute_vip(pls_model):
    """Compute VIP scores from a fitted sklearn PLSRegression.

    VIP_j = sqrt(p * sum_a(w_aj^2 * SS_a) / sum_a(SS_a))
    """
    t = pls_model.x_scores_       # (n_train, A)
    w = pls_model.x_weights_      # (p, A)
    q = pls_model.y_loadings_     # (1, A)
    p = w.shape[0]

    # SS per component: explained sum of squares
    ss = np.diag(t.T @ t) * (q ** 2).ravel()
    total_ss = ss.sum()

    if total_ss < 1e-12:
        return np.ones(p)

    vip = np.sqrt(p * (w ** 2 @ ss) / total_ss)
    return vip


def ridge_importance_loocv(oes_features, data):
    """Returns (20, 17) of |standardised Ridge coefficients| per LOOCV fold."""

    def factory(X_train, y_train):
        model = RidgeModel(alphas=RIDGE_ALPHAS)
        model.fit(X_train, y_train)
        return model

    def extract(model, X_train, y_train):
        return np.abs(model.model.coef_)

    return _loocv_importance_loop(oes_features, data, factory, extract)


def pls_importance_loocv(oes_features, data):
    """Returns (20, 17) of PLS VIP scores per LOOCV fold."""

    def factory(X_train, y_train):
        model = PLSModel(max_components=PLS_MAX_COMPONENTS)
        model.fit(X_train, y_train)
        return model

    def extract(model, X_train, y_train):
        return _compute_vip(model.model)

    return _loocv_importance_loop(oes_features, data, factory, extract)


def rf_importance_loocv(oes_features, data, rf_params):
    """Returns (20, 17) of RF permutation importance per LOOCV fold."""

    def factory(X_train, y_train):
        model = RFModel(params=rf_params)
        model.fit(X_train, y_train)
        return model

    def extract(model, X_train, y_train):
        perm = permutation_importance(
            model.model, X_train, y_train,
            n_repeats=10, random_state=RANDOM_SEED,
        )
        return perm.importances_mean

    return _loocv_importance_loop(oes_features, data, factory, extract)


def build_consensus_table(ridge_imp, pls_imp, rf_imp, mlp_imp, feature_names):
    """Build consensus importance table across 4 models.

    Args:
        ridge_imp: (20, 17) absolute Ridge coefficients per fold
        pls_imp: (20, 17) PLS VIP scores per fold
        rf_imp: (20, 17) RF permutation importance per fold
        mlp_imp: (17,) mean |SHAP| values
        feature_names: list of 17 feature names

    Returns:
        DataFrame with normalised importance, ranks, and consensus ranking
    """
    # Mean across folds
    ridge_mean = ridge_imp.mean(axis=0)
    pls_mean = pls_imp.mean(axis=0)
    rf_mean = rf_imp.mean(axis=0)
    mlp_mean = mlp_imp

    # Normalise each to sum to 1
    def normalise(arr):
        s = arr.sum()
        return arr / s if s > 0 else arr

    ridge_norm = normalise(ridge_mean)
    pls_norm = normalise(pls_mean)
    rf_norm = normalise(rf_mean)
    mlp_norm = normalise(mlp_mean)

    # Rank (1 = most important)
    def rank_desc(arr):
        return rankdata(-arr, method="min")

    ridge_rank = rank_desc(ridge_norm)
    pls_rank = rank_desc(pls_norm)
    rf_rank = rank_desc(rf_norm)
    mlp_rank = rank_desc(mlp_norm)

    mean_rank = (ridge_rank + pls_rank + rf_rank + mlp_rank) / 4.0
    consensus_rank = rankdata(mean_rank, method="min")

    df = pd.DataFrame({
        "feature": feature_names,
        "ridge_importance": ridge_norm,
        "ridge_rank": ridge_rank.astype(int),
        "pls_vip": pls_norm,
        "pls_rank": pls_rank.astype(int),
        "rf_perm_importance": rf_norm,
        "rf_rank": rf_rank.astype(int),
        "mlp_shap": mlp_norm,
        "mlp_rank": mlp_rank.astype(int),
        "mean_rank": mean_rank,
        "consensus_rank": consensus_rank.astype(int),
    })
    df = df.sort_values("consensus_rank").reset_index(drop=True)

    # Spearman rank correlations between model pairs
    models = {"Ridge": ridge_rank, "PLS": pls_rank, "RF": rf_rank, "MLP": mlp_rank}
    model_names = list(models.keys())
    spearman_pairs = {}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            rho, pval = spearmanr(models[model_names[i]], models[model_names[j]])
            spearman_pairs[f"{model_names[i]}_vs_{model_names[j]}"] = (rho, pval)

    # Store as metadata attribute
    df.attrs["spearman_correlations"] = spearman_pairs

    # OES vs discharge fractional importance
    n_oes = 13
    df.attrs["oes_fraction"] = {
        "Ridge": ridge_norm[:n_oes].sum(),
        "PLS": pls_norm[:n_oes].sum(),
        "RF": rf_norm[:n_oes].sum(),
        "MLP": mlp_norm[:n_oes].sum(),
    }
    df.attrs["discharge_fraction"] = {
        "Ridge": ridge_norm[n_oes:].sum(),
        "PLS": pls_norm[n_oes:].sum(),
        "RF": rf_norm[n_oes:].sum(),
        "MLP": mlp_norm[n_oes:].sum(),
    }

    return df
