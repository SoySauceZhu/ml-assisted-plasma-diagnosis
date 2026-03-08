import numpy as np
import torch
import shap
from sklearn.model_selection import LeaveOneOut

from phase3.evaluation import _scale_features, get_input_config
from phase3.tuner_mlp import MLPNetBN, _train_mlp
from .config import RANDOM_SEED


def compute_shap_loocv(oes_features, data, mlp_cfg):
    """Compute SHAP values for MLP Config C across LOOCV folds.

    For each fold:
    1. Scale features, assemble Config C (17 features).
    2. Train MLP on 19 samples.
    3. Create KernelExplainer with 19 training samples as background.
    4. Compute SHAP for the 1 held-out test sample (nsamples=200).

    Args:
        oes_features: (20, 13) engineered OES features
        data: dict from prepare_data()
        mlp_cfg: dict of tuned MLP hyperparameters

    Returns:
        shap_values: (20, 17) SHAP values, one row per held-out sample
        X_test_all: (20, 17) scaled feature values for each held-out sample
    """
    discharge = data["discharge_raw"]
    target = data["target"]
    loo = LeaveOneOut()

    shap_values_all = []
    X_test_all = []

    for fold_i, (train_idx, test_idx) in enumerate(loo.split(oes_features)):
        oes_tr_s, oes_te_s, dis_tr_s, dis_te_s = _scale_features(
            oes_features[train_idx], oes_features[test_idx],
            discharge[train_idx], discharge[test_idx],
        )
        X_train, X_test = get_input_config(
            "C", oes_tr_s, oes_te_s, dis_tr_s, dis_te_s
        )

        # Train MLP (reproducible)
        torch.manual_seed(RANDOM_SEED)
        input_dim = X_train.shape[1]
        net = MLPNetBN(
            input_dim, mlp_cfg["hidden_sizes"],
            mlp_cfg["dropout"], mlp_cfg.get("batch_norm", False),
        )
        _train_mlp(net, X_train, target[train_idx], mlp_cfg)
        net.eval()

        # Prediction wrapper for SHAP
        def predict_fn(X):
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32)
                return net(t).numpy().ravel()

        # KernelSHAP
        explainer = shap.KernelExplainer(predict_fn, X_train)
        sv = explainer.shap_values(np.atleast_2d(X_test), nsamples=200, silent=True)

        shap_values_all.append(sv[0] if isinstance(sv, list) else sv.ravel())
        X_test_all.append(X_test.ravel())

        print(f"    SHAP fold {fold_i + 1}/20 done")

    return np.array(shap_values_all), np.array(X_test_all)


def get_shap_importance(shap_values):
    """Mean absolute SHAP value per feature.

    Args:
        shap_values: (20, 17) SHAP values

    Returns:
        (17,) mean |SHAP| per feature
    """
    return np.mean(np.abs(shap_values), axis=0)
