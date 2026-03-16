"""
LOOCV Evaluation Framework
===========================
Runs Leave-One-Out Cross-Validation for all model x configuration combinations.
For each of the 20 folds:
  1. Splits data into 19 training + 1 test sample
  2. Fits StandardScaler and PCA on training data only (prevents data leakage)
  3. Assembles the correct input configuration (A/B/C)
  4. Trains the model and predicts the held-out sample
  5. Collects all 20 predictions for final metric computation

LOOCV is chosen because with only n=20 samples, k-fold CV would have too few
samples per fold. LOOCV maximises training data per fold (19/20 = 95%).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .config import RANDOM_SEED, TABLES_DIR
from .models.ridge import RidgeModel
from .models.pls import PLSModel
from .models.svr import SVRModel
from .models.xgboost_model import XGBoostModel
from .models.mlp import MLPModel
from .models.cnn import CNNModel
from .models.rf import RFModel


def compute_metrics(y_true, y_pred):

    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def _scale_and_pca(oes_train, oes_test, discharge_train, discharge_test, pca_k):
    """Fit scalers and PCA on training data, transform both train and test.

    This function is called inside each LOOCV fold to ensure no data leakage.
    Scale OES and discharge features separately using StandardScaler (mean = 0, sigma = 1).

    Args:
        oes_train: Training OES data (19, 701).
        oes_test: Test OES data (1, 701).
        discharge_train: Training discharge params (19, 4).
        discharge_test: Test discharge params (1, 4).
        pca_k: Number of PCA components to retain (11 from Phase 1 analysis).

    Returns:
        tuple of 6 arrays:
            - pca_train (19, k): PCA-transformed training OES
            - pca_test (1, k): PCA-transformed test OES
            - dis_train_s (19, 4): Scaled training discharge params
            - dis_test_s (1, 4): Scaled test discharge params
            - oes_train_s (19, 701): Scaled training OES (for CNN raw input)
            - oes_test_s (1, 701): Scaled test OES (for CNN raw input)
    """
    oes_scaler = StandardScaler()
    oes_train_s = oes_scaler.fit_transform(oes_train)
    oes_test_s = oes_scaler.transform(oes_test)

    dis_scaler = StandardScaler()
    dis_train_s = dis_scaler.fit_transform(discharge_train)
    dis_test_s = dis_scaler.transform(discharge_test)

    pca = PCA(n_components=pca_k)
    pca_train = pca.fit_transform(oes_train_s)
    pca_test = pca.transform(oes_test_s)

    return pca_train, pca_test, dis_train_s, dis_test_s, oes_train_s, oes_test_s


def get_input_config(config_name, pca_train, pca_test, dis_train, dis_test,
                     oes_train_s=None, oes_test_s=None, is_cnn=False):
    """Assemble X_train and X_test based on the input configuration.

    CNN uses raw OES (not PCA) because 1D convolution is designed to process
    sequential spectral data directly. For Config C, CNN uses a two-input
    architecture: OES through conv layers, discharge concatenated before the output.

    Args:
        config_name: "A", "B", or "C".
        pca_train, pca_test: PCA-transformed OES (for non-CNN models).
        dis_train, dis_test: Scaled discharge parameters.
        oes_train_s, oes_test_s: Scaled raw OES (for CNN).
        is_cnn: If True, returns raw OES instead of PCA, with separate extra features.

    Returns:
        For non-CNN: (X_train - 19rows, X_test - 1row) — single feature matrix per set.
        For CNN Config A: (oes_train, oes_test, None, None)
        For CNN Config C: (oes_train, oes_test, dis_train, dis_test) // Two type inputs will be sent to model at difference layer.
    """
    if config_name == "A":
        if is_cnn:
            return oes_train_s, oes_test_s, None, None
        return pca_train, pca_test
    elif config_name == "B":
        return dis_train, dis_test
    elif config_name == "C":
        if is_cnn:
            return oes_train_s, oes_test_s, dis_train, dis_test
        return (np.hstack([pca_train, dis_train]),
                np.hstack([pca_test, dis_test]))


def run_loocv_for_model(model_name, data, pca_k, config_name):
    """Fit model, and run LOOCV test

    Iterates through all 20 LOOCV folds. In each fold, one sample is held out
    as test, the remaining 19 are used for training. Scaling and PCA are fitted
    fresh in each fold to prevent data leakage.

    Args:
        model_name: One of "Ridge", "PLS", "SVR", "XGBoost", "RF", "MLP", "CNN".
        data: Data dict from prepare_data().
        pca_k: Number of PCA components (11).
        config_name: "A", "B", or "C".

    Returns:
        dict with keys: "model", "config", "R2", "RMSE", "MAE",
                        "y_true" (array of 20 actual values),
                        "y_pred" (array of 20 predicted values).
    """
    oes = data["oes_raw"]
    discharge = data["discharge_raw"]
    target = data["target"]
    n = len(target)
    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []
    is_cnn = model_name == "CNN"

    for train_idx, test_idx in loo.split(oes):
        # train_ids: 19, test_idx: 1
        oes_train, oes_test = oes[train_idx], oes[test_idx]
        dis_train, dis_test = discharge[train_idx], discharge[test_idx]
        y_train, y_test = target[train_idx], target[test_idx]

        # Scale and apply PCA within the fold (no data leakage)
        pca_tr, pca_te, dis_tr_s, dis_te_s, oes_tr_s, oes_te_s = _scale_and_pca(
            oes_train, oes_test, dis_train, dis_test, pca_k
        )

        if is_cnn:
            # CNN uses raw OES input (not PCA) with optional extra discharge features
            result = get_input_config(config_name, pca_tr, pca_te, dis_tr_s, dis_te_s,
                                       oes_tr_s, oes_te_s, is_cnn=True)
            oes_in_tr, oes_in_te, extra_tr, extra_te = result
            model = _create_model(model_name)
            model.fit(oes_in_tr, y_train, extra_tr)
            pred = model.predict(oes_in_te, extra_te)
        else:
            # Non-CNN models use PCA-reduced features or discharge params
            if config_name == "B":
                X_train, X_test = dis_tr_s, dis_te_s
            else:
                result = get_input_config(config_name, pca_tr, pca_te, dis_tr_s, dis_te_s)
                X_train, X_test = result
            model = _create_model(model_name)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        y_true_all.append(y_test[0])
        y_pred_all.append(pred[0])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    metrics = compute_metrics(y_true_all, y_pred_all)

    return {
        "model": model_name,
        "config": config_name,
        **metrics,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }


def _create_model(model_name):
    """
        Factories
    """
    factories = {
        "Ridge": RidgeModel,
        "PLS": PLSModel,
        "SVR": SVRModel,
        "XGBoost": XGBoostModel,
        "MLP": MLPModel,
        "CNN": CNNModel,
        "RF": RFModel,
    }
    return factories[model_name]()


def run_all_evaluations(data, pca_k):
    """Run LOOCV for all 7 models x 3 configurations (20 combinations).

    Skips CNN Config B (CNN needs sequential spectral data, not 4 scalar params).
    Saves two CSV files:
      - loocv_results_summary.csv: R2/RMSE/MAE per model-config
      - loocv_predictions_detail.csv: per-sample y_true and y_pred

    Args:
        data: Data dict from prepare_data().
        pca_k: Number of PCA components.

    Returns:
        tuple: (list of result dicts, summary pd.DataFrame)
    """
    model_names = ["Ridge", "PLS", "SVR", "XGBoost", "RF", "MLP", "CNN"]
    config_names = ["A", "B", "C"]
    all_results = []

    for mname in model_names:
        for cname in config_names:
            # Skip CNN Config B: CNN processes raw OES spectra, not scalar discharge params
            if mname == "CNN" and cname == "B":
                print(f"  Skipping {mname} Config {cname} (not applicable)")
                continue
            print(f"  Running {mname} Config {cname}...")

            result = run_loocv_for_model(mname, data, pca_k, cname)

            print(f"    R2={result['R2']:.3f}  RMSE={result['RMSE']:.3f}  MAE={result['MAE']:.3f}")
            all_results.append(result)

    # Build summary table
    summary_rows = [{
        "Model": r["model"],
        "Config": r["config"],
        "R2": r["R2"],
        "RMSE": r["RMSE"],
        "MAE": r["MAE"],
    } for r in all_results]
    results_df = pd.DataFrame(summary_rows)

    # Save summary CSV
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TABLES_DIR / "loocv_results_summary.csv", index=False)

    # Save per-sample prediction detail CSV (used by Phase 4 for residual analysis)
    detail_rows = []
    for r in all_results:
        for i, (yt, yp) in enumerate(zip(r["y_true"], r["y_pred"])):
            detail_rows.append({
                "Model": r["model"], "Config": r["config"],
                "Sample": i, "y_true": yt, "y_pred": yp,
            })
    pd.DataFrame(detail_rows).to_csv(TABLES_DIR / "loocv_predictions_detail.csv", index=False)

    return all_results, results_df
