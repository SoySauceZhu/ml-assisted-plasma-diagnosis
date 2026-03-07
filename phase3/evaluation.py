import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from phase1.evaluation import compute_metrics
from phase1.models.ridge import RidgeModel
from phase1.models.pls import PLSModel
from phase1.models.rf import RFModel
from .config import (
    RANDOM_SEED, TABLES_DIR, PHASE1_RESULTS_PATH, PHASE2_RESULTS_PATH,
    MODEL_CONFIGS, MODEL_NAMES, CONFIG_NAMES,
    RIDGE_ALPHAS, PLS_MAX_COMPONENTS, RF_PARAMS, MLP_CONFIG,
)


def _scale_features(oes_train, oes_test, dis_train, dis_test):
    """Fit scalers on training data, transform both train and test.

    Args:
        oes_train: (n_train, 13) engineered OES features
        oes_test: (n_test, 13) engineered OES features
        dis_train: (n_train, 4) discharge parameters
        dis_test: (n_test, 4) discharge parameters

    Returns:
        (oes_tr_s, oes_te_s, dis_tr_s, dis_te_s) — all StandardScaler-transformed
    """
    oes_scaler = StandardScaler()
    oes_tr_s = oes_scaler.fit_transform(oes_train)
    oes_te_s = oes_scaler.transform(oes_test)

    dis_scaler = StandardScaler()
    dis_tr_s = dis_scaler.fit_transform(dis_train)
    dis_te_s = dis_scaler.transform(dis_test)

    return oes_tr_s, oes_te_s, dis_tr_s, dis_te_s


def get_input_config(config_name, oes_tr_s, oes_te_s, dis_tr_s, dis_te_s):
    """Assemble X_train and X_test for a given config."""
    if config_name == "A":
        return oes_tr_s, oes_te_s
    elif config_name == "B":
        return dis_tr_s, dis_te_s
    elif config_name == "C":
        return (np.hstack([oes_tr_s, dis_tr_s]),
                np.hstack([oes_te_s, dis_te_s]))


def run_loocv_for_model(model_name, oes_features, data, config_name, params=None):
    """Run LOOCV for one model x one config using engineered features.

    Args:
        model_name: one of "Ridge", "PLS", "RF", "MLP"
        oes_features: (n_samples, 13) pre-extracted OES features
        data: dict from prepare_data() (need discharge_raw, target)
        config_name: "A", "B", or "C"
        params: model-specific parameters (dict or None for defaults)

    Returns:
        result dict with keys: model, config, R2, RMSE, MAE, y_true, y_pred
    """
    discharge = data["discharge_raw"]
    target = data["target"]
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(oes_features):
        oes_tr_s, oes_te_s, dis_tr_s, dis_te_s = _scale_features(
            oes_features[train_idx], oes_features[test_idx],
            discharge[train_idx], discharge[test_idx],
        )

        X_train, X_test = get_input_config(
            config_name, oes_tr_s, oes_te_s, dis_tr_s, dis_te_s
        )

        if model_name == "Ridge":
            model = RidgeModel(alphas=RIDGE_ALPHAS)
            model.fit(X_train, target[train_idx])
            pred = model.predict(X_test)

        elif model_name == "PLS":
            model = PLSModel(max_components=PLS_MAX_COMPONENTS)
            model.fit(X_train, target[train_idx])
            pred = model.predict(X_test)

        elif model_name == "RF":
            rf_params = params if params is not None else RF_PARAMS
            model = RFModel(params=rf_params)
            model.fit(X_train, target[train_idx])
            pred = model.predict(X_test)

        elif model_name == "MLP":
            from .tuner_mlp import MLPNetBN, _train_mlp
            torch.manual_seed(RANDOM_SEED)
            mlp_cfg = params if params is not None else MLP_CONFIG
            input_dim = X_train.shape[1]
            net = MLPNetBN(
                input_dim, mlp_cfg["hidden_sizes"],
                mlp_cfg["dropout"], mlp_cfg.get("batch_norm", False)
            )
            _train_mlp(net, X_train, target[train_idx], mlp_cfg)
            net.eval()
            with torch.no_grad():
                X_te = torch.tensor(np.atleast_2d(X_test), dtype=torch.float32)
                pred = net(X_te).numpy().ravel()

        y_true_all.append(target[test_idx][0])
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


def run_all_evaluations(oes_features, data, tuned_params=None):
    """Run LOOCV for all model x config combinations.

    Args:
        oes_features: (n_samples, 13) pre-extracted OES features
        data: dict from prepare_data()
        tuned_params: optional dict {("RF","A"): params, ("MLP","B"): config, ...}

    Returns:
        (all_results, results_df)
    """
    if tuned_params is None:
        tuned_params = {}

    all_results = []
    for model_name in MODEL_NAMES:
        for config_name in MODEL_CONFIGS[model_name]:
            params = tuned_params.get((model_name, config_name))
            print(f"  Running {model_name} Config {config_name}...")
            result = run_loocv_for_model(
                model_name, oes_features, data, config_name, params
            )
            print(f"    R2={result['R2']:.3f}  RMSE={result['RMSE']:.3f}  MAE={result['MAE']:.3f}")
            all_results.append(result)

    summary_rows = [{
        "Model": r["model"], "Config": r["config"],
        "R2": r["R2"], "RMSE": r["RMSE"], "MAE": r["MAE"],
    } for r in all_results]
    results_df = pd.DataFrame(summary_rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TABLES_DIR / "phase3_loocv_results_summary.csv", index=False)

    detail_rows = []
    for r in all_results:
        for i, (yt, yp) in enumerate(zip(r["y_true"], r["y_pred"])):
            detail_rows.append({
                "Model": r["model"], "Config": r["config"],
                "Sample": i, "y_true": yt, "y_pred": yp,
            })
    pd.DataFrame(detail_rows).to_csv(TABLES_DIR / "phase3_predictions_detail.csv", index=False)

    return all_results, results_df


def build_comparison_table(phase3_df):
    """Load Phase 1 and Phase 2 results, merge with Phase 3, compute deltas."""
    phase1_df = pd.read_csv(PHASE1_RESULTS_PATH)
    phase1_df = phase1_df[phase1_df["Model"].isin(MODEL_NAMES)].copy()
    phase1_df = phase1_df.rename(
        columns={"R2": "R2_P1", "RMSE": "RMSE_P1", "MAE": "MAE_P1"}
    )

    phase3_renamed = phase3_df.rename(
        columns={"R2": "R2_P3", "RMSE": "RMSE_P3", "MAE": "MAE_P3"}
    )

    merged = phase1_df.merge(phase3_renamed, on=["Model", "Config"], how="outer")

    # Optionally merge Phase 2 results
    if PHASE2_RESULTS_PATH.exists():
        phase2_df = pd.read_csv(PHASE2_RESULTS_PATH)
        phase2_df = phase2_df[phase2_df["Model"].isin(MODEL_NAMES)].copy()
        phase2_df = phase2_df.rename(
            columns={"R2": "R2_P2", "RMSE": "RMSE_P2", "MAE": "MAE_P2"}
        )
        merged = merged.merge(phase2_df, on=["Model", "Config"], how="outer")

    # Compute deltas (Phase 3 vs Phase 1)
    if "R2_P1" in merged.columns and "R2_P3" in merged.columns:
        merged["Delta_R2_P3_vs_P1"] = merged["R2_P3"] - merged["R2_P1"]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TABLES_DIR / "phase1_vs_phase2_vs_phase3_comparison.csv", index=False)
    return merged
