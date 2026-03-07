import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneOut

from phase1.evaluation import _scale_and_pca, get_input_config, compute_metrics
from phase1.models.rf import RFModel
from .tuner_mlp import MLPNetBN, _train_mlp
from .tuner_cnn import CNN1DTunable, _train_cnn
from .config import RANDOM_SEED, PCA_K, TABLES_DIR, PHASE1_RESULTS_PATH


def run_tuned_loocv(model_name, data, pca_k, config_name, best_config):
    """Run outer LOOCV with fixed best_config. Returns result dict."""
    oes = data["oes_raw"]
    discharge = data["discharge_raw"]
    target = data["target"]
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(oes):
        pca_tr, pca_te, dis_tr_s, dis_te_s, oes_tr_s, oes_te_s = _scale_and_pca(
            oes[train_idx], oes[test_idx],
            discharge[train_idx], discharge[test_idx], pca_k
        )

        if model_name == "RF":
            if config_name == "B":
                X_train, X_test = dis_tr_s, dis_te_s
            else:
                X_train, X_test = get_input_config(
                    config_name, pca_tr, pca_te, dis_tr_s, dis_te_s
                )
            model = RFModel(params=best_config)
            model.fit(X_train, target[train_idx])
            pred = model.predict(X_test)

        elif model_name == "MLP":
            torch.manual_seed(RANDOM_SEED)
            if config_name == "B":
                X_train, X_test = dis_tr_s, dis_te_s
            else:
                X_train, X_test = get_input_config(
                    config_name, pca_tr, pca_te, dis_tr_s, dis_te_s
                )
            input_dim = X_train.shape[1]
            net = MLPNetBN(
                input_dim, best_config["hidden_sizes"],
                best_config["dropout"], best_config.get("batch_norm", False)
            )
            _train_mlp(net, X_train, target[train_idx], best_config)
            net.eval()
            with torch.no_grad():
                X_te = torch.tensor(np.atleast_2d(X_test), dtype=torch.float32)
                pred = net(X_te).numpy().ravel()

        elif model_name == "CNN":
            torch.manual_seed(RANDOM_SEED)
            result = get_input_config(
                config_name, pca_tr, pca_te, dis_tr_s, dis_te_s,
                oes_tr_s, oes_te_s, is_cnn=True
            )
            oes_in_tr, oes_in_te, extra_tr, extra_te = result
            n_extra = extra_tr.shape[1] if extra_tr is not None else 0
            net = CNN1DTunable(
                input_length=oes_in_tr.shape[1],
                conv_channels=best_config["conv_channels"],
                kernel_size=best_config["kernel_size"],
                dropout=best_config["dropout"],
                n_extra_features=n_extra,
                pool_type=best_config.get("pool_type", "avg"),
                fc_hidden=best_config.get("fc_hidden"),
            )
            _train_cnn(net, oes_in_tr, target[train_idx], extra_tr, best_config)
            net.eval()
            with torch.no_grad():
                X_oes_t = torch.tensor(
                    np.atleast_2d(oes_in_te), dtype=torch.float32
                ).unsqueeze(1)
                X_ext_t = (torch.tensor(np.atleast_2d(extra_te), dtype=torch.float32)
                           if extra_te is not None else None)
                pred = net(X_oes_t, X_ext_t).numpy().ravel()

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


def run_all_tuned_evaluations(data, tuned_params_dict, pca_k=PCA_K):
    """Run outer LOOCV for all tuned model x config combinations.

    Args:
        tuned_params_dict: {("RF","A"): params, ("MLP","B"): config, ...}

    Returns:
        (all_results, results_df) in the same format as Phase 1.
    """
    all_results = []
    for (model_name, config_name), best_config in tuned_params_dict.items():
        print(f"  Evaluating tuned {model_name} Config {config_name}...")
        result = run_tuned_loocv(model_name, data, pca_k, config_name, best_config)
        print(f"    R2={result['R2']:.3f}  RMSE={result['RMSE']:.3f}  MAE={result['MAE']:.3f}")
        all_results.append(result)

    summary_rows = [{
        "Model": r["model"], "Config": r["config"],
        "R2": r["R2"], "RMSE": r["RMSE"], "MAE": r["MAE"],
    } for r in all_results]
    results_df = pd.DataFrame(summary_rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TABLES_DIR / "phase2_loocv_results_summary.csv", index=False)

    detail_rows = []
    for r in all_results:
        for i, (yt, yp) in enumerate(zip(r["y_true"], r["y_pred"])):
            detail_rows.append({
                "Model": r["model"], "Config": r["config"],
                "Sample": i, "y_true": yt, "y_pred": yp,
            })
    pd.DataFrame(detail_rows).to_csv(TABLES_DIR / "phase2_predictions_detail.csv", index=False)

    return all_results, results_df


def build_comparison_table(phase2_df):
    """Load Phase 1 results, merge with Phase 2, compute improvement deltas."""
    phase1_df = pd.read_csv(PHASE1_RESULTS_PATH)
    phase1_filtered = phase1_df[phase1_df["Model"].isin(["RF", "MLP", "CNN"])].copy()
    phase1_filtered = phase1_filtered.rename(
        columns={"R2": "R2_P1", "RMSE": "RMSE_P1", "MAE": "MAE_P1"}
    )

    phase2_renamed = phase2_df.rename(
        columns={"R2": "R2_P2", "RMSE": "RMSE_P2", "MAE": "MAE_P2"}
    )

    merged = phase1_filtered.merge(phase2_renamed, on=["Model", "Config"], how="outer")
    merged["Delta_R2"] = merged["R2_P2"] - merged["R2_P1"]
    merged["Delta_RMSE"] = merged["RMSE_P1"] - merged["RMSE_P2"]  # positive = improvement

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TABLES_DIR / "phase1_vs_phase2_comparison.csv", index=False)
    return merged
