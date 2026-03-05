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


def compute_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def _scale_and_pca(oes_train, oes_test, discharge_train, discharge_test, pca_k):
    """Fit scalers and PCA on training data, transform both train and test."""
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
    """Assemble X_train and X_test for a given config."""
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
    """Run LOOCV for one model x one config. Returns result dict."""
    oes = data["oes_raw"]
    discharge = data["discharge_raw"]
    target = data["target"]
    n = len(target)
    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []
    is_cnn = model_name == "CNN"

    for train_idx, test_idx in loo.split(oes):
        oes_train, oes_test = oes[train_idx], oes[test_idx]
        dis_train, dis_test = discharge[train_idx], discharge[test_idx]
        y_train, y_test = target[train_idx], target[test_idx]

        pca_tr, pca_te, dis_tr_s, dis_te_s, oes_tr_s, oes_te_s = _scale_and_pca(
            oes_train, oes_test, dis_train, dis_test, pca_k
        )

        if is_cnn:
            result = get_input_config(config_name, pca_tr, pca_te, dis_tr_s, dis_te_s,
                                       oes_tr_s, oes_te_s, is_cnn=True)
            oes_in_tr, oes_in_te, extra_tr, extra_te = result
            model = _create_model(model_name)
            model.fit(oes_in_tr, y_train, extra_tr)
            pred = model.predict(oes_in_te, extra_te)
        else:
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
    factories = {
        "Ridge": RidgeModel,
        "PLS": PLSModel,
        "SVR": SVRModel,
        "XGBoost": XGBoostModel,
        "MLP": MLPModel,
        "CNN": CNNModel,
    }
    return factories[model_name]()


def run_all_evaluations(data, pca_k):
    """Run all model x config combinations. Returns (list of result dicts, summary DataFrame)."""
    model_names = ["Ridge", "PLS", "SVR", "XGBoost", "MLP", "CNN"]
    config_names = ["A", "B", "C"]
    all_results = []

    for mname in model_names:
        for cname in config_names:
            if mname == "CNN" and cname == "B":
                print(f"  Skipping {mname} Config {cname} (not applicable)")
                continue
            print(f"  Running {mname} Config {cname}...")
            result = run_loocv_for_model(mname, data, pca_k, cname)
            print(f"    R2={result['R2']:.3f}  RMSE={result['RMSE']:.3f}  MAE={result['MAE']:.3f}")
            all_results.append(result)

    summary_rows = [{
        "Model": r["model"],
        "Config": r["config"],
        "R2": r["R2"],
        "RMSE": r["RMSE"],
        "MAE": r["MAE"],
    } for r in all_results]
    results_df = pd.DataFrame(summary_rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TABLES_DIR / "loocv_results_summary.csv", index=False)

    detail_rows = []
    for r in all_results:
        for i, (yt, yp) in enumerate(zip(r["y_true"], r["y_pred"])):
            detail_rows.append({
                "Model": r["model"], "Config": r["config"],
                "Sample": i, "y_true": yt, "y_pred": yp,
            })
    pd.DataFrame(detail_rows).to_csv(TABLES_DIR / "loocv_predictions_detail.csv", index=False)

    return all_results, results_df
