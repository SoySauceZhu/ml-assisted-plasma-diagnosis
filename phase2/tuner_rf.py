import numpy as np
import optuna
from sklearn.model_selection import LeaveOneOut

from phase1.evaluation import _scale_and_pca, get_input_config, compute_metrics
from phase1.models.rf import RFModel
from .config import RANDOM_SEED, N_TRIALS_RF, PCA_K


def _postprocess_rf_params(raw_params):
    """Convert Optuna raw params to valid RFModel params dict."""
    params = dict(raw_params)
    if params["max_depth"] == 0:
        params["max_depth"] = None
    if params["max_features"] in ("0.5", "0.8", "1.0"):
        params["max_features"] = float(params["max_features"])
    return params


def rf_objective(trial, data, config_name, pca_k):
    """Optuna objective for RF. Returns R2 from inner LOOCV."""
    raw_params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200, 500]),
        "max_depth": trial.suggest_categorical("max_depth", [0, 2, 3, 4, 5]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
        "max_features": trial.suggest_categorical("max_features",
                                                   ["sqrt", "log2", "0.5", "0.8", "1.0"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }
    params = _postprocess_rf_params(raw_params)

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

        if config_name == "B":
            X_train, X_test = dis_tr_s, dis_te_s
        else:
            X_train, X_test = get_input_config(
                config_name, pca_tr, pca_te, dis_tr_s, dis_te_s
            )

        model = RFModel(params=params)
        model.fit(X_train, target[train_idx])
        pred = model.predict(X_test)

        y_true_all.append(target[test_idx][0])
        y_pred_all.append(pred[0])

    metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
    return metrics["R2"]


def tune_rf(data, config_name, pca_k=PCA_K, n_trials=N_TRIALS_RF):
    """Run Optuna study for RF on a given config. Returns (best_params, study)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"RF_Config{config_name}",
    )
    study.optimize(
        lambda trial: rf_objective(trial, data, config_name, pca_k),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_params = _postprocess_rf_params(study.best_params)
    print(f"  Best R2: {study.best_value:.4f}")
    print(f"  Best params: {best_params}")
    return best_params, study
