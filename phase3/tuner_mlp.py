import numpy as np
import torch
import torch.nn as nn
import optuna
from sklearn.model_selection import LeaveOneOut

from phase1.evaluation import compute_metrics
from .evaluation import _scale_features, get_input_config
from .config import RANDOM_SEED, N_TRIALS_MLP


class MLPNetBN(nn.Module):
    """MLP with optional batch normalization."""

    def __init__(self, input_dim, hidden_sizes, dropout, batch_norm=False):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp(model, X_train, y_train, cfg):
    """Train MLP with given config. Returns trained model."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    best_loss = np.inf
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(cfg["max_epochs"]):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _parse_hidden_sizes(s):
    """Parse underscore-joined string to list of ints: '16_8' -> [16, 8]."""
    return [int(x) for x in s.split("_")]


def _postprocess_mlp_config(raw_params):
    """Convert Optuna raw params to valid MLP config dict."""
    cfg = dict(raw_params)
    cfg["hidden_sizes"] = _parse_hidden_sizes(cfg["hidden_sizes"])
    return cfg


def mlp_objective(trial, oes_features, data, config_name):
    """Optuna objective for MLP. Returns R2 from inner LOOCV."""
    raw_cfg = {
        "hidden_sizes": trial.suggest_categorical(
            "hidden_sizes", ["8", "16", "32", "8_4", "16_8"]
        ),
        "dropout": trial.suggest_float("dropout", 0.3, 0.6),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "max_epochs": trial.suggest_categorical("max_epochs", [300, 500, 1000]),
        "patience": trial.suggest_categorical("patience", [30, 50, 100]),
        "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
    }
    cfg = _postprocess_mlp_config(raw_cfg)

    discharge = data["discharge_raw"]
    target = data["target"]
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(oes_features):
        torch.manual_seed(RANDOM_SEED)

        oes_tr_s, oes_te_s, dis_tr_s, dis_te_s = _scale_features(
            oes_features[train_idx], oes_features[test_idx],
            discharge[train_idx], discharge[test_idx],
        )

        X_train, X_test = get_input_config(
            config_name, oes_tr_s, oes_te_s, dis_tr_s, dis_te_s
        )

        input_dim = X_train.shape[1]
        model = MLPNetBN(input_dim, cfg["hidden_sizes"], cfg["dropout"], cfg["batch_norm"])
        _train_mlp(model, X_train, target[train_idx], cfg)

        model.eval()
        with torch.no_grad():
            X_te = torch.tensor(np.atleast_2d(X_test), dtype=torch.float32)
            pred = model(X_te).numpy().ravel()

        y_true_all.append(target[test_idx][0])
        y_pred_all.append(pred[0])

    metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
    return metrics["R2"]


def tune_mlp(oes_features, data, config_name, n_trials=N_TRIALS_MLP):
    """Run Optuna study for MLP. Returns (best_config, study)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"MLP_Config{config_name}",
    )
    study.optimize(
        lambda trial: mlp_objective(trial, oes_features, data, config_name),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_config = _postprocess_mlp_config(study.best_params)
    print(f"  Best R2: {study.best_value:.4f}")
    print(f"  Best config: {best_config}")
    return best_config, study
