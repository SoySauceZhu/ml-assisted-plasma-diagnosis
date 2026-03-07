import numpy as np
import torch
import torch.nn as nn
import optuna
from sklearn.model_selection import LeaveOneOut

from phase1.evaluation import _scale_and_pca, get_input_config, compute_metrics
from .config import RANDOM_SEED, N_TRIALS_CNN, PCA_K


class CNN1DTunable(nn.Module):
    """Extended 1D CNN with configurable pool_type and fc_hidden."""

    def __init__(self, input_length, conv_channels, kernel_size, dropout,
                 n_extra_features=0, pool_type="avg", fc_hidden=None):
        super().__init__()
        self.n_extra = n_extra_features

        layers = []
        in_ch = 1
        for i, out_ch in enumerate(conv_channels):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            if i == 0:
                layers.append(nn.MaxPool1d(2))
            in_ch = out_ch

        if pool_type == "max":
            layers.append(nn.AdaptiveMaxPool1d(1))
        else:
            layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*layers)

        fc_input = conv_channels[-1] + n_extra_features
        head_layers = [nn.Dropout(dropout)]
        if fc_hidden is not None and fc_hidden > 0:
            head_layers += [nn.Linear(fc_input, fc_hidden), nn.ReLU(), nn.Dropout(dropout)]
            fc_input = fc_hidden
        head_layers.append(nn.Linear(fc_input, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x_oes, x_extra=None):
        out = self.conv(x_oes)          # (batch, C, 1)
        out = out.squeeze(-1)           # (batch, C)
        if x_extra is not None and self.n_extra > 0:
            out = torch.cat([out, x_extra], dim=-1)
        return self.head(out).squeeze(-1)


def _train_cnn(model, X_oes_train, y_train, X_extra_train, cfg):
    """Train CNN with given config. Returns trained model."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = nn.MSELoss()

    X_oes_t = torch.tensor(X_oes_train, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_ext_t = (torch.tensor(X_extra_train, dtype=torch.float32)
               if X_extra_train is not None else None)

    best_loss = np.inf
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(cfg["max_epochs"]):
        optimizer.zero_grad()
        pred = model(X_oes_t, X_ext_t)
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


def _parse_conv_channels(s):
    """Parse underscore-joined string to list of ints: '16_32' -> [16, 32]."""
    return [int(x) for x in s.split("_")]


def _postprocess_cnn_config(raw_params):
    """Convert Optuna raw params to valid CNN config dict."""
    cfg = dict(raw_params)
    cfg["conv_channels"] = _parse_conv_channels(cfg["conv_channels"])
    if cfg["fc_hidden"] == 0:
        cfg["fc_hidden"] = None
    return cfg


def cnn_objective(trial, data, config_name, pca_k):
    """Optuna objective for CNN. Returns R2 from inner LOOCV."""
    raw_cfg = {
        "conv_channels": trial.suggest_categorical(
            "conv_channels", ["8", "16", "8_16", "16_32", "32_64"]
        ),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7, 11, 15, 21]),
        "dropout": trial.suggest_float("dropout", 0.3, 0.6),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "max_epochs": trial.suggest_categorical("max_epochs", [300, 500, 1000]),
        "patience": trial.suggest_categorical("patience", [30, 50, 100]),
        "pool_type": trial.suggest_categorical("pool_type", ["avg", "max"]),
        "fc_hidden": trial.suggest_categorical("fc_hidden", [0, 8, 16]),
    }
    cfg = _postprocess_cnn_config(raw_cfg)

    oes = data["oes_raw"]
    discharge = data["discharge_raw"]
    target = data["target"]
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(oes):
        torch.manual_seed(RANDOM_SEED)

        pca_tr, pca_te, dis_tr_s, dis_te_s, oes_tr_s, oes_te_s = _scale_and_pca(
            oes[train_idx], oes[test_idx],
            discharge[train_idx], discharge[test_idx], pca_k
        )

        result = get_input_config(
            config_name, pca_tr, pca_te, dis_tr_s, dis_te_s,
            oes_tr_s, oes_te_s, is_cnn=True
        )
        oes_in_tr, oes_in_te, extra_tr, extra_te = result

        n_extra = extra_tr.shape[1] if extra_tr is not None else 0
        model = CNN1DTunable(
            input_length=oes_in_tr.shape[1],
            conv_channels=cfg["conv_channels"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
            n_extra_features=n_extra,
            pool_type=cfg["pool_type"],
            fc_hidden=cfg["fc_hidden"],
        )
        _train_cnn(model, oes_in_tr, target[train_idx], extra_tr, cfg)

        model.eval()
        with torch.no_grad():
            X_oes_t = torch.tensor(np.atleast_2d(oes_in_te), dtype=torch.float32).unsqueeze(1)
            X_ext_t = (torch.tensor(np.atleast_2d(extra_te), dtype=torch.float32)
                       if extra_te is not None else None)
            pred = model(X_oes_t, X_ext_t).numpy().ravel()

        y_true_all.append(target[test_idx][0])
        y_pred_all.append(pred[0])

    metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
    return metrics["R2"]


def tune_cnn(data, config_name, pca_k=PCA_K, n_trials=N_TRIALS_CNN):
    """Run Optuna study for CNN. Returns (best_config, study)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"CNN_Config{config_name}",
    )
    study.optimize(
        lambda trial: cnn_objective(trial, data, config_name, pca_k),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_config = _postprocess_cnn_config(study.best_params)
    print(f"  Best R2: {study.best_value:.4f}")
    print(f"  Best config: {best_config}")
    return best_config, study
