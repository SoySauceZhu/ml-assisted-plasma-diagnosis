"""
Phase 2: Hyperparameter Tuning for RF, CNN, MLP.
Consolidated version for Google Colab with GPU support.

Usage in Colab:
    from colab import main
    main()

Or via command line:
    !python colab.py
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
PCA_K = 11  # From Phase 1 analysis (95% variance threshold)

N_TRIALS_RF = 200
N_TRIALS_MLP = 100
N_TRIALS_CNN = 100

# Column definitions
META_COLS = ["sheet", "condition"]
DISCHARGE_COLS = ["frequency_hz", "pulse_width_ns", "rise_time_ns", "flow_rate_sccm"]
TARGET_COL = "h2o2_rate"
OES_PREFIX = "I_"

# Model x Config applicability
MODEL_CONFIGS = {
    "RF": ["A", "B", "C"],
    "MLP": ["A", "B", "C"],
    "CNN": ["A", "C"],
}

# Search spaces
RF_SEARCH_SPACE = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_split": [2, 3, 4, 5],
    "min_samples_leaf": [1, 2, 3],
    "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
    "bootstrap": [True, False],
}

MLP_SEARCH_SPACE = {
    "hidden_sizes": [[8], [16], [32], [8, 4], [16, 8], [32, 16]],
    "dropout": (0.3, 0.7),
    "weight_decay": (1e-3, 1e-1),
    "lr": (1e-4, 5e-3),
    "max_epochs": [200, 500, 1000],
    "patience": [30, 50, 100],
    "batch_norm": [True, False],
}

CNN_SEARCH_SPACE = {
    "conv_channels": [[8], [16], [8, 16], [16, 32], [32, 64]],
    "kernel_size": [3, 5, 7, 11, 15, 21],
    "dropout": (0.3, 0.6),
    "weight_decay": (1e-3, 5e-2),
    "lr": (1e-4, 1e-3),
    "max_epochs": [300, 500, 1000],
    "patience": [30, 50, 100],
    "pool_type": ["avg", "max"],
    "fc_hidden": [None, 8, 16],
}

# ============================================================
# Device Selection (GPU Support)
# ============================================================

def get_device():
    """Select GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

DEVICE = get_device()


# ============================================================
# Data Loading (from phase1)
# ============================================================

def load_dataset(csv_path="oes_ml_dataset_1nm.csv"):
    df = pd.read_csv(csv_path)
    return df


def separate_features(df):
    oes_cols = [c for c in df.columns if c.startswith(OES_PREFIX)]
    oes_df = df[oes_cols]
    discharge_df = df[DISCHARGE_COLS]
    target = df[TARGET_COL]
    return oes_df, discharge_df, target


def baseline_correction(oes_df, df):
    """Subtract the spectrum of the pulse_width=0 ns sample."""
    baseline_mask = df["pulse_width_ns"] == 0.0
    if baseline_mask.sum() == 0:
        print("Warning: no pulse_width=0 baseline sample found, skipping baseline correction")
        return oes_df
    baseline_spectrum = oes_df.loc[baseline_mask].values.mean(axis=0)
    corrected = oes_df.values - baseline_spectrum
    return pd.DataFrame(corrected, columns=oes_df.columns, index=oes_df.index)


def prepare_data(csv_path="oes_ml_dataset_1nm.csv"):
    """Load and prepare data for modeling."""
    df = load_dataset(csv_path)
    oes_df, discharge_df, target = separate_features(df)
    oes_df = baseline_correction(oes_df, df)

    oes_cols = [c for c in oes_df.columns]
    wavelengths = np.array([int(c.replace(OES_PREFIX, "")) for c in oes_cols])

    return {
        "oes_raw": oes_df.values.astype(np.float64),
        "discharge_raw": discharge_df.values.astype(np.float64),
        "target": target.values.astype(np.float64),
        "wavelengths": wavelengths,
        "sample_info": df[META_COLS].copy(),
    }


# ============================================================
# Evaluation Functions (from phase1)
# ============================================================

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


# ============================================================
# RF Model (from phase1)
# ============================================================

class RFModel:
    def __init__(self, params=None):
        self.params = params or {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
        }
        self.model = None

    def fit(self, X_train, y_train):
        self.model = RandomForestRegressor(
            **self.params,
            random_state=RANDOM_SEED,
        )
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(np.atleast_2d(X_test)).ravel()


# ============================================================
# MLP Model (from phase2 tuner_mlp)
# ============================================================

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
        # Move to GPU
        self.to(DEVICE)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp(model, X_train, y_train, cfg):
    """Train MLP with given config. Returns trained model."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

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


def mlp_objective(trial, data, config_name, pca_k):
    """Optuna objective for MLP. Returns R2 from inner LOOCV."""
    raw_cfg = {
        "hidden_sizes": trial.suggest_categorical(
            "hidden_sizes", ["8", "16", "32", "8_4", "16_8", "32_16"]
        ),
        "dropout": trial.suggest_float("dropout", 0.3, 0.7),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "max_epochs": trial.suggest_categorical("max_epochs", [200, 500, 1000]),
        "patience": trial.suggest_categorical("patience", [30, 50, 100]),
        "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
    }
    cfg = _postprocess_mlp_config(raw_cfg)

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

        if config_name == "B":
            X_train, X_test = dis_tr_s, dis_te_s
        else:
            X_train, X_test = get_input_config(
                config_name, pca_tr, pca_te, dis_tr_s, dis_te_s
            )

        input_dim = X_train.shape[1]
        model = MLPNetBN(input_dim, cfg["hidden_sizes"], cfg["dropout"], cfg["batch_norm"])
        _train_mlp(model, X_train, target[train_idx], cfg)

        model.eval()
        with torch.no_grad():
            X_te = torch.tensor(np.atleast_2d(X_test), dtype=torch.float32).to(DEVICE)
            pred = model(X_te).cpu().numpy().ravel()

        y_true_all.append(target[test_idx][0])
        y_pred_all.append(pred[0])

    metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
    return metrics["R2"]


def tune_mlp(data, config_name, pca_k=PCA_K, n_trials=N_TRIALS_MLP):
    """Run Optuna study for MLP. Returns (best_config, study)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"MLP_Config{config_name}",
    )
    study.optimize(
        lambda trial: mlp_objective(trial, data, config_name, pca_k),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_config = _postprocess_mlp_config(study.best_params)
    print(f"  Best R2: {study.best_value:.4f}")
    print(f"  Best config: {best_config}")
    return best_config, study


# ============================================================
# CNN Model (from phase2 tuner_cnn)
# ============================================================

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

        # Move to GPU
        self.to(DEVICE)

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

    X_oes_t = torch.tensor(X_oes_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_ext_t = (torch.tensor(X_extra_train, dtype=torch.float32).to(DEVICE)
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
    # Convert 0 to None for fc_hidden
    if raw_cfg["fc_hidden"] == 0:
        raw_cfg["fc_hidden"] = None

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

        net = CNN1DTunable(
            input_length=oes_in_tr.shape[1],
            conv_channels=cfg["conv_channels"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
            n_extra_features=n_extra,
            pool_type=cfg["pool_type"],
            fc_hidden=cfg["fc_hidden"],
        )
        _train_cnn(net, oes_in_tr, target[train_idx], extra_tr, cfg)

        net.eval()
        with torch.no_grad():
            X_oes_t = torch.tensor(
                np.atleast_2d(oes_in_te), dtype=torch.float32
            ).unsqueeze(1).to(DEVICE)
            X_ext_t = (torch.tensor(np.atleast_2d(extra_te), dtype=torch.float32).to(DEVICE)
                       if extra_te is not None else None)
            pred = net(X_oes_t, X_ext_t).cpu().numpy().ravel()

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
    # Convert None back for serialization
    if best_config.get("fc_hidden") is None:
        best_config["fc_hidden"] = 0
    print(f"  Best R2: {study.best_value:.4f}")
    print(f"  Best config: {best_config}")
    return best_config, study


# ============================================================
# RF Tuner (from phase2 tuner_rf)
# ============================================================

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


# ============================================================
# Evaluation (from phase2 evaluation.py)
# ============================================================

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
                X_te = torch.tensor(np.atleast_2d(X_test), dtype=torch.float32).to(DEVICE)
                pred = net(X_te).cpu().numpy().ravel()

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
                fc_hidden=best_config.get("fc_hidden") if best_config.get("fc_hidden") != 0 else None,
            )
            _train_cnn(net, oes_in_tr, target[train_idx], extra_tr, best_config)
            net.eval()
            with torch.no_grad():
                X_oes_t = torch.tensor(
                    np.atleast_2d(oes_in_te), dtype=torch.float32
                ).unsqueeze(1).to(DEVICE)
                X_ext_t = (torch.tensor(np.atleast_2d(extra_te), dtype=torch.float32).to(DEVICE)
                           if extra_te is not None else None)
                pred = net(X_oes_t, X_ext_t).cpu().numpy().ravel()

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


def run_all_tuned_evaluations(data, tuned_params_dict, pca_k=PCA_K, output_dir="results"):
    """Run outer LOOCV for all tuned model x config combinations."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

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

    results_df.to_csv(f"{output_dir}/tables/phase2_loocv_results_summary.csv", index=False)

    detail_rows = []
    for r in all_results:
        for i, (yt, yp) in enumerate(zip(r["y_true"], r["y_pred"])):
            detail_rows.append({
                "Model": r["model"], "Config": r["config"],
                "Sample": i, "y_true": yt, "y_pred": yp,
            })
    pd.DataFrame(detail_rows).to_csv(f"{output_dir}/tables/phase2_predictions_detail.csv", index=False)

    return all_results, results_df


# ============================================================
# Plotting (from phase2 plotting.py)
# ============================================================

def plot_optimization_history(study, model_name, config_name, save_dir="results/figures"):
    """Plot Optuna optimization history."""
    trials = study.trials
    trial_numbers = [t.number for t in trials if t.value is not None]
    values = [t.value for t in trials if t.value is not None]

    best_values = []
    current_best = -np.inf
    for v in values:
        current_best = max(current_best, v)
        best_values.append(current_best)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(trial_numbers, values, alpha=0.4, s=15, label="Trial R2")
    ax.plot(trial_numbers, best_values, color="red", linewidth=2, label="Best R2")
    ax.set_xlabel("Trial")
    ax.set_ylabel("R2")
    ax.set_title(f"Optimization History: {model_name} Config {config_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/optuna_history_{model_name}_{config_name}.png", dpi=150)
    plt.close(fig)


def plot_param_importances(study, model_name, config_name, save_dir="results/figures"):
    """Plot hyperparameter importance from Optuna study."""
    try:
        from optuna.importance import get_param_importances
        importances = get_param_importances(study)
    except Exception:
        print(f"    Skipping param importance plot for {model_name} Config {config_name}")
        return

    names = list(importances.keys())
    values = list(importances.values())
    sorted_idx = np.argsort(values)
    names = [names[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    ax.barh(names, values)
    ax.set_xlabel("Importance")
    ax.set_title(f"Param Importance: {model_name} Config {config_name}")
    fig.tight_layout()
    fig.savefig(f"{save_dir}/optuna_importance_{model_name}_{config_name}.png", dpi=150)
    plt.close(fig)


def plot_comparison_bar(comparison_df, save_path="results/figures/phase1_vs_phase2_comparison.png"):
    """Side-by-side bar chart: Phase 1 vs Phase 2 R2."""
    df = comparison_df.dropna(subset=["R2_P1", "R2_P2"])
    labels = [f"{row['Model']}\nConfig {row['Config']}" for _, row in df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    bars1 = ax.bar(x - width / 2, df["R2_P1"].values, width, label="Phase 1", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, df["R2_P2"].values, width, label="Phase 2 (Tuned)", color="#DD8452")

    ax.set_ylabel("R2")
    ax.set_title("Phase 1 vs Phase 2: R2 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_phase2_predicted_vs_actual(all_results, save_dir="results/figures"):
    """Predicted vs actual scatter plots for tuned models."""
    for r in all_results:
        fig, ax = plt.subplots(figsize=(5, 5))
        y_true = r["y_true"]
        y_pred = r["y_pred"]

        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidth=0.5)
        mn = min(y_true.min(), y_pred.min()) - 0.05
        mx = max(y_true.max(), y_pred.max()) + 0.05
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Ideal")
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_xlabel("Actual H2O2 Rate")
        ax.set_ylabel("Predicted H2O2 Rate")
        ax.set_title(f"{r['model']} Config {r['config']} (Tuned)\n"
                      f"R2={r['R2']:.3f}  RMSE={r['RMSE']:.3f}")
        ax.legend()
        ax.set_aspect("equal")
        fig.tight_layout()
        fname = f"predicted_vs_actual_{r['model']}_{r['config']}_tuned.png"
        fig.savefig(f"{save_dir}/{fname}", dpi=150)
        plt.close(fig)


def generate_all_phase2_plots(all_results, results_df, studies_dict, comparison_df=None):
    """Master plotting function for Phase 2."""
    os.makedirs("results/figures", exist_ok=True)

    for (model_name, config_name), study in studies_dict.items():
        plot_optimization_history(study, model_name, config_name)
        plot_param_importances(study, model_name, config_name)

    plot_phase2_predicted_vs_actual(all_results)
    if comparison_df is not None:
        plot_comparison_bar(comparison_df)

    print(f"  Plots saved to results/figures")


# ============================================================
# Main Function
# ============================================================

def _serialize_params(tuned_params):
    """Convert tuned_params dict to JSON-serializable format."""
    out = {}
    for (model, config), params in tuned_params.items():
        key = f"{model}_{config}"
        serializable = {}
        for k, v in params.items():
            if isinstance(v, list):
                serializable[k] = v
            elif v is None:
                serializable[k] = None
            else:
                serializable[k] = v
        out[key] = serializable
    return out


def _deserialize_params(raw):
    """Convert JSON-loaded dict back to tuned_params format."""
    tuned = {}
    for key, params in raw.items():
        parts = key.split("_", 1)
        model, config = parts[0], parts[1]
        tuned[(model, config)] = params
    return tuned


def main(csv_path="oes_ml_dataset_1nm.csv",
         tune_only=False,
         eval_only=False,
         models=None,
         pca_k=PCA_K,
         n_trials_rf=N_TRIALS_RF,
         n_trials_mlp=N_TRIALS_MLP,
         n_trials_cnn=N_TRIALS_CNN):
    """
    Main function for Phase 2 hyperparameter tuning.

    Args:
        csv_path: Path to the dataset CSV file
        tune_only: Only run tuning, skip final evaluation
        eval_only: Only run evaluation with saved params
        models: List of models to tune (default: ["RF", "MLP", "CNN"])
        pca_k: Number of PCA components
        n_trials_rf: Number of Optuna trials for RF
        n_trials_mlp: Number of Optuna trials for MLP
        n_trials_cnn: Number of Optuna trials for CNN
    """
    global N_TRIALS_RF, N_TRIALS_MLP, N_TRIALS_CNN
    N_TRIALS_RF = n_trials_rf
    N_TRIALS_MLP = n_trials_mlp
    N_TRIALS_CNN = n_trials_cnn

    if models is None:
        models = ["RF", "MLP", "CNN"]

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("=" * 60)
    print("Phase 2: Hyperparameter Tuning with GPU Acceleration")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\nLoading data...")
    data = prepare_data(csv_path)
    print(f"  Samples: {len(data['target'])}, PCA k={pca_k}")

    tuned_params = {}
    studies = {}
    params_path = "results/tables/tuned_hyperparameters.json"

    if not eval_only:
        tuner_map = {"RF": tune_rf, "MLP": tune_mlp, "CNN": tune_cnn}

        for model_name in models:
            if model_name not in MODEL_CONFIGS:
                print(f"Warning: Unknown model {model_name}, skipping")
                continue
            configs = MODEL_CONFIGS[model_name]
            for config_name in configs:
                print(f"\n{'=' * 50}")
                print(f"Tuning {model_name} Config {config_name}...")
                print(f"{'=' * 50}")

                tune_fn = tuner_map[model_name]
                best_params, study = tune_fn(data, config_name, pca_k=pca_k)

                tuned_params[(model_name, config_name)] = best_params
                studies[(model_name, config_name)] = study

        with open(params_path, "w") as f:
            json.dump(_serialize_params(tuned_params), f, indent=2)
        print(f"\nTuned params saved to {params_path}")

    if tune_only:
        print("\nTuning complete. Exiting (--tune-only).")
        return tuned_params, studies

    if eval_only:
        print(f"\nLoading tuned params from {params_path}...")
        with open(params_path) as f:
            raw = json.load(f)
        tuned_params = _deserialize_params(raw)

    print(f"\nRunning outer LOOCV with tuned parameters...")
    all_results, results_df = run_all_tuned_evaluations(
        data, tuned_params, pca_k=pca_k
    )

    print("\nGenerating plots...")
    generate_all_phase2_plots(all_results, results_df, studies)

    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to results/")

    return all_results, results_df, tuned_params, studies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Hyperparameter Tuning")
    parser.add_argument("--csv-path", type=str, default="oes_ml_dataset_1nm.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--tune-only", action="store_true",
                        help="Only run tuning, skip final evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation with saved params")
    parser.add_argument("--models", nargs="+", default=["RF", "MLP", "CNN"],
                        choices=["RF", "MLP", "CNN"],
                        help="Models to tune (default: all three)")
    parser.add_argument("--pca-k", type=int, default=PCA_K,
                        help=f"Number of PCA components (default: {PCA_K})")
    parser.add_argument("--n-trials-rf", type=int, default=N_TRIALS_RF,
                        help=f"Number of Optuna trials for RF (default: {N_TRIALS_RF})")
    parser.add_argument("--n-trials-mlp", type=int, default=N_TRIALS_MLP,
                        help=f"Number of Optuna trials for MLP (default: {N_TRIALS_MLP})")
    parser.add_argument("--n-trials-cnn", type=int, default=N_TRIALS_CNN,
                        help=f"Number of Optuna trials for CNN (default: {N_TRIALS_CNN})")

    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        tune_only=args.tune_only,
        eval_only=args.eval_only,
        models=args.models,
        pca_k=args.pca_k,
        n_trials_rf=args.n_trials_rf,
        n_trials_mlp=args.n_trials_mlp,
        n_trials_cnn=args.n_trials_cnn,
    )
