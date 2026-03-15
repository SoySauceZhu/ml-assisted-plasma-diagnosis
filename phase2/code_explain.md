# Phase 2: Bayesian Hyperparameter Tuning — Code Architecture

## Overview

Phase 2 optimises hyperparameters for RF, MLP, and CNN using **Optuna** (Bayesian optimisation with TPE sampler). The goal is to determine whether tuning alone can close the performance gap between OES-based models (Config A/C) and discharge-only models (Config B). Each model-config combination is tuned via inner LOOCV, then evaluated via outer LOOCV with fixed best parameters.

**Input**: Same dataset as Phase 1 + Phase 1 results for comparison
**Output**: Tuned hyperparameters (JSON), Phase 1 vs Phase 2 comparison table, optimisation history plots

## File Structure

```
phase2/
├── config.py           # Search spaces, trial counts, paths
├── tuner_rf.py         # Optuna tuning for Random Forest
├── tuner_mlp.py        # Optuna tuning for MLP (+ MLPNetBN class)
├── tuner_cnn.py        # Optuna tuning for CNN (+ CNN1DTunable class)
├── evaluation.py       # Outer LOOCV with tuned params, Phase 1 comparison
├── plotting.py         # Optimisation history, param importance, comparison plots
└── main.py             # CLI entry point
```

## Data Flow

```
main.py
  │
  ├─► phase1.data_loader.prepare_data()     # Reuses Phase 1 data loading
  │
  ├─► TUNING (for each model x config):
  │     tuner_rf.tune_rf() / tuner_mlp.tune_mlp() / tuner_cnn.tune_cnn()
  │       └─► Optuna study with TPE sampler
  │             └─► Objective function: inner LOOCV → R2
  │                   └─► Reuses phase1.evaluation._scale_and_pca() + get_input_config()
  │       └─► Returns (best_params, study)
  │     └─► Saves tuned_hyperparameters.json
  │
  ├─► EVALUATION:
  │     evaluation.run_all_tuned_evaluations(data, tuned_params)
  │       └─► For each (model, config): run_tuned_loocv() with fixed best params
  │     evaluation.build_comparison_table()
  │       └─► Loads Phase 1 results, computes Delta_R2
  │
  └─► plotting.generate_all_phase2_plots()
```

## Detailed File Documentation

---

### config.py

**Purpose**: Search spaces and tuning configuration.

Key constants:

| Constant | Value | Description |
|---|---|---|
| `PCA_K` | `11` | Fixed from Phase 1 (95% variance threshold) |
| `N_TRIALS_RF` | `200` | Optuna trials for RF |
| `N_TRIALS_MLP` | `100` | Optuna trials for MLP |
| `N_TRIALS_CNN` | `100` | Optuna trials for CNN |
| `RF_SEARCH_SPACE` | `{n_estimators, max_depth, ...}` | RF hyperparameter ranges |
| `MLP_SEARCH_SPACE` | `{hidden_sizes, dropout, lr, ...}` | MLP hyperparameter ranges |
| `CNN_SEARCH_SPACE` | `{conv_channels, kernel_size, pool_type, ...}` | CNN hyperparameter ranges |
| `MODEL_CONFIGS` | `{"RF": [A,B,C], "MLP": [A,B,C], "CNN": [A,C]}` | Which configs to tune per model |

---

### tuner_rf.py

**Purpose**: Optuna-based hyperparameter optimisation for Random Forest.

| Function | Signature | Description |
|---|---|---|
| `_postprocess_rf_params` | `(raw_params) -> dict` | Converts Optuna categorical outputs to valid RF params. E.g., max_depth=0 becomes None (unlimited), string max_features like "0.5" becomes float. |
| `rf_objective` | `(trial, data, config_name, pca_k) -> float` | **Optuna objective**. Suggests: n_estimators [50-500], max_depth [None,2-5], min_samples_split [2-5], min_samples_leaf [1-3], max_features [sqrt/log2/0.5/0.8/1.0], bootstrap [T/F]. Runs inner LOOCV with suggested params. Returns R2. |
| `tune_rf` | `(data, config_name, pca_k=11, n_trials=200) -> (dict, Study)` | Creates Optuna study (direction=maximize, TPE sampler). Runs n_trials optimisation. Returns best params dict and study object. |

---

### tuner_mlp.py

**Purpose**: Optuna tuning for MLP neural network. Defines `MLPNetBN`, an enhanced MLP with optional batch normalisation.

**`class MLPNetBN(nn.Module)`**:

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(input_dim, hidden_sizes, dropout, batch_norm=False)` | Builds MLP with optional BatchNorm1d after each Linear layer, before ReLU and Dropout. Output: Linear(last_hidden, 1). |
| `forward` | `(x) -> Tensor` | Forward pass, squeezed output. |

**Functions**:

| Function | Signature | Description |
|---|---|---|
| `_train_mlp` | `(model, X_train, y_train, cfg) -> nn.Module` | Trains MLPNetBN with Adam optimiser (lr, weight_decay from cfg). MSELoss with early stopping (patience). Saves and restores best state_dict. |
| `_parse_hidden_sizes` | `(s) -> list[int]` | Parses string "16_8" to list [16, 8]. Used because Optuna categorical values must be strings. |
| `_postprocess_mlp_config` | `(raw_params) -> dict` | Converts Optuna params to valid config: parses hidden_sizes string to list. |
| `mlp_objective` | `(trial, data, config_name, pca_k) -> float` | **Optuna objective**. Suggests: hidden_sizes [8/16/32/8_4/16_8/32_16], dropout [0.3-0.7], weight_decay [1e-3 to 1e-1], lr [1e-4 to 5e-3], max_epochs [200/500/1000], patience [30/50/100], batch_norm [T/F]. Runs inner LOOCV. Returns R2. |
| `tune_mlp` | `(data, config_name, pca_k=11, n_trials=100) -> (dict, Study)` | Creates Optuna study and runs optimisation. Returns best config and study. |

---

### tuner_cnn.py

**Purpose**: Optuna tuning for 1D CNN. Defines `CNN1DTunable`, an enhanced CNN with configurable pooling type and optional FC hidden layer.

**`class CNN1DTunable(nn.Module)`**:

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(input_length, conv_channels, kernel_size, dropout, n_extra_features=0, pool_type="avg", fc_hidden=None)` | Builds 1D CNN with: multiple Conv1d layers (ReLU, MaxPool1d after first), AdaptiveAvgPool1d or AdaptiveMaxPool1d (configurable), optional FC hidden layer before output. Supports two-input architecture for Config C (extra discharge features). |
| `forward` | `(x_oes, x_extra=None) -> Tensor` | Processes spectral input through conv layers, pools, optionally concatenates discharge features, passes through head. |

**Functions**:

| Function | Signature | Description |
|---|---|---|
| `_train_cnn` | `(model, X_oes_train, y_train, X_extra_train, cfg) -> nn.Module` | Trains CNN1DTunable. Reshapes OES to (N, 1, 701). Adam + MSELoss + early stopping. |
| `_parse_conv_channels` | `(s) -> list[int]` | Parses "16_32" to [16, 32]. |
| `_postprocess_cnn_config` | `(raw_params) -> dict` | Converts Optuna params. fc_hidden=0 becomes None. |
| `cnn_objective` | `(trial, data, config_name, pca_k) -> float` | **Optuna objective**. Suggests: conv_channels [8/16/8_16/16_32/32_64], kernel_size [3/5/7/11/15/21], dropout [0.3-0.6], weight_decay [1e-3 to 5e-2], lr [1e-4 to 1e-3], max_epochs [300/500/1000], patience [30/50/100], pool_type [avg/max], fc_hidden [0/8/16]. Returns R2. |
| `tune_cnn` | `(data, config_name, pca_k=11, n_trials=100) -> (dict, Study)` | Creates and runs Optuna study. Returns best config and study. |

---

### evaluation.py

**Purpose**: Outer LOOCV with fixed tuned hyperparameters, and cross-phase comparison.

| Function | Signature | Description |
|---|---|---|
| `run_tuned_loocv` | `(model_name, data, pca_k, config_name, best_config) -> dict` | Runs outer LOOCV for one model-config with fixed best hyperparameters. For RF: creates RFModel with best_config. For MLP: instantiates MLPNetBN and calls _train_mlp. For CNN: instantiates CNN1DTunable and calls _train_cnn. Returns R2, RMSE, MAE, y_true, y_pred. |
| `run_all_tuned_evaluations` | `(data, tuned_params_dict, pca_k=11) -> (list, DataFrame)` | Iterates all (model, config) pairs, calls run_tuned_loocv for each. Saves summary and detail CSVs. |
| `build_comparison_table` | `(phase2_df) -> DataFrame` | Loads Phase 1 results, merges with Phase 2 on (Model, Config), computes Delta_R2 and Delta_RMSE (improvement). Saves comparison CSV. |

---

### plotting.py

**Purpose**: Visualisation of tuning process and cross-phase comparison.

| Function | Signature | Description |
|---|---|---|
| `plot_optimization_history` | `(study, model_name, config_name, save_dir=None)` | Scatter plot of R2 per trial + best-so-far line. Shows tuning convergence. |
| `plot_param_importances` | `(study, model_name, config_name, save_dir=None)` | Horizontal bar chart of hyperparameter importances (via Optuna fANOVA). Shows which params matter most. |
| `_compute_param_importances` | `(study) -> dict` | Helper calling `optuna.importance.get_param_importances`. |
| `plot_comparison_bar` | `(comparison_df, save_path=None)` | Side-by-side bars: Phase 1 R2 vs Phase 2 R2 for each model-config. |
| `plot_phase2_predicted_vs_actual` | `(all_results, save_dir=None)` | Scatter plots of pred vs actual for each tuned model-config. |
| `generate_all_phase2_plots` | `(all_results, results_df, studies_dict, comparison_df)` | **Master function**. Generates all plot types and saves to FIGURES_DIR. |

---

### main.py

**Purpose**: CLI entry point for Phase 2.

| Function | Signature | Description |
|---|---|---|
| `_serialize_params` | `(tuned_params) -> dict` | Converts `{(model, config): params}` to JSON-serialisable `{"model_config": params}`. |
| `_deserialize_params` | `(raw) -> dict` | Reverses serialisation for loading from JSON. |
| `main` | `()` | Parses args: `--tune-only`, `--eval-only`, `--models [RF MLP CNN]`, `--pca-k`. Full pipeline: seed setup → data load → tune all models → save JSON → evaluate with tuned params → build comparison → plot. |

**Usage**:
```bash
python -m phase2.main                    # Full pipeline (tune + evaluate)
python -m phase2.main --tune-only        # Only tune, skip evaluation
python -m phase2.main --eval-only        # Load saved params, evaluate only
python -m phase2.main --models RF MLP    # Tune only RF and MLP
```
