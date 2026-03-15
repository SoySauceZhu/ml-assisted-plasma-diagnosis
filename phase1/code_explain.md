# Phase 1: Baseline ML Pipeline — Code Architecture

## Overview

Phase 1 establishes baseline ML performance for predicting H2O2 yield rate from OES data. It evaluates **7 models** (Ridge, PLS, SVR, XGBoost, RF, MLP, CNN) across **3 input configurations** (A: OES only, B: discharge params only, C: OES + discharge) using **LOOCV** (Leave-One-Out Cross-Validation). PCA reduces the 701-wavelength OES spectrum to 11 components (95% variance).

**Input**: `oes_ml_dataset_1nm.csv` (20 samples, 701 OES wavelengths + 4 discharge params + H2O2 target)
**Output**: R2/RMSE/MAE per model-config pair, scatter plots, heatmaps

## File Structure

```
phase1/
├── config.py                 # All hyperparameters and paths
├── data_loader.py            # Load CSV, separate features, baseline correction
├── pca_analysis.py           # PCA fitting, variance plots, loading plots
├── evaluation.py             # LOOCV loop for all model x config combos
├── plotting.py               # Result visualisation (scatter, heatmap, bar)
├── main.py                   # CLI entry point
└── models/
    ├── ridge.py              # Ridge regression (RidgeCV)
    ├── pls.py                # Partial Least Squares
    ├── svr.py                # Support Vector Regression
    ├── xgboost_model.py      # XGBoost
    ├── rf.py                 # Random Forest
    ├── mlp.py                # PyTorch fully-connected network
    └── cnn.py                # PyTorch 1D convolutional network
```

## Data Flow

```
main.py
  │
  ├─► data_loader.prepare_data()
  │     └─► load_dataset() → separate_features() → baseline_correction()
  │     └─► Returns dict: {oes_raw, discharge_raw, target, wavelengths, sample_info}
  │
  ├─► pca_analysis.run_pca_analysis(data)
  │     └─► fit_pca() → determine_optimal_k() → plot_scree/cumvar/loadings/scores
  │     └─► Returns (PCA object, k=11)
  │
  ├─► evaluation.run_all_evaluations(data, pca_k)
  │     └─► For each model x config:
  │           run_loocv_for_model()
  │             └─► For each LOOCV fold:
  │                   _scale_and_pca() → get_input_config() → model.fit() → model.predict()
  │     └─► Returns (results_list, summary_df)
  │
  └─► plotting.generate_all_plots(results, summary_df)
```

## Detailed File Documentation

---

### config.py

**Purpose**: Centralised configuration — all paths, column names, hyperparameters, and constants.

No functions. Key constants:

| Constant | Value | Description |
|---|---|---|
| `DATA_PATH` | `../oes_ml_dataset_1nm.csv` | Path to the raw dataset |
| `META_COLS` | `["sheet", "condition"]` | Metadata columns to exclude from features |
| `DISCHARGE_COLS` | `["frequency_hz", "pulse_width_ns", "rise_time_ns", "flow_rate_sccm"]` | 4 discharge parameters |
| `TARGET_COL` | `"h2o2_rate"` | Prediction target |
| `OES_PREFIX` | `"I_"` | Prefix identifying OES wavelength columns (I_200 to I_900) |
| `PCA_VARIANCE_THRESHOLD` | `0.95` | Cumulative variance to retain in PCA |
| `RIDGE_ALPHAS` | `[0.01, 0.1, 1.0, 10.0, 100.0]` | Candidate regularisation strengths |
| `PLS_MAX_COMPONENTS` | `10` | Maximum PLS components to try |
| `SVR_PARAM_GRID` | `{C, epsilon, gamma}` | Grid search space for SVR |
| `MLP_CONFIG` | `{hidden_sizes=[32,16], dropout=0.4, lr=1e-3, ...}` | MLP architecture and training |
| `CNN_CONFIG` | `{conv_channels=[16,32], kernel_size=7, ...}` | CNN architecture and training |
| `RANDOM_SEED` | `42` | Reproducibility seed |

---

### data_loader.py

**Purpose**: Load and preprocess the raw CSV dataset into separate feature groups.

| Function | Signature | Description |
|---|---|---|
| `load_dataset` | `(csv_path=None) -> pd.DataFrame` | Reads CSV file from disk (default: DATA_PATH). Returns full DataFrame. |
| `separate_features` | `(df) -> (oes_df, discharge_df, target)` | Splits DataFrame into OES columns (prefix "I_"), discharge columns, and target array. |
| `baseline_correction` | `(oes_df, df) -> pd.DataFrame` | Subtracts mean spectrum of pulse_width=0 samples (near-zero H2O2 reference) from all OES spectra. If no baseline samples exist, returns original. |
| `prepare_data` | `(csv_path=None) -> dict` | **Master function**. Calls load → separate → baseline_correct. Returns dict with keys: `oes_raw` (n,701), `discharge_raw` (n,4), `target` (n,), `wavelengths` (701,), `sample_info`. |

---

### pca_analysis.py

**Purpose**: Dimensionality reduction of 701 OES wavelengths via PCA. Determines optimal number of components and generates diagnostic plots.

| Function | Signature | Description |
|---|---|---|
| `fit_pca` | `(oes_scaled, n_components=None) -> (PCA, scores)` | Fits sklearn PCA to scaled OES data. Returns fitted PCA object and transformed score matrix. |
| `determine_optimal_k` | `(pca, threshold=0.95) -> int` | Finds minimum k such that cumulative explained variance >= threshold. |
| `plot_scree` | `(pca, save_path=None)` | Bar chart of individual explained variance per component. |
| `plot_cumulative_variance` | `(pca, k_optimal, save_path=None)` | Line plot of cumulative variance with threshold line and optimal k marker. |
| `plot_loadings` | `(pca, wavelengths, n_components=3, save_path_prefix=None)` | Plots PCA loading vectors vs wavelength, annotates diagnostic spectral lines (OH 309, N2 337, Halpha 656, O 777). |
| `plot_scores_2d` | `(scores, target, sample_info, save_path=None)` | PC1 vs PC2 scatter, colour-coded by H2O2 rate, annotated with experimental conditions. |
| `run_pca_analysis` | `(data) -> (PCA, k)` | **Master function**. Standardises OES → fits PCA → determines k → generates all 4 plot types. Returns fitted PCA and optimal k. |

---

### evaluation.py

**Purpose**: LOOCV evaluation framework. For each of the 20 folds, scales data, applies PCA, assembles the correct input configuration, trains a model, and collects predictions.

| Function | Signature | Description |
|---|---|---|
| `compute_metrics` | `(y_true, y_pred) -> dict` | Computes R2, RMSE, MAE from true vs predicted arrays. |
| `_scale_and_pca` | `(oes_train, oes_test, dis_train, dis_test, pca_k) -> tuple` | Fits StandardScaler on training OES & discharge, transforms both. Fits PCA on scaled training OES, transforms both. Returns 6 arrays: (pca_train, pca_test, dis_train_scaled, dis_test_scaled, oes_train_scaled, oes_test_scaled). |
| `get_input_config` | `(config_name, pca_tr, pca_te, dis_tr, dis_te, oes_tr=None, oes_te=None, is_cnn=False) -> (X_train, X_test)` | Assembles feature matrix based on config: **A** = PCA features (or raw OES for CNN), **B** = discharge only, **C** = PCA + discharge (or raw OES + discharge for CNN). For CNN Config C, returns extra tuple for the two-input architecture. |
| `_create_model` | `(model_name) -> model` | Factory function. Returns an instance of the requested model class (Ridge, PLS, SVR, XGBoost, RF, MLP, or CNN). |
| `run_loocv_for_model` | `(model_name, data, pca_k, config_name) -> dict` | Runs full LOOCV for one model x config. Iterates 20 folds, calls _scale_and_pca → get_input_config → model.fit → model.predict. Returns dict with R2, RMSE, MAE, y_true, y_pred arrays. |
| `run_all_evaluations` | `(data, pca_k) -> (list, DataFrame)` | Iterates all 7 models x 3 configs (skips CNN Config B). Calls run_loocv_for_model for each. Saves CSV tables. Returns results list and summary DataFrame. |

---

### plotting.py

**Purpose**: Visualisation of Phase 1 results.

| Function | Signature | Description |
|---|---|---|
| `plot_predicted_vs_actual` | `(result, save_path=None)` | Single scatter plot: predicted vs actual H2O2, with 1:1 line and R2/RMSE in title. |
| `plot_all_predicted_vs_actual` | `(all_results, save_dir=None)` | Grid of scatter plots (7 models x 3 configs). Marks unavailable combos as "N/A". |
| `plot_summary_heatmap` | `(results_df, metric, save_path=None)` | Heatmap of metric values (R2/RMSE/MAE): models on rows, configs on columns. Uses RdYlGn colourmap for R2. |
| `plot_model_comparison_bar` | `(results_df, save_path=None)` | Grouped bar chart of R2 by model, with separate bars per config (A/B/C). |
| `generate_all_plots` | `(all_results, results_df)` | **Master function**. Calls all 4 plot functions and saves to FIGURES_DIR. |

---

### main.py

**Purpose**: Command-line entry point for Phase 1.

| Function | Signature | Description |
|---|---|---|
| `main` | `()` | Parses arguments (`--pca-only`, `--eval-only`, `--pca-k`). Sets random seeds (numpy, torch). Calls prepare_data → run_pca_analysis → run_all_evaluations → generate_all_plots. |

**Usage**:
```bash
python -m phase1.main              # Full pipeline
python -m phase1.main --pca-only   # Only PCA analysis
python -m phase1.main --eval-only --pca-k 11  # Skip PCA, evaluate with k=11
```

---

### models/ridge.py — `class RidgeModel`

**Purpose**: Ridge regression with built-in cross-validated alpha selection.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(alphas=None)` | Stores candidate alpha values (default: RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100]). |
| `fit` | `(X_train, y_train)` | Trains sklearn `RidgeCV` with internal LOOCV to select best alpha. |
| `predict` | `(X_test) -> np.ndarray` | Returns predictions. Handles 1D input via `atleast_2d`. |

---

### models/pls.py — `class PLSModel`

**Purpose**: Partial Least Squares regression with adaptive component selection. PLS simultaneously reduces dimensionality and regresses — the standard chemometric method.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(max_components=None)` | Sets maximum components to try (default: 10). |
| `fit` | `(X_train, y_train)` | Inner LOOCV to find optimal k (1 to max_k) by minimising MSE. Fits final PLS model with best k on full training data. |
| `predict` | `(X_test) -> np.ndarray` | Returns predictions via fitted PLS. |

---

### models/svr.py — `class SVRModel`

**Purpose**: Support Vector Regression with RBF kernel and grid-searched hyperparameters.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(param_grid=None)` | Stores hyperparameter grid: C, epsilon, gamma (default: SVR_PARAM_GRID). |
| `fit` | `(X_train, y_train)` | Runs `GridSearchCV` with LOOCV over param_grid (scoring: neg_MSE, parallel with n_jobs=-1). Stores best estimator. |
| `predict` | `(X_test) -> np.ndarray` | Returns predictions from best SVR. |

---

### models/xgboost_model.py — `class XGBoostModel`

**Purpose**: Gradient boosting via XGBoost with conservative hyperparameters (max_depth=2, n_estimators=50) to mitigate overfitting on n=20.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(params=None)` | Stores XGBoost parameters (default: XGBOOST_PARAMS). |
| `fit` | `(X_train, y_train)` | Trains `XGBRegressor` with fixed params and RANDOM_SEED. |
| `predict` | `(X_test) -> np.ndarray` | Returns predictions. |

---

### models/rf.py — `class RFModel`

**Purpose**: Random Forest regression with fixed hyperparameters.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(params=None)` | Stores RF parameters (default: n_estimators=100, max_depth=3). |
| `fit` | `(X_train, y_train)` | Trains `RandomForestRegressor` with fixed params and RANDOM_SEED. |
| `predict` | `(X_test) -> np.ndarray` | Returns predictions. |

---

### models/mlp.py — `class MLPNet` (nn.Module) + `class MLPModel`

**Purpose**: Fully-connected neural network (Multi-Layer Perceptron) built in PyTorch with early stopping.

**`MLPNet`** (the network architecture):

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(input_dim, hidden_sizes, dropout)` | Builds sequential layers: Linear → ReLU → Dropout for each hidden size, then Linear(last_hidden, 1) output. Default: [32, 16] hidden with 0.4 dropout. |
| `forward` | `(x) -> Tensor` | Forward pass through all layers. Squeezes output to 1D. |

**`MLPModel`** (the training wrapper):

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(config=None)` | Stores training config: hidden_sizes, dropout, weight_decay, lr, max_epochs, patience (default: MLP_CONFIG). |
| `fit` | `(X_train, y_train)` | Converts to float32 tensors. Trains with Adam optimiser and MSELoss. Implements early stopping: saves best model state, restores it after patience epochs with no improvement. |
| `predict` | `(X_test) -> np.ndarray` | Sets model to eval mode, disables gradients, returns numpy predictions. |

---

### models/cnn.py — `class CNN1D` (nn.Module) + `class CNNModel`

**Purpose**: 1D Convolutional Neural Network for processing raw 701-point OES spectra. Can optionally accept extra features (discharge params for Config C) via a two-input architecture.

**`CNN1D`** (the network architecture):

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(input_length, conv_channels, kernel_size, dropout, n_extra_features=0)` | Builds: Conv1d layers with ReLU → MaxPool1d after first conv → AdaptiveAvgPool1d(1) → Dropout → Linear head. If n_extra_features > 0 (Config C), concatenates extra features before the output layer. Default: [16, 32] channels, kernel_size=7. |
| `forward` | `(x_oes, x_extra=None) -> Tensor` | x_oes shape: (batch, 1, 701). Passes through conv layers, pools, flattens. Concatenates x_extra (discharge features) if provided. Returns (batch,) predictions. |

**`CNNModel`** (the training wrapper):

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(config=None)` | Stores CNN training config (default: CNN_CONFIG). |
| `fit` | `(X_train_oes, y_train, X_train_extra=None)` | Reshapes OES to (N, 1, 701) for Conv1d. Trains with Adam + MSELoss + early stopping. X_train_extra is passed for Config C (discharge features). |
| `predict` | `(X_test_oes, X_test_extra=None) -> np.ndarray` | Eval mode inference. Returns numpy predictions. |
