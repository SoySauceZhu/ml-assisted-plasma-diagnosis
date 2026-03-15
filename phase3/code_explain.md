# Phase 3: Domain-Knowledge Feature Engineering — Code Architecture

## Overview

Phase 3 replaces the blind PCA (701 wavelengths → 11 components) with **13 hand-crafted OES features** grounded in plasma chemistry. This is the breakthrough phase: Ridge Config C jumps from R2 = -0.17 (Phase 1) to 0.80, confirming that domain knowledge matters more than model complexity on small datasets.

The 13 features fall into 3 categories:
- **7 single-wavelength intensities**: OH 309, O 777, Halpha 656, Hbeta 486, N2 337, CO2+ 406, C2 516 nm
- **3 band integrals**: OH 306-312, CO2+ 398-412, CO+Hbeta 460-500 nm (noise-robust)
- **3 intensity ratios**: OH/Halpha, O/OH, Halpha/Hbeta (self-normalising diagnostics)

**Input**: Same raw dataset + wavelength array
**Output**: 13-feature matrix, tuned model results, 3-phase comparison table

## File Structure

```
phase3/
├── config.py               # Feature definitions, model params, paths
├── feature_engineer.py     # Extract 13 OES features from raw spectra
├── tuner_rf.py             # Optuna tuning for RF on engineered features
├── tuner_mlp.py            # Optuna tuning for MLP (MLPNetBN class)
├── evaluation.py           # LOOCV with engineered features, 3-phase comparison
├── plotting.py             # Optuna history, pred vs actual, 3-way comparison
└── main.py                 # CLI entry point
```

## Data Flow

```
main.py
  │
  ├─► phase1.data_loader.prepare_data()     # Reuses Phase 1 data loading
  │
  ├─► feature_engineer.extract_oes_features(oes_raw, wavelengths)
  │     └─► Returns (20, 13) feature matrix + feature_names list
  │
  ├─► TUNING (RF and MLP only, for each config):
  │     tuner_rf.tune_rf(oes_features, data, config)
  │     tuner_mlp.tune_mlp(oes_features, data, config)
  │       └─► Optuna study → inner LOOCV on 13 features → best params
  │     └─► Saves tuned_hyperparameters.json
  │
  ├─► evaluation.run_all_evaluations(oes_features, data, tuned_params)
  │     └─► For each model (Ridge, PLS, RF, MLP) x config (A, B, C):
  │           run_loocv_for_model()
  │             └─► _scale_features() → get_input_config() → train → predict
  │     └─► Saves phase3_loocv_results_summary.csv, phase3_predictions_detail.csv
  │
  ├─► evaluation.build_comparison_table(phase3_df)
  │     └─► Loads Phase 1 (+ Phase 2) results, computes Delta_R2
  │
  └─► plotting.generate_all_phase3_plots()
```

## Detailed File Documentation

---

### config.py

**Purpose**: Feature definitions and Phase 3 configuration.

Key constants:

| Constant | Value | Description |
|---|---|---|
| `SINGLE_WAVELENGTHS` | 7 entries: `{name, wavelength, species}` | Single-point OES features (e.g., `{"name": "I_309_OH", "wavelength": 309, "species": "OH"}`) |
| `BAND_INTEGRALS` | 3 entries: `{name, start, end, species}` | Band integral features (e.g., `{"name": "band_OH_306_312", "start": 306, "end": 312}`) |
| `INTENSITY_RATIOS` | 3 entries: `{name, numerator, denominator}` | Ratio features (e.g., `{"name": "ratio_309_656", "numerator": 309, "denominator": 656}`) |
| `RF_PARAMS` | `{n_estimators=200, max_depth=4, ...}` | Default RF hyperparameters |
| `MLP_CONFIG` | `{hidden_sizes=[16], dropout=0.4, ...}` | Default MLP config (smaller network than Phase 1 due to fewer features) |
| `N_TRIALS_RF` | `200` | Optuna trials for RF |
| `N_TRIALS_MLP` | `100` | Optuna trials for MLP |

---

### feature_engineer.py

**Purpose**: The core innovation of Phase 3. Extracts 13 physically meaningful features from the raw 701-point OES spectrum.

| Function | Signature | Description |
|---|---|---|
| `extract_oes_features` | `(oes_raw, wavelengths) -> (features, feature_names)` | **Key function**. Takes (n, 701) raw OES and (701,) wavelength array. Returns (n, 13) feature matrix and list of 13 names. |

**Internal process**:
1. Builds `{wavelength: index}` mapping for fast lookup
2. **Single wavelengths** (7 features): Extracts intensity at exact wavelength index (e.g., `I_309 = oes_raw[:, idx[309]]`)
3. **Band integrals** (3 features): Slices spectrum between start/end wavelengths, computes area via `np.trapezoid(intensities, wavelengths_slice)` — trapezoidal numerical integration
4. **Intensity ratios** (3 features): Divides two wavelength intensities with epsilon=1e-10 protection against division by zero (e.g., `ratio_309_656 = I_309 / (I_656 + eps)`)

---

### tuner_rf.py

**Purpose**: Optuna tuning for RF on the 13 engineered features (instead of PCA components).

| Function | Signature | Description |
|---|---|---|
| `_postprocess_rf_params` | `(raw_params) -> dict` | Same logic as Phase 2: converts max_depth=0 to None, string max_features to float. |
| `rf_objective` | `(trial, oes_features, data, config_name) -> float` | Optuna objective. Note: takes `oes_features` (13-dim) instead of raw OES. Suggests same RF hyperparameters as Phase 2. Runs inner LOOCV using Phase 3's `_scale_features` and `get_input_config`. Returns R2. |
| `tune_rf` | `(oes_features, data, config_name, n_trials=200) -> (dict, Study)` | Creates Optuna study, runs optimisation. Returns best params and study. |

**Key difference from Phase 2**: Input is (20, 13) engineered features, not (20, 701) raw OES or PCA components. No PCA step needed.

---

### tuner_mlp.py

**Purpose**: Optuna tuning for MLP on engineered features.

**`class MLPNetBN(nn.Module)`**: Same as Phase 2 — MLP with optional batch normalisation.

| Function | Signature | Description |
|---|---|---|
| `_train_mlp` | `(model, X_train, y_train, cfg) -> nn.Module` | Adam optimiser + MSELoss + early stopping. Returns trained model. |
| `_parse_hidden_sizes` | `(s) -> list[int]` | Parses "16_8" → [16, 8]. |
| `_postprocess_mlp_config` | `(raw_params) -> dict` | Converts Optuna params to valid config. |
| `mlp_objective` | `(trial, oes_features, data, config_name) -> float` | Optuna objective on 13 engineered features. Search space: hidden_sizes [8/16/32/8_4/16_8], dropout, weight_decay, lr, epochs, patience, batch_norm. |
| `tune_mlp` | `(oes_features, data, config_name, n_trials=100) -> (dict, Study)` | Runs Optuna optimisation. |

---

### evaluation.py

**Purpose**: LOOCV evaluation with 13 engineered features. Key difference from Phase 1: no PCA step, uses `_scale_features` instead of `_scale_and_pca`.

| Function | Signature | Description |
|---|---|---|
| `_scale_features` | `(oes_train, oes_test, dis_train, dis_test) -> tuple` | Fits StandardScaler on training OES (13 features) and discharge (4 features) separately. Transforms both train and test. **No PCA** — features are already low-dimensional. |
| `get_input_config` | `(config_name, oes_tr, oes_te, dis_tr, dis_te) -> (X_train, X_test)` | Assembles input: **A** = 13 OES features, **B** = 4 discharge params, **C** = hstack(13 OES + 4 discharge) = 17 features. |
| `run_loocv_for_model` | `(model_name, oes_features, data, config_name, params=None) -> dict` | LOOCV for one model-config. For Ridge/PLS/RF: uses Phase 1 model classes. For MLP: instantiates MLPNetBN with config, trains via _train_mlp. Returns R2, RMSE, MAE, predictions. |
| `run_all_evaluations` | `(oes_features, data, tuned_params=None) -> (list, DataFrame)` | Iterates all 4 models (Ridge, PLS, RF, MLP) x 3 configs. Saves summary CSV and per-sample predictions CSV. |
| `build_comparison_table` | `(phase3_df) -> DataFrame` | Loads Phase 1 and Phase 2 results, merges with Phase 3, computes Delta_R2_P3_vs_P1. Saves 3-phase comparison CSV. |

---

### plotting.py

**Purpose**: Visualisation of Phase 3 results and cross-phase comparison.

| Function | Signature | Description |
|---|---|---|
| `plot_optimization_history` | `(study, model_name, config_name, save_dir=None)` | Optuna trial R2 scatter + best-so-far curve. |
| `plot_param_importances` | `(study, model_name, config_name, save_dir=None)` | Hyperparameter importance bar chart. |
| `plot_predicted_vs_actual` | `(all_results, save_dir=None)` | Scatter plots for each model-config with R2/RMSE in title. |
| `plot_three_way_comparison` | `(comparison_df, save_dir=None)` | Grouped bar chart: Phase 1 vs Phase 2 vs Phase 3 R2 side by side. Detects whether Phase 2 data is present and adjusts layout. |
| `generate_all_phase3_plots` | `(all_results, results_df, studies_dict, comparison_df)` | **Master function**. Generates all plots and saves to FIGURES_DIR. |

---

### main.py

**Purpose**: CLI entry point for Phase 3.

| Function | Signature | Description |
|---|---|---|
| `_serialize_params` | `(tuned_params) -> dict` | Converts `{(model, config): params}` to JSON format. |
| `_deserialize_params` | `(raw) -> dict` | Reverses serialisation. |
| `main` | `()` | Parses args: `--initial-only` (skip tuning), `--tune-only`, `--eval-only`, `--models`. Pipeline: data load → extract_oes_features → tune RF/MLP → evaluate all models → build comparison → plot. |

**Usage**:
```bash
python -m phase3.main                    # Full pipeline (tune + evaluate)
python -m phase3.main --initial-only     # Evaluate with default params (no tuning)
python -m phase3.main --eval-only        # Load saved tuned params, evaluate only
python -m phase3.main --models Ridge PLS # Evaluate only Ridge and PLS
```
