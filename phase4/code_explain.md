# Phase 4: Interpretability, Stability & Feature Redundancy — Code Architecture

## Overview

Phase 4 shifts from "how well can we predict?" to "WHY do predictions work?", "HOW RELIABLE are they?", and "CAN we simplify further?". It performs 8 analyses:

1. **Multi-model feature importance** — Ridge coefficients, PLS VIP, RF permutation importance, MLP SHAP
2. **SHAP analysis** — KernelSHAP for MLP Config C
3. **Consensus ranking** — Average rank across 4 models, Spearman correlations
4. **Bootstrap confidence intervals** — 500 resamples for R2/RMSE
5. **Fold-to-fold stability** — CV of importance across 20 LOOCV folds
6. **Residual analysis** — Outlier detection, residual-feature correlations
7. **Feature redundancy** — VIF, backward elimination, category ablation, permutation test
8. **Publication figures** — 12 figures (Fig 1-12)

**Key finding**: 9/13 OES features are redundant (VIF > 10). A pruned model with just 3 intensity ratios + 4 discharge params achieves R2 = 0.920 (p < 0.0005).

## File Structure

```
phase4/
├── config.py                    # Feature names, analysis settings, paths
├── interpretability.py          # Feature importance from 4 models
├── shap_analysis.py             # KernelSHAP for MLP
├── stability.py                 # Bootstrap CIs, fold stability
├── feature_redundancy.py        # VIF, backward elimination, category ablation
├── feature_redundancy_eval.py   # Permutation test, supplementary figures (Fig 9-12)
├── residual_analysis.py         # Outlier detection, residual correlations
├── plotting.py                  # Publication figures (Fig 1-8)
└── main.py                      # Pipeline orchestrator (8 steps)
```

## Data Flow

```
main.py
  │
  ├─► phase1.data_loader.prepare_data()
  ├─► phase3.feature_engineer.extract_oes_features()
  ├─► Load phase3/tuned_hyperparameters.json
  │
  ├─► Step 1: FEATURE IMPORTANCE
  │     interpretability.ridge_importance_loocv()    → (20, 17) per-fold importance
  │     interpretability.pls_importance_loocv()      → (20, 17)
  │     interpretability.rf_importance_loocv()       → (20, 17)
  │
  ├─► Step 2: SHAP ANALYSIS
  │     shap_analysis.compute_shap_loocv()           → (20, 17) SHAP values
  │     shap_analysis.get_shap_importance()           → (17,) mean |SHAP|
  │
  ├─► Step 3: CONSENSUS RANKING
  │     interpretability.build_consensus_table()      → DataFrame with ranks
  │
  ├─► Step 4: BOOTSTRAP CIs
  │     stability.bootstrap_all_models()              → CI table for B and C configs
  │
  ├─► Step 5: FOLD STABILITY
  │     stability.fold_importance_stability()         → CV per feature per model
  │
  ├─► Step 6: RESIDUAL ANALYSIS
  │     residual_analysis.analyse_residuals()         → outlier flags
  │     residual_analysis.residual_feature_correlation() → which features predict error
  │
  ├─► Step 7: FEATURE REDUNDANCY
  │     feature_redundancy.compute_correlation_vif()  → VIF table
  │     feature_redundancy.ablation_backward_elimination() → R2 trajectory
  │     feature_redundancy.ablation_category()        → R2 by feature category
  │
  └─► Step 8: PLOTTING
        plotting.generate_all_phase4_plots()          → Fig 1-8 (PDF)
        feature_redundancy_eval.main()                → Fig 9-12 + permutation test
```

## Detailed File Documentation

---

### config.py

**Purpose**: Analysis configuration and feature name definitions.

Key constants:

| Constant | Value | Description |
|---|---|---|
| `OES_FEATURE_NAMES` | 13 names | The 13 engineered OES features from Phase 3 |
| `DISCHARGE_FEATURE_NAMES` | 4 names | `["frequency_hz", "pulse_width_ns", "rise_time_ns", "flow_rate_sccm"]` |
| `ALL_FEATURE_NAMES_C` | 17 names | OES + discharge concatenated (Config C feature order) |
| `MODELS_FOR_IMPORTANCE` | `["Ridge", "PLS", "RF", "MLP"]` | Models to extract importance from |
| `FOCUS_CONFIG` | `"C"` | All Phase 4 analysis focuses on Config C (OES + discharge) |
| `BOOTSTRAP_N_ITER` | `500` | Bootstrap resamples |
| `BOOTSTRAP_CI_LEVEL` | `0.95` | 95% confidence intervals |
| `FIGURE_DPI` | `300` | Publication-quality plots |
| `FIGURE_FORMAT` | `"pdf"` | Vector format for publication |

---

### interpretability.py

**Purpose**: Extract feature importance from 4 different models across all 20 LOOCV folds. Each fold gives a separate importance vector, capturing fold-to-fold variability.

| Function | Signature | Description |
|---|---|---|
| `_loocv_importance_loop` | `(oes_features, data, model_factory_fn, importance_extract_fn) -> (20, 17)` | **Generic LOOCV importance extractor**. For each of 20 folds: scales features → assembles Config C (17 features) → trains model via `model_factory_fn` → extracts importance via `importance_extract_fn`. Returns (20, 17) array of per-fold importances. |
| `_compute_vip` | `(pls_model) -> (p,)` | Computes Variable Importance in Projection (VIP) scores from a fitted PLS model. Formula: `VIP_j = sqrt(p * sum(w_aj^2 * SS_a) / sum(SS_a))`. Features with VIP > 1.0 are considered important. |
| `ridge_importance_loocv` | `(oes_features, data) -> (20, 17)` | Extracts absolute standardised Ridge coefficients per fold. Importance = \|beta_j\|. |
| `pls_importance_loocv` | `(oes_features, data) -> (20, 17)` | Extracts PLS VIP scores per fold. |
| `rf_importance_loocv` | `(oes_features, data, rf_params) -> (20, 17)` | Extracts RF permutation importance per fold (sklearn `permutation_importance` with n_repeats=10). Measures performance drop when each feature is shuffled. |
| `build_consensus_table` | `(ridge_imp, pls_imp, rf_imp, mlp_imp, feature_names) -> DataFrame` | **Consensus ranking**. For each model: computes mean importance across 20 folds → normalises to sum=1 → ranks features (1=most important). Consensus rank = average of 4 model ranks. Also computes Spearman rank correlations between model pairs and OES vs discharge importance fractions. |

---

### shap_analysis.py

**Purpose**: SHAP (SHapley Additive exPlanations) values for MLP Config C using KernelSHAP.

| Function | Signature | Description |
|---|---|---|
| `compute_shap_loocv` | `(oes_features, data, mlp_cfg) -> (shap_values, X_test_all)` | For each of 20 LOOCV folds: trains MLP on 19 samples → creates `shap.KernelExplainer` with 19 training samples as background → computes SHAP for 1 held-out sample (nsamples=200). Returns (20, 17) SHAP values and (20, 17) corresponding feature values. KernelSHAP is model-agnostic (treats MLP as black box). |
| `get_shap_importance` | `(shap_values) -> (17,)` | Computes mean(\|SHAP\|) per feature across 20 folds. This is the standard SHAP importance metric. |

---

### stability.py

**Purpose**: Statistical reliability assessment — bootstrap CIs for model performance and fold-to-fold importance stability.

| Function | Signature | Description |
|---|---|---|
| `bootstrap_metrics` | `(y_true, y_pred, n_iter=500, ci=0.95, seed=42) -> dict` | Resamples (y_true, y_pred) pairs with replacement n_iter times. Computes R2 and RMSE for each resample. Returns dict with: R2_mean, R2_lo (2.5th percentile), R2_hi (97.5th percentile), R2_std, RMSE_mean/lo/hi/std, R2_distribution array. Skips degenerate resamples (var < 1e-10). |
| `bootstrap_all_models` | `(predictions_path) -> DataFrame` | Loads Phase 3 per-sample predictions CSV. Runs bootstrap_metrics for each (model, config) pair (both B and C). Returns summary table with point estimates and 95% CIs. Stores `.distributions` dict for histogram plotting. |
| `fold_importance_stability` | `(ridge_imp, pls_imp, rf_imp, shap_vals, feature_names) -> DataFrame` | For each model and each of 17 features: computes CV (coefficient of variation = std/mean) across 20 LOOCV folds. Flags features as **unstable** if CV > 1.0. Key finding: Ridge has 6% instability (1/17 features), MLP has 82% (14/17). |

---

### feature_redundancy.py

**Purpose**: Multicollinearity detection and feature pruning via backward elimination.

| Function | Signature | Description |
|---|---|---|
| `compute_correlation_vif` | `(oes_features, feature_names) -> (corr_df, vif_df)` | Computes 13x13 Pearson correlation matrix among OES features. Computes Variance Inflation Factor (VIF) for each feature using `statsmodels.variance_inflation_factor`. VIF > 10 indicates severe multicollinearity. Returns correlation DataFrame and VIF table with is_high_vif flag. |
| `_run_ridge_loocv_subset` | `(oes_subset, data) -> dict` | Helper: runs Ridge LOOCV on Config C using a subset of OES features + all 4 discharge params. Manually constructs X = hstack(oes_scaled, discharge_scaled). Returns R2, RMSE, MAE. |
| `ablation_backward_elimination` | `(oes_features, data, feature_names) -> DataFrame` | **Backward elimination**: starting from all 13 OES features (R2 = 0.798), iteratively removes the feature with the smallest \|Ridge coefficient\| and re-evaluates. Continues until 3 OES features remain. Key result: R2 monotonically improves to 0.918 at 1 OES feature (ratio_656_486). Returns trajectory DataFrame with n_features, removed, remaining, R2, RMSE, MAE per step. |
| `ablation_category` | `(oes_features, data, feature_names) -> DataFrame` | **Category ablation**: evaluates Ridge Config C using each feature category alone: all 13, single-wavelength only (7), band integrals only (3), ratios only (3). Key result: 3 ratios + 4 discharge (R2 = 0.906) beats 7 single-wavelengths + 4 discharge (R2 = 0.823). |

---

### feature_redundancy_eval.py

**Purpose**: Supplementary analyses — MLP ablation, permutation test, and additional figures (Fig 9-12).

| Function | Signature | Description |
|---|---|---|
| `_load_data` | `() -> (data, oes_features)` | Helper to load dataset and extract OES features. |
| `_run_mlp_loocv_subset` | `(oes_subset, data, mlp_cfg) -> dict` | Like _run_ridge_loocv_subset but for MLP. |
| `_run_mlp_ablation_backward_elimination` | `(oes_features, data, feature_names, mlp_cfg) -> DataFrame` | MLP backward elimination using Ridge coefficient ranking to determine removal order. Shows whether the pruning trajectory holds for non-linear models too. |
| `run_permutation_test` | `(oes_features, data, n_permutations=2000) -> (observed_r2, null_array, p_value)` | **Statistical significance test**. Uses the optimal 3-ratio + 4-discharge model. Computes observed R2 via LOOCV. Then runs 2000 permutations: shuffles y_target, re-runs LOOCV, records null R2. p-value = proportion of null R2 >= observed R2. Result: p < 0.0005. |
| `fig9_ablation_trajectory` | `(ablation_ridge, ablation_mlp=None)` | Line plot: R2 vs number of OES features for backward elimination. Gold star marks optimal point. Reference lines at Config B (0.904) and full 13-feature (0.798). |
| `fig10_category_ablation` | `(ablation_ridge)` | Bar chart: R2 for each feature category (ratios, bands, single-wavelength, all). |
| `fig11_vif_barchart` | `(vif_df, optimal_features=None)` | VIF bar chart with log scale. Green (VIF <= 10) vs red (VIF > 10). Gold borders on optimal features. Threshold line at VIF = 10. |
| `fig12_permutation_test` | `(observed_r2, null_r2_array, p_value)` | Histogram of null R2 distribution with observed R2 vertical line and p-value annotation. |
| `create_ablation_summary` | `(ablation_ridge, ablation_mlp, category_ablation) -> DataFrame` | Combines all ablation results into one publication-ready table. |
| `main` | `()` | Runs all supplementary analyses: MLP ablation, permutation test, generates Fig 9-12, creates summary table. |

---

### residual_analysis.py

**Purpose**: Analyse prediction errors to identify systematic model weaknesses and outliers.

| Function | Signature | Description |
|---|---|---|
| `analyse_residuals` | `(predictions_path, data) -> DataFrame` | Loads Phase 3 per-sample predictions. Filters to Config C (Ridge and MLP). Computes residual = y_pred - y_true. Flags outliers where \|residual\| > 2 * std. Key finding: Sample 9 (2000 ns pulse_width, highest H2O2 = 0.83) is a consistent outlier for both models. |
| `residual_feature_correlation` | `(residual_df, oes_features, discharge, feature_names) -> DataFrame` | Correlates \|residual\| with each of 17 features using Pearson correlation. Identifies which features predict model error. Key finding: pulse_width_ns has r = 0.46 (p = 0.04) with Ridge \|residual\| — the model underperforms at extreme discharge conditions. |
| `condition_grouped_summary` | `(residual_df) -> DataFrame` | Groups residuals by experimental condition. Computes mean/std residual and mean \|residual\| per group. Identifies which experimental conditions are hardest to predict. |

---

### plotting.py

**Purpose**: Publication-quality figures (Fig 1-8) saved as PDF at 300 DPI.

| Function | Signature | Description |
|---|---|---|
| `_savefig` | `(fig, name)` | Helper: saves figure to FIGURES_DIR as PDF (or configured format) at FIGURE_DPI. |
| `fig1_importance_heatmap` | `(importance_df)` | **4-model feature importance heatmap**. 17 features (rows, sorted by consensus rank) x 4 models (columns). Cell values are normalised importance; annotations show rank numbers. |
| `fig2_shap_beeswarm` | `(shap_values, X_test_all, feature_names)` | **SHAP beeswarm plot** for MLP Config C. Each dot is one sample; x-axis = SHAP value (impact on prediction); colour = feature value (high/low). Uses `shap.summary_plot`. |
| `fig3_shap_dependence` | `(shap_values, X_test_all, feature_names, top_n=4)` | **SHAP dependence plots** for top 4 features. 2x2 grid. Each subplot: feature value (x) vs SHAP value (y), showing the marginal effect. Uses `shap.dependence_plot`. |
| `fig4_stability_errorbars` | `(stability_df)` | **Importance stability bar chart**. 2x2 grid (one per model). Each bar = mean importance with error bar = +/- 1 std across 20 folds. Visually shows MLP's high instability. |
| `fig5_bootstrap_distributions` | `(bootstrap_df)` | **Bootstrap R2 distributions**. 2x2 grid: Ridge B, Ridge C, MLP B, MLP C. Histograms with mean line, 95% CI shading, and point estimate marker. Shows CI overlap between Ridge C and MLP C. |
| `fig6_residual_pred_vs_actual` | `(residual_df, data)` | **Predicted vs actual scatter** for Ridge C and MLP C (1x2 layout). Colour-coded by experimental condition. 1:1 line + +/-1sigma bands. Outliers annotated. |
| `fig7_correlation_vif` | `(corr_df, vif_df)` | **Correlation matrix + VIF** (1x2 layout). Left: 13x13 lower-triangular Pearson correlation heatmap. Right: VIF bar chart with threshold at 10 (green/red colouring). |
| `fig8_chemistry_mapping` | `(importance_df=None)` | **Chemistry mapping table**. Maps each of 17 features to: species/parameter, pathway/role in H2O2 chemistry, mechanism. Rendered as a matplotlib table figure. |
| `generate_all_phase4_plots` | `(importance_df, shap_values, X_test_all, stability_df, bootstrap_df, residual_df, corr_df, vif_df, data)` | **Master function**. Calls Fig 1-8 sequentially. |

---

### main.py

**Purpose**: Orchestrates all Phase 4 analyses in 8 sequential steps.

| Function | Signature | Description |
|---|---|---|
| `main` | `()` | Full pipeline (no command-line args): |

**Step-by-step execution**:

1. **Initialisation**: Create output directories, set random seeds (numpy, torch), load data, extract OES features, load Phase 3 tuned hyperparameters
2. **Step 1 — Feature importance**: Extract per-fold importance for Ridge (|coefficients|), PLS (VIP), RF (permutation importance). Each returns (20, 17) array.
3. **Step 2 — SHAP**: Compute KernelSHAP for MLP Config C → (20, 17) SHAP values. Extract mean |SHAP| importance → (17,) vector. Save to CSV.
4. **Step 3 — Consensus**: Build consensus table from 4 importance sources. Compute Spearman rank correlations between model pairs. Compute OES vs discharge importance fractions.
5. **Step 4 — Bootstrap CIs**: Run 500-iteration bootstrap for all model-config pairs (B and C). Report 95% CIs.
6. **Step 5 — Fold stability**: Compute CV of importance across 20 folds per feature-model. Flag unstable features (CV > 1.0).
7. **Step 6 — Residuals**: Analyse residuals for Config C (Ridge, MLP). Compute residual-feature correlations. Group by condition.
8. **Step 7 — Feature redundancy**: Compute VIF, run backward elimination (R2 trajectory from 0.798 → 0.918), run category ablation.
9. **Step 8 — Plotting**: Generate all 8 publication figures.

**Usage**:
```bash
python -m phase4.main    # Runs all 8 steps sequentially
```

**Note**: Phase 4 has no `--tune-only` / `--eval-only` flags. It always runs the full analysis suite. The supplementary analyses (Fig 9-12, permutation test) are in `feature_redundancy_eval.py` and run separately:

```bash
python -m phase4.feature_redundancy_eval    # Supplementary analyses
```
