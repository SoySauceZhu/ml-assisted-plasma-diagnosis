# Phase 3: plan

## Background
In previous two phases, different type of model are tuned and trained on the 20 samples. The result is summarized here:

- phase 1 result:

| Model | Config A (OES only) | Config B (Params only) | Config C (OES + Params) |
|-------|:---:|:---:|:---:|
| **Ridge** | R²= −0.31 | **R²= 0.90** | R²= −0.17 |
| **PLS** | R²= −0.60 | **R²= 0.90** | R²= 0.63 |
| **SVR** | R²= 0.05 | R²= 0.62 | R²= 0.09 |
| **XGBoost** | R²= −0.11 | R²= −0.11 | R²= −0.11 |
| **RF** | R²= 0.04 | R²= 0.38 | R²= 0.24 |
| **MLP** | R²= −0.85 | R²= 0.57 | R²= −1.13 |
| **CNN** | R²= 0.30 | N/A | **R²= 0.69** |

- phase 2 result:

| Model | Config | R² (Phase 1) | R² (Phase 2) | ΔR² | RMSE (P1) | RMSE (P2) | ΔRMSE |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RF** | A | 0.04 | 0.22 | +0.18 | 0.227 | 0.204 | −0.023 |
| **RF** | B | 0.38 | **0.75** | +0.37 | 0.182 | 0.116 | −0.066 |
| **RF** | C | 0.24 | 0.46 | +0.22 | 0.202 | 0.170 | −0.031 |
| **MLP** | A | −0.85 | 0.37 | +1.22 | 0.314 | 0.183 | −0.131 |
| **MLP** | B | 0.57 | **0.86** | +0.29 | 0.152 | 0.086 | −0.066 |
| **MLP** | C | −1.13 | 0.37 | +1.50 | 0.337 | 0.184 | −0.154 |
| **CNN** | A | 0.30 | 0.53 | +0.23 | 0.193 | 0.158 | −0.035 |
| **CNN** | C | 0.69 | **0.77** | +0.09 | 0.129 | 0.110 | −0.019 |


In phase 3, instead of using automatically selected Principal Components, manually selected features will be used as inputs to the models. The features are suggested in 'phase3_feature_engineering.md'.

## Phase 3 goal:
- Keep following the working style in phase 1 and phase 2
- Build machine learning pipeline based on domain-knowledge selected features(given in 'phase2_feature_engineering.md'). Use four models: Ridge, PLS, RF, MLP
- Use all 13 features listed in phase2_feature_engineering as OES feature. 
- Use 3 Configs: (A) OES feature only. (B) discharge parameters only. (C) OES features + discharge parameters. 
- Firstly suggest the setting/parameter of RF, MLP, use your suggest meta param to do training and evaluation
- Then suggest the method to automatically find the best parameters for Random Forest and MLP models
- Generate you response for phase3 action and append to the following section "#Phase3: action"

# Phase 3: action

## Problem Recap

Phase 1 and 2 used blind PCA (701 → 11 components) to represent OES data. The key limitation: with 20 samples and 11 PCA components, the feature-to-sample ratio (p/n = 0.55) was too high for most models to generalise. OES-based models consistently underperformed discharge-parameter-only models.

Phase 3 replaces PCA with **13 domain-knowledge-driven features** (p/n = 0.65 for Config A, p/n = 0.85 for Config C with 17 total features). While the ratio is slightly higher numerically, these features are physically grounded and non-redundant, which should provide better signal-to-noise for learning.

### Key change from Phase 1/2

| Aspect | Phase 1/2 | Phase 3 |
|--------|-----------|---------|
| OES representation | PCA (11 blind components from 701 dims) | 13 handcrafted features (intensities, band integrals, ratios) |
| Config A input | 11 PCA scores | 13 engineered features |
| Config B input | 4 discharge parameters | 4 discharge parameters (unchanged) |
| Config C input | 11 PCA + 4 params = 15 features | 13 OES features + 4 params = 17 features |
| Models | Ridge, PLS, SVR, XGBoost, RF, MLP, CNN | Ridge, PLS, RF, MLP |

### Why CNN is excluded from Phase 3

In Phase 1/2, CNN's strength was operating on the **raw 701-dim continuous spectrum** — convolution could learn spectral features (peak shapes, band structures) directly. This is why CNN was the only model where OES actually helped (R² = 0.77 on Config C in Phase 2). With Phase 3's 13 engineered features, the input is no longer a continuous spectrum but a set of discrete, heterogeneous measurements (intensities, integrals, ratios). A 1D-CNN on length-13 tabular data offers no architectural advantage over MLP and adds unnecessary complexity. CNN's Phase 1/2 results on raw spectra remain valid as a comparison point for the paper.

---

## 1. Feature Extraction Pipeline

### 1.1 Feature definitions (from phase3_feature_engineering.md)

The 13 features are extracted from the baseline-corrected OES spectra:

```python
# Category 1: Single-wavelength intensities (7 features)
F1  = I_309                          # OH (A-X) 0-0 band head
F2  = I_777                          # Atomic O triplet
F3  = I_656                          # Hα
F4  = I_486                          # Hβ
F5  = I_337                          # N₂ SPS 0-0
F6  = I_406                          # CO₂⁺ FDB
F7  = I_516                          # C₂ Swan Δv=0

# Category 2: Band integrals (3 features)
F8  = ∫I(306–312 nm)                 # OH 0-0 band integral
F9  = ∫I(398–412 nm)                 # CO₂⁺ FDB band integral
F10 = ∫I(460–500 nm)                 # CO Angstrom + Hβ composite

# Category 3: Intensity ratios (3 features)
F11 = I_309 / I_656                  # OH-to-Hα ratio
F12 = I_777 / I_309                  # Atomic O-to-OH ratio
F13 = I_656 / I_486                  # Balmer decrement
```

### 1.2 Implementation approach

Add a `feature_engineer.py` module in `phase3/` that:
1. Takes baseline-corrected OES spectra (20 × 701) and wavelength array as input.
2. Extracts single-wavelength intensities by finding the nearest wavelength index (the dataset uses 1 nm resolution, so I_309 maps to column `I_309`).
3. Computes band integrals using trapezoidal summation over the relevant wavelength ranges.
4. Computes intensity ratios with a small epsilon (1e-10) to avoid division by zero.
5. Returns a (20 × 13) feature matrix with named columns F1–F13.

---

## 2. Input Configurations

| Config | Features | Dimensionality |
|--------|----------|:-:|
| **A** | F1–F13 (OES features only) | 13 |
| **B** | frequency_hz, pulse_width_ns, rise_time_ns, flow_rate_sccm | 4 |
| **C** | F1–F13 + discharge parameters | 17 |

All features are StandardScaler-normalised within each LOOCV fold (fit on train, transform on test), consistent with Phase 1/2.

---

## 3. Model Configurations

### 3.1 Ridge Regression

No hyperparameters to tune beyond regularisation strength, which is handled automatically by `RidgeCV`.

```python
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]  # same as Phase 1
```

**Rationale:** Ridge with 13 features and 20 samples is well-conditioned (p/n < 1). This should serve as a strong baseline. With physically meaningful features, Ridge may outperform its Phase 1 OES results (R² = −0.31 on Config A) significantly.

### 3.2 PLS Regression

PLS selects the optimal number of components via inner LOOCV (same mechanism as Phase 1).

```python
PLS_MAX_COMPONENTS = 10  # min(10, n_features) applied internally
```

**Rationale:** PLS is designed for correlated predictors. The 13 features have natural correlations (e.g., F1 and F8 both measure OH; F3 and F4 both from hydrogen lines), so PLS should extract meaningful latent variables. With 13 features instead of 11 PCA components, PLS may find a more interpretable decomposition.

### 3.3 Random Forest — Initial Parameters

```python
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_split": 3,
    "min_samples_leaf": 2,
    "max_features": 0.8,
    "bootstrap": False,
}
```

**Rationale:** Based on Phase 2 tuning insights:
- `max_depth=4`: Phase 2 showed unconstrained depth (`None`) was best, but with 13 features instead of 15 (11 PCA + 4 params), moderate depth prevents overfitting while allowing enough expressivity.
- `bootstrap=False`: Consistently best in Phase 2 across all configs — with only 20 samples, bootstrap resampling loses too much data per tree.
- `max_features=0.8`: Phase 2 found 0.5 (Config B) and sqrt/log2 (Configs A/C) worked well; 0.8 is a reasonable starting point for 13 physically meaningful features that are less redundant than PCA components.
- `n_estimators=200`: Sufficient for stable predictions with low-dimensional input.

### 3.4 MLP — Initial Parameters

```python
MLP_CONFIG = {
    "hidden_sizes": [16],
    "dropout": 0.4,
    "weight_decay": 0.01,
    "lr": 0.004,
    "max_epochs": 500,
    "patience": 50,
    "batch_norm": True,
}
```

**Rationale:** Based on Phase 2 tuning insights:
- `hidden_sizes=[16]`: Phase 2 Config B (best MLP result, R² = 0.86) used a single layer of 16 neurons. With 13–17 input features (depending on config), a single layer of 16 neurons gives ~289 parameters for Config A (13×16 + 16 + 16×1 + 1), which is manageable for 20 samples.
- `batch_norm=True`: Consistently enabled across all Phase 2 best configs.
- `dropout=0.4` and `weight_decay=0.01`: Moderate regularisation. Phase 2 values ranged 0.37–0.47 for dropout and 0.003–0.069 for weight decay.
- `lr=0.004`: Near the optimal range found in Phase 2 (0.004–0.005).

---

## 4. Implementation Plan

### Step 1: Create `phase3/` directory structure

```
phase3/
├── __init__.py
├── config.py              # feature definitions, model configs, search spaces
├── feature_engineer.py    # extract 13 OES features from raw spectra
├── evaluation.py          # LOOCV with engineered features (replaces PCA)
├── tuner_rf.py            # RF hyperparameter tuning
├── tuner_mlp.py           # MLP hyperparameter tuning (Optuna)
├── plotting.py            # result visualisation
├── main.py                # orchestrator
└── results/
    ├── figures/
    └── tables/
```

### Step 2: Implement feature extraction

`feature_engineer.py` should:
1. Accept the output of `phase1.data_loader.prepare_data()`.
2. Extract F1–F13 from the baseline-corrected OES matrix using wavelength mapping.
3. Return the (20 × 13) feature matrix.

```python
def extract_oes_features(oes_raw, wavelengths):
    """Extract 13 domain-knowledge features from raw OES spectra.

    Args:
        oes_raw: (n_samples, 701) baseline-corrected OES intensities
        wavelengths: (701,) array of wavelength values (200–900 nm)

    Returns:
        features: (n_samples, 13) engineered feature matrix
        feature_names: list of 13 feature names
    """
```

### Step 3: Implement LOOCV evaluation

Modify the evaluation pipeline to:
1. Replace the PCA step with feature extraction.
2. For each LOOCV fold, fit StandardScaler on the 13 OES features (train), transform test.
3. For Config A: use 13 scaled OES features.
4. For Config B: use 4 scaled discharge parameters (same as Phase 1/2).
5. For Config C: concatenate 13 OES + 4 discharge features = 17 features.

### Step 4: Run initial evaluation with suggested parameters

Run all 4 models × 3 configs (12 combinations) using the parameters from Section 3.

### Step 5: Hyperparameter tuning for RF, MLP

After initial evaluation, run automated tuning (see Section 5 below).

### Step 6: Re-evaluate with tuned parameters

Same protocol as Phase 2: fix best parameters, run outer LOOCV, compare against initial Phase 3 results and Phase 1/2 results.

---

## 5. Hyperparameter Tuning Strategy

### Approach

Use the same **Approach B** from Phase 2: run Optuna/GridSearchCV once on all 20 samples with inner LOOCV, find best params, then run outer LOOCV with those fixed params.

### 5.1 Random Forest Search Space

```python
RF_SEARCH_SPACE = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 4, 5, None],
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": [1, 2, 3],
    "max_features": [0.5, 0.8, 1.0, "sqrt"],
    "bootstrap": [True, False],
}
```

**Tuning method:** `RandomizedSearchCV` with `cv=LeaveOneOut()`, ~200 iterations. RF is fast enough for this.

### 5.2 MLP Search Space

```python
MLP_SEARCH_SPACE = {
    "hidden_sizes": [[8], [16], [32], [8, 4], [16, 8]],
    "dropout": (0.3, 0.6),          # continuous, Optuna float
    "weight_decay": (1e-3, 1e-1),   # continuous, log-uniform
    "lr": (1e-4, 5e-3),             # continuous, log-uniform
    "max_epochs": [300, 500, 1000],
    "patience": [30, 50, 100],
    "batch_norm": [True, False],
}
```

**Tuning method:** Optuna with TPE sampler, `n_trials=100`.

**Key difference from Phase 2:** With 13 interpretable features instead of 11 PCA + 4 params, the MLP may benefit from slightly larger architectures (e.g., [16, 8]) since the features are less noisy and more physically meaningful. But we keep [32, 16] out of the search space to limit parameter count.

---

## 6. Expected Outcomes

### 6.1 Predictions by model

| Model | Config A (13 OES features) | Config B (4 params) | Config C (17 features) |
|-------|---|---|---|
| **Ridge** | Expect significant improvement over Phase 1 (R² = −0.31). With 13 meaningful features vs. 11 noisy PCA components, Ridge should achieve R² ≈ 0.3–0.6. | Same as Phase 1 (R² ≈ 0.90) — input unchanged. | Expect improvement over Phase 1 (R² = −0.17). With interpretable OES features, Config C may approach Config B. |
| **PLS** | Expect major improvement over Phase 1 (R² = −0.60). PLS thrives on correlated predictors with physical meaning. | Same as Phase 1 (R² ≈ 0.90). | Expect improvement over Phase 1 (R² = 0.63). |
| **RF** | Expect improvement over Phase 2 (R² = 0.22). Tree splits on individual wavelength features are more interpretable and effective than splits on PCA components. | Same as Phase 2 (R² ≈ 0.75). | Expect improvement over Phase 2 (R² = 0.46). |
| **MLP** | Expect improvement over Phase 2 (R² = 0.37). Physically meaningful features reduce the need for the network to learn feature extraction. | Same as Phase 2 (R² ≈ 0.86). | Expect improvement over Phase 2 (R² = 0.37). |

### 6.2 Key hypotheses to test

1. **Do domain-knowledge features outperform blind PCA for linear models (Ridge, PLS)?** If yes, this validates the feature engineering approach and demonstrates that OES contains predictive information that PCA failed to extract efficiently.

2. **Does Config C become competitive with Config B?** In Phase 1/2, adding OES (PCA) to discharge params often hurt performance. With 13 targeted features, Config C should at least not degrade Config B, and ideally improve it.

3. **Can MLP with engineered features match or beat CNN on raw spectra?** Phase 2's best CNN result was R² = 0.77 (Config C, raw 701-dim input). If MLP with 13 engineered features achieves comparable or better R², it demonstrates that domain-knowledge feature engineering on a simple model can replace end-to-end deep learning on small datasets.

---

## 7. Comparison Framework

Phase 3 results will be compared across three dimensions:

### 7.1 Phase 3 initial vs. Phase 3 tuned
Same as Phase 2: measure the effect of hyperparameter tuning on the new feature set.

### 7.2 Phase 3 (engineered features) vs. Phase 1/2 (PCA features)
The central comparison. For each model × config, compare:
- Phase 1 (default params, PCA) → Phase 2 (tuned params, PCA) → Phase 3 (tuned params, engineered features)
- This isolates the effect of feature engineering from hyperparameter tuning.

### 7.3 Cross-model ranking
Update the overall model ranking table with all Phase 3 results to determine the best model-config combination across the entire study.

---

## 8. Dependencies

```
optuna>=3.0          # reused from Phase 2
scikit-learn         # already installed
torch                # already installed
numpy, pandas        # already installed
matplotlib           # already installed
```

No new dependencies needed. Phase 3 reuses `phase1.data_loader.prepare_data()` for data loading and `phase1.evaluation.compute_metrics()` for evaluation metrics.

---

## 9. Compute Estimate

| Model | Inner LOOCV fits per trial | Trials | Total model fits | Estimated time |
|-------|:---:|:---:|:---:|---|
| Ridge | 20 (RidgeCV) | N/A | 20 | < 1 sec |
| PLS | 20 × k_max | N/A | ~200 | < 5 sec |
| RF | 20 | 200 (RandomizedSearchCV) | 4,000 | ~1–2 min |
| MLP | 19 | 100 (Optuna) | 1,900 | ~5–10 min |

**Total: ~10–15 minutes.** Much faster than Phase 2 since CNN (the slowest model on 701-dim input) is excluded, and MLP now processes 13–17 features instead of 15 PCA+params.
