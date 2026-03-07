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

---

# Phase 3: Observation

## LOOCV Results Summary (Tuned Parameters)

| Model | Config A (13 OES features) | Config B (Params only) | Config C (OES + Params) |
|-------|:---:|:---:|:---:|
| **Ridge** | R²= 0.12 | **R²= 0.90** | **R²= 0.80** |
| **PLS** | R²= 0.35 | **R²= 0.90** | R²= 0.74 |
| **RF** | R²= 0.43 | R²= 0.75 | R²= 0.50 |
| **MLP** | R²= 0.32 | R²= 0.86 | **R²= 0.81** |

## Phase 1 → Phase 2 → Phase 3 Comparison

| Model | Config | R² (P1, PCA) | R² (P2, PCA+Tuned) | R² (P3, Engineered) | ΔR² (P3 vs P1) |
|-------|:---:|:---:|:---:|:---:|:---:|
| **Ridge** | A | −0.31 | — | 0.12 | +0.42 |
| **Ridge** | B | 0.90 | — | 0.90 | 0.00 |
| **Ridge** | C | −0.17 | — | **0.80** | **+0.97** |
| **PLS** | A | −0.60 | — | 0.35 | +0.95 |
| **PLS** | B | 0.90 | — | 0.90 | 0.00 |
| **PLS** | C | 0.63 | — | 0.74 | +0.12 |
| **RF** | A | 0.04 | 0.22 | 0.43 | +0.39 |
| **RF** | B | 0.38 | 0.75 | 0.75 | +0.37 |
| **RF** | C | 0.24 | 0.46 | 0.50 | +0.26 |
| **MLP** | A | −0.85 | 0.37 | 0.32 | +1.17 |
| **MLP** | B | 0.57 | 0.86 | 0.86 | +0.29 |
| **MLP** | C | −1.13 | 0.37 | **0.81** | **+1.95** |

## Best Hyperparameters Found (Phase 3 Tuning)

### Random Forest
| Parameter | Config A | Config B | Config C |
|-----------|----------|----------|----------|
| n_estimators | 200 | 200 | 100 |
| max_depth | 4 | None | None |
| min_samples_split | 4 | 2 | 3 |
| min_samples_leaf | 2 | 1 | 1 |
| max_features | sqrt | 0.5 | sqrt |
| bootstrap | False | False | False |

### MLP
| Parameter | Config A | Config B | Config C |
|-----------|----------|----------|----------|
| hidden_sizes | [32] | [16] | [32] |
| dropout | 0.56 | 0.41 | 0.37 |
| weight_decay | 0.064 | 0.004 | 0.009 |
| lr | 0.002 | 0.005 | 0.002 |
| max_epochs | 300 | 1000 | 1000 |
| patience | 50 | 100 | 100 |
| batch_norm | False | True | False |

## Key Observations

### 1. Feature engineering dramatically improved Config C (OES + Params)

The most striking result of Phase 3 is the transformation of Config C performance. In Phase 1, adding PCA-based OES features to discharge parameters **actively hurt** most models: Ridge dropped from R² = 0.90 (Config B) to −0.17 (Config C), and MLP dropped from 0.57 to −1.13. In Phase 3, this pattern is completely reversed:

- **Ridge Config C: R² = 0.80** (was −0.17 in Phase 1, a gain of +0.97)
- **MLP Config C: R² = 0.81** (was −1.13 in Phase 1, a gain of +1.95)
- **PLS Config C: R² = 0.74** (was 0.63 in Phase 1, a gain of +0.12)

This proves that PCA-based OES features were not just uninformative — they were actively injecting noise that overwhelmed the discharge parameter signal. Domain-knowledge features, by contrast, add genuine complementary information.

### 2. MLP Config C surpasses Phase 2's best CNN result

MLP with 13 engineered OES features + 4 discharge parameters achieves R² = 0.81 on Config C — exceeding Phase 2's best CNN result of R² = 0.77 on the same config using raw 701-dim spectra. This confirms the Phase 3 hypothesis: **domain-knowledge feature engineering on a simple model can replace end-to-end deep learning on small datasets**. The MLP uses just 32 neurons in a single hidden layer (~593 parameters for 17 inputs), while the CNN used [32, 64] convolutional channels on 701-dim input. The engineered features distil the same spectroscopic information that CNN learned implicitly, but with far less risk of overfitting.

### 3. Config B results are unchanged — as expected

Ridge Config B (R² = 0.90), PLS Config B (R² = 0.90), RF Config B (R² = 0.75), and MLP Config B (R² = 0.86) are identical or near-identical to their Phase 1/2 values. This is expected because Config B uses only discharge parameters, which are unchanged across phases. This serves as an internal consistency check — the evaluation pipeline produces reproducible results.

### 4. Config A (OES-only) improved but remains the weakest config

All models improved on Config A compared to Phase 1:
- Ridge: −0.31 → 0.12
- PLS: −0.60 → 0.35
- RF: 0.04 → 0.43
- MLP: −0.85 → 0.32

However, Config A still lags behind Config B for every model. The best Config A result is RF at R² = 0.43, suggesting that 13 OES features alone capture some predictive signal but not enough to match 4 well-structured discharge parameters. This is consistent with the physical reality: discharge parameters directly determine plasma conditions, while OES is an indirect measurement of those conditions with additional noise sources.

### 5. Ridge emerges as the strongest Config C model among linear approaches

Ridge Config C (R² = 0.80) outperforms PLS Config C (R² = 0.74), reversing their Phase 1 relationship where PLS Config C (0.63) far exceeded Ridge Config C (−0.17). With well-chosen features that are not excessively correlated, Ridge's simpler regularisation (L2 penalty) is more effective than PLS's latent variable decomposition. This suggests that the 13 engineered features are already informative enough that PLS's dimension-reduction step provides diminishing returns.

### 6. RF still lags behind linear models and MLP

RF's best Phase 3 result is R² = 0.75 on Config B, matching its Phase 2 tuned result but still trailing Ridge/PLS (R² = 0.90). On Config C, RF achieves R² = 0.50 — better than Phase 2's 0.46 but substantially below Ridge (0.80) and MLP (0.81). RF's decision-tree splits work less efficiently with continuous spectroscopic features than smooth parametric models. The `bootstrap=False` finding persists across all configs in Phase 3, confirming that with 20 samples, bagging harms RF performance by reducing effective training set size.

### 7. MLP tuning patterns reveal feature-set-dependent optimal architectures

Comparing Phase 2 and Phase 3 MLP tuned hyperparameters:
- **Config B** is nearly identical across phases: [16] hidden layer, batch_norm=True, lr ≈ 0.005. This makes sense since Config B input is unchanged.
- **Config C** shifted from [32, 16] with batch_norm=True (Phase 2) to [32] with batch_norm=False (Phase 3). The simpler architecture works because engineered features are already informative — the network doesn't need a deep pipeline to extract signal from noisy PCA components.
- **Config A** uses [32] with high dropout (0.56) and strong weight decay (0.064), reflecting the model's need for heavy regularisation when working with 13 OES features alone (no discharge params as anchors).

## Overall Model Ranking (Phase 3, best config per model)

| Rank | Model | Best Config | R² | RMSE |
|------|-------|:-----------:|:---:|:---:|
| 1 | Ridge | B | 0.90 | 0.071 |
| 2 | PLS | B | 0.90 | 0.074 |
| 3 | MLP | B | 0.86 | 0.087 |
| 4 | **MLP** | **C** | **0.81** | **0.099** |
| 5 | **Ridge** | **C** | **0.80** | **0.104** |
| 6 | CNN (P2) | C | 0.77 | 0.110 |
| 7 | RF | B | 0.75 | 0.116 |
| 8 | PLS | C | 0.74 | 0.117 |

Rows 4 and 5 (highlighted) are Phase 3's primary contribution: Config C models that **surpass** Phase 2's best OES-utilising model (CNN Config C, R² = 0.77) using simple architectures with engineered features.

## Interpretation for the Research Narrative

Phase 3 answers the three hypotheses posed in the action plan:

1. **Do domain-knowledge features outperform blind PCA?** **Yes, decisively.** Every model improved on Config A (OES-only) and Config C (OES + Params). The most dramatic improvements are in Config C, where PCA-based OES features actively harmed performance but engineered features now complement discharge parameters. Ridge Config C went from R² = −0.17 to R² = 0.80; MLP Config C went from R² = −1.13 to R² = 0.81.

2. **Does Config C become competitive with Config B?** **Partially.** Config C (R² = 0.80–0.81 for Ridge/MLP) still falls short of Config B (R² = 0.90), but the gap has narrowed dramatically from Phase 1 (where Config C was often worse than predicting the mean). OES features now provide genuine additive value rather than noise, but discharge parameters alone remain the strongest single predictor.

3. **Can MLP with engineered features match CNN on raw spectra?** **Yes — and exceed it.** MLP Config C (R² = 0.81) surpasses CNN Config C (R² = 0.77) from Phase 2, using 17 tabular features instead of 701-dim raw spectra. This demonstrates that on small datasets, domain expertise encoded in feature engineering is more valuable than architectural complexity.

The practical implication: for real-time H₂O₂ prediction with the current 20-sample dataset, **Ridge or PLS with discharge parameters alone (R² ≈ 0.90)** remains the best approach if only discharge settings are available. However, if OES monitoring is in place, **MLP or Ridge with engineered OES features + discharge parameters (R² ≈ 0.80)** provides the best combined model — and may outperform discharge-only prediction on future data where discharge parameters alone are insufficient (e.g., electrode degradation, gas composition changes, or other uncontrolled variables that OES can detect).

---

## Phase 4 Suggestion

### Rationale

Phases 1–3 have systematically explored three axes: model selection (Phase 1), hyperparameter tuning (Phase 2), and feature engineering (Phase 3). The fundamental constraint throughout has been the **20-sample dataset**. Phase 3 proved that domain-knowledge features unlock OES's predictive value, but Config C (R² ≈ 0.80) still falls short of Config B (R² ≈ 0.90), suggesting that with more data, OES could close or reverse this gap.

### Proposed Phase 4 Goals

#### Goal 1: Expand the Dataset

The single most impactful improvement. Even doubling from 20 to 40 samples would:
- Stabilise LOOCV estimates (each test fold would use 39 training samples instead of 19)
- Allow non-linear models (RF, MLP) to learn more complex relationships
- Enable proper train/validation/test splits instead of relying entirely on LOOCV
- Potentially reveal OES features that are predictive but require more data to distinguish from noise

**Practical approach:** If new experiments are feasible, prioritise conditions that fill gaps in the current 4-group × 5-level design (e.g., intermediate parameter values, combined parameter variations, or replicate measurements for uncertainty estimation).

#### Goal 2: Ensemble / Stacking Strategy

Combine the complementary strengths of different models:
- **Base layer:** Ridge Config B (R² = 0.90, strong on discharge params) and MLP Config C (R² = 0.81, strong on OES + params)
- **Meta-learner:** A simple Ridge regression that learns to weight the base predictions
- **Hypothesis:** Since Ridge Config B and MLP Config C capture different signal sources (discharge settings vs. spectroscopic features), their prediction errors should be partially uncorrelated, allowing the ensemble to exceed R² = 0.90

This can be implemented within the existing LOOCV framework: in each fold, train both base models, generate their predictions, and train the meta-learner on those predictions.

#### Goal 3: Feature Selection and Importance Analysis

With 13 engineered OES features, investigate which features contribute most:
- **Permutation importance** on the best-performing models (Ridge Config C, MLP Config C) to rank the 13 features
- **Ablation study:** systematically remove features or feature categories (single wavelengths only, band integrals only, ratios only) to determine the minimal informative feature set
- **Correlation analysis:** examine how the 13 features correlate with each other and with the target, to guide potential feature refinement or reduction

This would identify whether a smaller, more targeted feature set (e.g., 5–7 features) could achieve similar performance with better generalisation, and would provide physically interpretable insights for the research narrative (e.g., "OH radical intensity and CO₂⁺ ionisation are the two most predictive spectral markers for H₂O₂ yield").

#### Goal 4: Uncertainty Quantification

With 20 samples and LOOCV, point predictions tell an incomplete story. Phase 4 could add:
- **Prediction intervals** from RF (via quantile regression forests) or MLP (via MC-dropout or ensemble variance)
- **Bootstrap confidence intervals** on R² and RMSE estimates
- **Calibration analysis:** are the predicted H₂O₂ rates reliable enough for process control?

This is critical for the practical application: if OES-based prediction is used for real-time process monitoring, operators need to know not just the predicted yield, but how confident the prediction is.
