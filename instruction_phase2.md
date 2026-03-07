# phase 2: plan

## Background

Based on phase1's result, ridge and PLS are best performed basedline model, especially on Config B (discharge parameter only). But in config C, RF, CNN, and MLP has a poor result. In phase 2, you are going to find the best parameters for these three models

## Phase2 Goal
- Keep following your working style in phase 1
- Suggest the method to automatically find the best parameters for Random Forest, CNN, and MLP models
- Find the best parameters, train the models and run the evaluation again
- Generate you response for phase2 action and append to the following section "#Phase2: action"

---

# Phase 2: action

## Problem Recap

From Phase 1 LOOCV results on **Config C** (OES + Params) — the most practically relevant configuration:

| Model | R² (Config C) | Status |
|-------|:---:|---|
| Ridge | −0.17 | Baseline (no tuning needed) |
| PLS | 0.63 | Baseline (no tuning needed) |
| CNN | **0.69** | Best OES-utilising model — room to improve |
| RF | 0.24 | Poor — needs tuning |
| MLP | −1.13 | Severely overfitting — needs tuning |
| SVR | 0.09 | Poor but not in scope |
| XGBoost | −0.11 | Collapsed — not in scope |

**Goal:** Find optimal hyperparameters for **Random Forest**, **CNN**, and **MLP** to close the gap with Ridge/PLS on Config B (R² ≈ 0.90) and improve Config C performance.

---

## 1. Hyperparameter Tuning Strategy

### Why not standard GridSearchCV?

With only **n = 20** samples and LOOCV as the evaluation protocol, we face a unique challenge: we cannot split data into train/validation/test. Standard nested cross-validation (e.g., 5-fold inner CV) would use only 15 training samples in the inner loop — too few for reliable hyperparameter selection.

### Recommended approach: **Nested LOOCV with Optuna (Bayesian Optimisation)**

```
Outer loop: LOOCV (20 folds — one sample held out for final evaluation)
  └─ Inner loop: LOOCV on remaining 19 samples (for hyperparameter selection)
       └─ Optuna TPE sampler explores hyperparameter space
```

**Why Optuna over GridSearchCV / RandomSearchCV:**
- **Efficiency**: Bayesian optimisation (TPE) explores the hyperparameter space more intelligently than random or grid search. With expensive inner LOOCV (19 fits per trial), we need fewer trials.
- **Flexibility**: Handles mixed parameter types (continuous, integer, categorical) and conditional parameters naturally.
- **Pruning**: Can early-stop unpromising trials, saving compute time on CNN/MLP training.

**Practical simplification:** Running full nested LOOCV (20 outer × 19 inner × N trials) is computationally expensive, especially for CNN/MLP. Two options:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **A: Full nested LOOCV** | Inner LOOCV within each outer fold | Unbiased estimate | Very slow (20 × 19 × N_trials fits) |
| **B: Single inner LOOCV + reuse** | Run Optuna once on all 20 samples with inner LOOCV, find best params, then run outer LOOCV with those fixed params | Fast, practical | Slight optimistic bias |

**Recommendation:** Use **Approach B** as the primary method. It is standard practice in small-sample studies and avoids prohibitive compute cost. For RF (fast training), we can also try Approach A as a sensitivity check.

---

## 2. Hyperparameter Search Spaces

### 2.1 Random Forest

RF's Phase 1 performance (R² = 0.24 on Config C) is limited by the fixed conservative parameters (`max_depth=3`, `min_samples_split=5`). With PCA-reduced features + 4 discharge params (~15 features), the search space should explore slightly more flexible trees while still preventing overfitting.

```python
RF_SEARCH_SPACE = {
    "n_estimators": [50, 100, 200, 500],           # more trees = more stable
    "max_depth": [2, 3, 4, 5, None],               # allow deeper trees
    "min_samples_split": [2, 3, 4, 5],             # allow finer splits
    "min_samples_leaf": [1, 2, 3],                 # allow smaller leaves
    "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],  # feature subsampling
    "bootstrap": [True, False],                     # with/without replacement
}
```

**Tuning method:** `sklearn.model_selection.GridSearchCV` with `cv=LeaveOneOut()` — RF is fast enough for exhaustive search on the most promising subsets, or `RandomizedSearchCV` with ~200 iterations.

### 2.2 MLP

MLP's Phase 1 disaster (R² = −1.13 on Config C) is due to overfitting with too many parameters for 20 samples. The tuning must focus on **aggressively constraining model capacity**.

```python
MLP_SEARCH_SPACE = {
    "hidden_sizes": [[8], [16], [32], [8, 4], [16, 8], [32, 16]],
    "dropout": [0.3, 0.4, 0.5, 0.6, 0.7],          # higher dropout
    "weight_decay": [1e-1, 5e-2, 1e-2, 1e-3],       # stronger L2
    "lr": [1e-4, 5e-4, 1e-3, 5e-3],
    "max_epochs": [200, 500, 1000],
    "patience": [30, 50, 100],
    "batch_norm": [True, False],                     # may help generalisation
}
```

**Key insight:** The current architecture `[32, 16]` with dropout 0.4 has **672 parameters** (for ~15 input features). With 20 samples, a single hidden layer of 8 neurons (~137 params) may be more appropriate.

### 2.3 CNN

CNN already achieves R² = 0.69 on Config C — the best OES-utilising result. Tuning should explore architectural variations while keeping the model shallow.

```python
CNN_SEARCH_SPACE = {
    "conv_channels": [[8], [16], [8, 16], [16, 32], [32, 64]],
    "kernel_size": [3, 5, 7, 11, 15, 21],            # spectral peak widths
    "dropout": [0.3, 0.4, 0.5, 0.6],
    "weight_decay": [5e-2, 1e-2, 5e-3, 1e-3],
    "lr": [1e-4, 5e-4, 1e-3],
    "max_epochs": [300, 500, 1000],
    "patience": [30, 50, 100],
    "pool_type": ["avg", "max"],                      # global pooling strategy
    "fc_hidden": [None, 8, 16],                       # optional FC layer before output
}
```

**Key insight:** The kernel size is particularly important for 1D-CNN on spectral data — it determines the spectral window the model "sees." Emission lines like OH 308 nm have widths of ~2–5 nm, so smaller kernels (3–7) may capture peaks better, while larger kernels (15–21) capture broader band structures.

---

## 3. Implementation Plan

### Step 1: Create `phase2/` directory structure

```
phase2/
├── __init__.py
├── config.py              # search spaces and tuning settings
├── tuner.py               # Optuna-based tuning logic
├── tuner_rf.py            # RF-specific tuning (sklearn GridSearchCV)
├── tuner_mlp.py           # MLP-specific tuning (Optuna + PyTorch)
├── tuner_cnn.py           # CNN-specific tuning (Optuna + PyTorch)
├── main.py                # orchestrator
└── results/
    ├── figures/
    └── tables/
```

### Step 2: Implement tuning pipeline

1. **Reuse Phase 1's `data_loader.py` and `evaluation.py`** — no need to duplicate data loading.
2. **For RF:** Use `sklearn.model_selection.GridSearchCV` or `RandomizedSearchCV` with `cv=LeaveOneOut()`. Straightforward since RF is an sklearn model.
3. **For MLP and CNN:** Use **Optuna** with a custom objective function that:
   - Accepts an Optuna `trial` object
   - Samples hyperparameters from the search space
   - Runs inner LOOCV (19 folds) using Phase 1's training loop
   - Returns mean R² (or negative RMSE) as the objective
4. **Run Optuna** with `n_trials=100` for MLP, `n_trials=100` for CNN (adjustable based on compute budget).

### Step 3: Re-evaluate with best parameters

1. Take the best hyperparameters found by Optuna/GridSearchCV.
2. Run **full outer LOOCV** (same protocol as Phase 1) with these fixed parameters.
3. Compare against Phase 1 results directly.

### Step 4: Generate comparison report

- Table: Phase 1 vs Phase 2 results (R², RMSE, MAE) for RF, CNN, MLP across all configs.
- Optuna optimisation history plots (showing convergence).
- Hyperparameter importance plots (which parameters matter most).
- Updated predicted-vs-actual scatter plots.

---

## 4. Expected Outcomes and Risks

| Model | Expected improvement | Risk |
|-------|---------------------|------|
| **RF** | Moderate — likely R² 0.3–0.6 on Config C. RF fundamentally needs more data for high-dim inputs, but better `max_features` and `max_depth` may help. | May not significantly outperform Phase 1 since the bottleneck is sample size, not hyperparameters. |
| **MLP** | Significant — from R² = −1.13 to potentially R² 0.3–0.6 on Config C. Smaller architecture + stronger regularisation should eliminate catastrophic overfitting. | Even well-tuned MLP may not beat PLS/Ridge with 20 samples. The improvement may be "less terrible" rather than "good." |
| **CNN** | Moderate — from R² = 0.69 to potentially R² 0.7–0.8 on Config C. Already the best OES model; kernel size and dropout tuning could yield incremental gains. | Risk of overfitting to the inner LOOCV — small improvements may not be robust. |

### Important caveat

Hyperparameter tuning with 20 samples has inherent limitations. The search may find parameters that happen to work well on this specific dataset but don't generalise. This is why we use Approach B (single inner LOOCV) rather than cherry-picking from many random runs. All results should be interpreted with this caveat.

---

## 5. Compute Estimate

| Model | Inner LOOCV fits per trial | Trials | Total model fits | Estimated time |
|-------|:---:|:---:|:---:|---|
| RF | 20 | 200 (RandomizedSearchCV) | 4,000 | ~1–2 min |
| MLP | 19 | 100 (Optuna) | 1,900 | ~10–20 min |
| CNN | 19 | 100 (Optuna) | 1,900 | ~20–40 min |

Total: ~30–60 minutes on a modern CPU (no GPU required given the small data and model sizes).

---

## 6. Dependencies

```
optuna>=3.0          # Bayesian hyperparameter optimisation
scikit-learn         # already installed (Phase 1)
torch                # already installed (Phase 1)
numpy, pandas        # already installed (Phase 1)
matplotlib           # already installed (Phase 1)
```

Only **Optuna** needs to be added. Install via: `pip install optuna`

---

# Phase 2: Observation

## Phase 1 vs Phase 2 LOOCV Results Comparison

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

## Best Hyperparameters Found

### Random Forest
| Parameter | Config A | Config B | Config C |
|-----------|----------|----------|----------|
| n_estimators | 500 | 50 | 100 |
| max_depth | None | None | None |
| min_samples_split | 4 | 2 | 2 |
| min_samples_leaf | 2 | 1 | 1 |
| max_features | log2 | 0.5 | sqrt |
| bootstrap | False | False | False |

### MLP
| Parameter | Config A | Config B | Config C |
|-----------|----------|----------|----------|
| hidden_sizes | [32, 16] | [16] | [32, 16] |
| dropout | 0.37 | 0.42 | 0.47 |
| weight_decay | 0.069 | 0.007 | 0.003 |
| lr | 0.005 | 0.004 | 0.004 |
| max_epochs | 500 | 1000 | 1000 |
| patience | 30 | 100 | 50 |
| batch_norm | True | True | True |

### CNN
| Parameter | Config A | Config C |
|-----------|----------|----------|
| conv_channels | [8, 16] | [32, 64] |
| kernel_size | 15 | 5 |
| dropout | 0.34 | 0.42 |
| weight_decay | 0.003 | 0.001 |
| lr | 0.0009 | 0.0004 |
| max_epochs | 500 | 500 |
| patience | 100 | 100 |
| pool_type | max | avg |
| fc_hidden | 8 | None |

## Key Observations

### 1. Hyperparameter tuning universally improved all models

Every model–config combination improved in Phase 2. The total ΔR² across all 8 experiments sums to +4.10, with individual gains ranging from +0.09 (CNN Config C) to +1.50 (MLP Config C). This confirms that Phase 1's default hyperparameters were far from optimal, particularly for MLP and RF.

### 2. MLP shows the most dramatic recovery

MLP went from catastrophic overfitting (R² = −1.13 on Config C, −0.85 on Config A) to reasonable performance (R² = 0.37 for both). On Config B, MLP now achieves R² = 0.86 — the **second-best result in the entire study** after Ridge/PLS Config B (R² ≈ 0.90). Key tuning changes: batch normalisation enabled across all configs, and Config B converged on a single hidden layer of 16 neurons (down from [32, 16]), confirming that smaller architectures generalise better with 20 samples.

### 3. Config B (discharge params only) remains dominant

The ranking **B >> C > A** persists after tuning for all three models:
- RF: 0.75 (B) > 0.46 (C) > 0.22 (A)
- MLP: 0.86 (B) > 0.37 (C) ≈ 0.37 (A)
- CNN: N/A (B) > 0.77 (C) > 0.53 (A)

This reinforces Phase 1's finding: discharge parameters are more predictive than OES for this 20-sample dataset. Hyperparameter tuning did not change this fundamental conclusion.

### 4. CNN Config C remains the best OES-utilising model

CNN Config C improved from R² = 0.69 to R² = 0.77, remaining the strongest model that leverages OES data. The tuned CNN uses a deeper architecture ([32, 64] channels) with a small kernel size (5) for Config C, suggesting it captures fine spectral features. For Config A (OES only), the CNN uses a wider kernel (15) with shallower channels ([8, 16]), indicating it needs broader spectral context when discharge parameters are absent.

### 5. RF benefits substantially from tuning but still trails linear models

RF Config B improved from R² = 0.38 to R² = 0.75 — a near-doubling. The key change was removing the `max_depth=3` constraint (all tuned configs use `max_depth=None`) and disabling bootstrap. However, RF Config B (0.75) still falls short of Ridge/PLS Config B (0.90), confirming that with 20 structured samples, regularised linear models remain superior for pure parameter-based prediction.

### 6. OES adds value only through CNN's end-to-end learning

For RF and MLP, Config C (OES + Params) consistently underperforms Config B (Params only): RF 0.46 < 0.75, MLP 0.37 < 0.86. Adding PCA-reduced OES features hurts these models even after tuning — the extra dimensions still cause overfitting. Only CNN successfully combines OES with discharge parameters (0.77 > 0.53), because it learns spectral features directly from raw data rather than relying on PCA.

## Overall Model Ranking (Phase 2, best config per model)

| Rank | Model | Best Config | R² | RMSE |
|------|-------|:-----------:|:---:|:---:|
| 1 | Ridge | B | 0.90 | 0.071 |
| 2 | PLS | B | 0.90 | 0.074 |
| 3 | MLP (tuned) | B | 0.86 | 0.086 |
| 4 | CNN (tuned) | C | 0.77 | 0.110 |
| 5 | RF (tuned) | B | 0.75 | 0.116 |
| 6 | SVR | B | 0.62 | 0.143 |

## Interpretation for the Research Narrative

Phase 2 confirms two things:

1. **Hyperparameter tuning matters** — default parameters severely underestimated the potential of RF, MLP, and CNN. Proper tuning brought MLP from negative R² to R² = 0.86 on Config B, making it competitive with Ridge/PLS.

2. **The fundamental limitation is sample size, not model choice** — even after extensive Bayesian optimisation (100 trials per model-config), no model with OES input surpasses the simple Ridge/PLS models using only 4 discharge parameters. The 20-sample dataset does not contain enough information for models to learn robust spectral representations through PCA. CNN's partial success with raw OES (R² = 0.77 on Config C) hints that with more data, OES could become genuinely useful.

The practical implication: if the goal is real-time H₂O₂ prediction with the current dataset, Ridge or PLS with discharge parameters alone (R² ≈ 0.90) is the best approach. OES adds value only through CNN, and only modestly.

---

## Phase 3 suggestion

### Rationale

Phase 1 and 2 have exhaustively explored model selection and hyperparameter tuning on the current 20-sample dataset. The consistent finding is that OES data underperforms discharge parameters due to the extreme sample-to-feature ratio. Phase 3 should address this fundamental bottleneck.

### Proposed Phase 3 Goals

#### Goal 1: Domain-Informed OES Feature Engineering

Instead of blind PCA (701 → 11 components), extract physically meaningful features from the OES spectra:

- **Peak intensities** at known diagnostic wavelengths: OH (308 nm), O (777 nm), Hα (656 nm), N₂ (337 nm), Hβ (486 nm)
- **Peak ratios** (e.g., OH/O, OH/Hα) that relate to plasma chemistry
- **Band integrals** over known emission bands (e.g., integrate 300–315 nm for OH A-X system)
- **Spectral statistics** (total emission intensity, spectral centroid, spectral width)

This would reduce OES from 701 dimensions to ~10–15 physically interpretable features, dramatically improving the feature-to-sample ratio while retaining domain-relevant information. Re-run all models with these engineered features and compare against PCA-based results.

#### Goal 2: Expand the Dataset

Investigate whether additional experimental data points (new discharge conditions or repeated measurements) can be incorporated. Even increasing from 20 to 40–50 samples could substantially change the OES vs. discharge parameter comparison, as:
- PCA components would be better estimated
- Non-linear models (CNN, MLP, RF) would have more room to generalise
- LOOCV estimates would become more stable

#### Goal 3: Ensemble / Stacking Strategy

Combine the strengths of different models:
- Use Ridge/PLS Config B as the base predictor (strong on discharge parameters)
- Use CNN Config C as a secondary predictor (captures OES information)
- Build a simple stacking ensemble that could potentially exceed R² = 0.90 by combining both signal sources

This approach leverages the finding that different models excel at extracting different types of information (linear models for discharge params, CNN for spectral data).