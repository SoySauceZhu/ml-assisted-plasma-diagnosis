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