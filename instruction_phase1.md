# phase 1: plan

## Background

oes_ml_dataset_1nm.csv is a small scale dataset that obtained from CO2 discharge in water. The data is obtained from Gao's research.


## Overall objective
I'm conducting a research that use machine learning method to build a pipepline that takes in the instant OES data captured during the reaction (and possible parameters like discharge experience setting, voltage/frequency) and predicat the H2O2 yeild rate in real time.

## Phase 1 Goal

- Suggest the pipeline that input OES and other useful parameters, and output the H2O2 yield.
- Suggest machine learning models suitable for the prediction task. maybe XGBoost, CNN, MLP?
- Suggest should I compare building the ML model with (Only OES spectrum vs. OES + discharge parameters setting)
- Generate you response and append to the following section "#Phase1: action"



# Phase 1: action

## Dataset Summary

- **Samples:** 20 (4 experimental groups x 5 conditions)
- **OES features:** 701 (`I_200` to `I_900`, 1 nm resolution)
- **Discharge parameters:** 4 (`frequency_hz`, `pulse_width_ns`, `rise_time_ns`, `flow_rate_sccm`)
- **Target:** `h2o2_rate` (continuous, range 0.02 – 0.83)
- **Critical constraint:** n=20 with 701+ features — extreme high-dimension-low-sample-size (HDLSS) problem. This shapes every decision below.

---

## 1. Suggested Pipeline

```
Raw CSV
  |
  +-- [A] Discharge parameters (4 features)
  |        |
  |        +-- StandardScaler
  |
  +-- [B] OES spectra (701 features)
           |
           +-- Preprocessing
           |     - Normalize (SNV or Min-Max per sample)
           |     - Optional: Savitzky-Golay smoothing (denoise)
           |
           +-- Dimensionality reduction  <-- ESSENTIAL with n=20
                 - Option 1: PCA (retain 95% variance, likely ~5-10 components)
                 - Option 2: Select known emission peaks
                              (OH 309nm, N2 337nm, H-alpha 656nm, O 777nm, etc.)
                 - Option 3: PLS components (built into PLS model)
                 |
                 +-- Reduced OES features (~5-15 dimensions)
                           |
  +---------------------------+
  |
  v
Merge [A] + reduced [B]  (or [B] alone for OES-only experiment)
  |
  v
Model --> Predicted h2o2_rate
  |
  v
Evaluation (LOOCV / Repeated k-fold)
```

**Preprocessing notes:**
- StandardScaler on discharge params because they have different units/scales (Hz vs ns vs sccm)
- SNV or area-normalize OES to remove intensity variations unrelated to chemistry
- PCA or peak selection to reduce 701 dims to ~5-15; prevents overfitting and multicollinearity

---

## 2. Recommended Models

### Tier 1 — Best fit for n=20 spectral regression

| Model | Why suitable | Key hyperparameters |
|---|---|---|
| **PLS Regression** | Purpose-built for spectral data with few samples; handles multicollinearity natively; combines dim-reduction + regression | `n_components` |
| **Ridge Regression** | Strong L2 regularization prevents overfitting in HDLSS; simple, interpretable | `alpha` |
| **ElasticNet** | L1+L2 mix; can zero out irrelevant wavelengths (built-in feature selection) | `alpha`, `l1_ratio` |
| **SVR (RBF kernel)** | Kernel trick captures non-linearity without needing many samples | `C`, `epsilon`, `gamma` |

### Tier 2 — Usable with care

| Model | Caveats |
|---|---|
| **XGBoost / Random Forest** | Must heavily constrain: `max_depth<=3`, few estimators, high regularization. Apply **after** PCA. |
| **Simple MLP** | Only 1 hidden layer (8-16 neurons), strong dropout (0.3-0.5), weight decay, early stopping. Very prone to overfitting at n=20. |

### Tier 3 — NOT recommended for n=20

| Model | Why not |
|---|---|
| **CNN (1D-Conv)** | Needs hundreds-thousands of samples to learn spectral filters. Will memorize with n=20. |
| **Deep MLP (>2 layers)** | Parameter count dwarfs sample count. |

**Bottom line:** Start with **PLS Regression** as baseline (gold standard in chemometrics), compare against **Ridge, ElasticNet, SVR**, and **XGBoost (post-PCA)** as a non-linear benchmark. Skip CNN/deep networks unless dataset grows to 100+ samples.

---

## 3. OES-only vs. OES + Discharge Parameters: Yes, compare them

This is a standard and valuable ablation study. Run three experiments:

| Experiment | Inputs | What it tests |
|---|---|---|
| **A: OES only** | Reduced OES (PCA or selected peaks) | Can the spectrum alone predict H2O2 yield? |
| **B: OES + discharge params** | Reduced OES + 4 discharge settings | Do discharge settings add predictive power beyond what OES already captures? |
| **C: Discharge params only** | 4 discharge parameters | Naive baseline — how far can you get without any spectral data? |

**Why this matters:**
- If A approx B: OES already encodes discharge effects → supports "OES alone is sufficient for real-time monitoring" (stronger practical claim)
- If B >> A: Combined model needed, but both inputs are available in practice
- Experiment C provides the naive baseline that reviewers will expect
- This ablation directly answers a core research question and strengthens the paper

---

## 4. Evaluation Strategy

With n=20, evaluation must be handled carefully:

- **Leave-One-Out Cross-Validation (LOOCV):** train on 19, predict 1, repeat 20 times. Best option at this sample size.
- **Alternative:** Repeated 5-fold CV (10 repeats) for confidence intervals.
- **Metrics:** R-squared, RMSE, MAE on held-out samples.
- **Do NOT** use a single train/test split — variance would be too high.

---

## 5. Action Plan Summary

| Step | Action |
|---|---|
| 1 | Load and explore data; check for missing values, outliers in spectra |
| 2 | Preprocess OES (normalize, optional smoothing) and scale discharge params |
| 3 | Reduce OES dimensionality (PCA to ~5-10 components) |
| 4 | Train and evaluate via LOOCV: **PLS, Ridge, ElasticNet, SVR, XGBoost** |
| 5 | For each model, run three input experiments: **(A) OES-only, (B) OES+params, (C) params-only** |
| 6 | Compare results (R-squared, RMSE) across models and input configs |
| 7 | Identify best model + best input configuration |
| 8 | Analyze feature importance / PLS loadings to interpret which wavelengths matter most |