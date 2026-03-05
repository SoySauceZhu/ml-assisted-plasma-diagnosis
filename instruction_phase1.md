# phase 1: plan

## Background

oes_ml_dataset_1nm.csv is a small scale dataset that obtained from CO2 discharge in water. The data is obtained from Gao's research.


## Overall objective
I'm conducting a research that use machine learning method to build a pipepline that takes in the instant OES data captured during the reaction (and possible parameters like discharge experience setting, voltage/frequency) and predicat the H2O2 yeild rate in real time.

## Phase 1 Goal

- Suggest the pipeline that input OES and other useful parameters, and output the H2O2 yield.
- Since the dimension is extremely high, so you need consider PCA anaylisys firstly.
- Suggest machine learning models suitable for the prediction task. maybe XGBoost, CNN, MLP?
- Suggest should I compare building the ML model with (Only OES spectrum vs. OES + discharge parameters setting)
- Generate you response and append to the following section "#Phase1: action"



# Phase 1: action

## Dataset Summary

| Property | Value |
|---|---|
| Total samples | 20 |
| OES features (I_200 – I_900) | 701 wavelength intensities at 1 nm resolution |
| Discharge parameters | 4: `frequency_hz`, `pulse_width_ns`, `rise_time_ns`, `flow_rate_sccm` |
| Target variable | `h2o2_rate` (range: 0.02 – 0.83) |
| Experimental groups | 4 groups × 5 levels each (rise time, pulse width, frequency, flow rate) |

**Critical constraint:** 20 samples with 701+ features is an extreme case of the "curse of dimensionality." This fundamentally shapes every modelling decision below.

---

## 1. Proposed Pipeline

```
Raw OES spectrum (701 dims)
        │
        ▼
┌─────────────────────┐
│  Preprocessing      │
│  - Baseline corr.   │
│  - Normalisation    │
│  - Outlier check    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Dimensionality     │
│  Reduction (PCA)    │
│  701 → k components │
└────────┬────────────┘
         │
         ├──── Path A: PCA scores only
         │
         ├──── Path B: PCA scores + discharge params
         │          (frequency, pulse_width, rise_time, flow_rate)
         ▼
┌─────────────────────┐
│  ML Model           │
│  (see §3 below)     │
└────────┬────────────┘
         │
         ▼
   H₂O₂ yield rate
```

### Preprocessing steps
1. **Baseline correction** – subtract or divide by a baseline (e.g., the `0 ns` pulse-width condition which yields nearly zero H₂O₂, rate = 0.02) to remove background/instrument drift.
2. **Normalisation** – StandardScaler or Min-Max on OES intensities so large-magnitude peaks do not dominate PCA.
3. **Outlier / sanity check** – inspect for any spectra with anomalous spikes (the raw data already shows some very large values, e.g., ~11 000 around I_308, I_777; these are likely real emission lines such as OH 308 nm and O 777 nm, but should be verified).

---

## 2. PCA Analysis Plan

Given n = 20 and p = 701, PCA is not just recommended — it is **mandatory**. Without it, any model will overfit immediately.

### PCA strategy
- Fit PCA on the 701 OES columns.
- Select k components via **cumulative explained variance** (target ≥ 90–95%).
- With 20 samples, the maximum possible non-trivial components is 19 (= n − 1). In practice, the effective rank of the spectral matrix is likely much lower (often 3–8 components capture >95% of OES variance), because:
  - Emission lines are correlated (e.g., OH A-X band around 306–310 nm dominates).
  - Continuum background shifts uniformly across many wavelengths.
- Also produce a **scree plot** and **loading plots** to interpret which spectral regions drive each PC.

### Additional consideration: alternative to PCA
- **Feature selection by domain knowledge**: instead of blind PCA, manually select known diagnostic wavelengths (OH 308 nm, O 777 nm, Hα 656 nm, N₂ 337 nm, etc.) and use their intensities directly. This gives physically interpretable features and dramatically reduces dimensionality. Can be compared with PCA approach.

---

## 3. Recommended ML Models

Given the constraint of **n = 20 samples**, model choice must prioritise low complexity and resistance to overfitting. Here is a ranked recommendation:

| Rank | Model | Why suitable | Why risky | Recommendation |
|------|-------|-------------|-----------|----------------|
| 1 | **Ridge / Lasso Regression** | Built-in regularisation; works well with small n; interpretable coefficients; fast | Linear assumption may miss non-linear relationships | **Use as baseline** |
| 2 | **Partial Least Squares (PLS) Regression** | Specifically designed for high-dim, low-sample spectral data; simultaneously reduces dimensions and fits regression | Less flexible than tree-based | **Strongly recommended** — this is the standard in chemometrics/spectroscopy |
| 3 | **Support Vector Regression (SVR)** | Kernel trick handles non-linearity; effective in high-dim spaces; regularised | Hyperparameter tuning (C, ε, kernel) needs careful CV | **Recommended** |
| 4 | **XGBoost** | Handles non-linearity; feature importance built-in | Prone to overfit with n = 20; needs aggressive regularisation (max_depth ≤ 3, high reg_alpha/lambda, few estimators) | **Use with caution** — apply strong regularisation |
| 5 | **MLP (small)** | Can capture non-linear patterns | Very prone to overfit with 20 samples; many hyperparameters | **Include for comparison** — use small architecture (e.g., 1–2 hidden layers, ≤32 neurons), strong dropout (0.3–0.5), early stopping, and weight decay |
| 6 | **CNN (1D)** | Can learn local spectral features (peak shapes, band structures) automatically; weight sharing reduces parameters vs. MLP | Requires more data than classical methods; will likely overfit | **Include for comparison** — use shallow architecture (1–2 conv layers, small filters), global average pooling, dropout, and early stopping |

### Evaluation strategy
- **Leave-One-Out Cross-Validation (LOOCV)** — the only reliable CV strategy with n = 20.
- Metrics: **R²**, **RMSE**, **MAE**.
- Also report train vs. test error to detect overfitting.

---

## 4. Comparison Study: OES-only vs. OES + Discharge Parameters

**Yes, you should compare both approaches.** This is a valuable and publishable comparison. Here is the rationale:

### Experimental design: 3 input configurations

| Config | Input features | Rationale |
|--------|---------------|-----------|
| **A** | OES PCA scores only | Tests whether the spectrum alone is a sufficient predictor (i.e., does OES encode all the discharge physics?) |
| **B** | Discharge parameters only (4 features) | Baseline — can we predict H₂O₂ from settings alone without measuring OES? |
| **C** | OES PCA scores + discharge parameters | Tests whether discharge settings provide complementary information beyond what OES captures |

### Why this comparison matters
- If **A ≈ C >> B**: OES is the dominant predictor; discharge params add no value → supports the claim that real-time OES monitoring is sufficient for prediction.
- If **C >> A > B**: Both contribute → the combined model is best for deployment, but OES provides the core signal.
- If **B ≈ C >> A**: Discharge settings alone are sufficient → questions the need for OES in the pipeline.
- The result strengthens the research narrative regardless of outcome.

### Practical note
When combining OES PCA scores with discharge parameters (Config C), standardise all features to the same scale before feeding them into the model, since PCA scores and physical parameters (Hz, nm, sccm) have very different magnitudes.

---

## 5. Summary of Recommended Actions for Phase 2 (Implementation)

1. Load and preprocess the OES data (baseline correction, normalisation).
2. Perform PCA on the 701 OES features; determine optimal k via scree plot / cumulative variance.
3. Build and evaluate the following models under LOOCV:
   - **PLS Regression** (primary model)
   - **Ridge Regression** (linear baseline)
   - **SVR with RBF kernel** (non-linear option)
   - **XGBoost** (with aggressive regularisation, as a comparison)
   - **MLP** (small architecture: 1–2 hidden layers ≤32 neurons, dropout 0.3–0.5, weight decay, early stopping)
   - **1D-CNN** (shallow: 1–2 conv layers, global average pooling, dropout, early stopping; takes raw OES spectrum as input — PCA not applied for CNN, as conv layers learn their own feature extraction)
4. For each model, compare three input configs: (A) OES-only, (B) Params-only, (C) OES + Params.
5. Report R², RMSE, MAE and produce predicted-vs-actual scatter plots.
6. Produce PCA loading plots to identify which wavelengths matter most (physical interpretability).

---

# Phase 1: Observation

## PCA Results

PCA on the 701 OES features required **11 components** to reach the 95% cumulative variance threshold. The first 5 PCs capture ~74% of variance, with PC1 alone explaining 43%. This confirms that the spectral data has high redundancy — the effective dimensionality is far below 701, but not as low as initially hoped (3–8 components); 11 components are needed for 95%.

## LOOCV Results Summary

| Model | Config A (OES only) | Config B (Params only) | Config C (OES + Params) |
|-------|:---:|:---:|:---:|
| **Ridge** | R²= −0.31 | **R²= 0.90** | R²= −0.17 |
| **PLS** | R²= −0.60 | **R²= 0.90** | R²= 0.63 |
| **SVR** | R²= 0.05 | R²= 0.62 | R²= 0.09 |
| **XGBoost** | R²= −0.11 | R²= −0.11 | R²= −0.11 |
| **RF** | R²= 0.04 | R²= 0.38 | R²= 0.24 |
| **MLP** | R²= −0.85 | R²= 0.57 | R²= −1.13 |
| **CNN** | R²= 0.30 | N/A | **R²= 0.69** |

## Key Observations

### 1. Config B (discharge params only) dominates for most models

The clearest and most surprising finding: **discharge parameters alone (Config B) outperform OES-based inputs** for 5 out of 6 applicable models. Ridge and PLS both achieve R² ≈ 0.90 with just 4 discharge parameters, while the same models score negative R² on OES-only input (Config A). This follows the pattern **B >> A**, which questions the standalone predictive value of OES in this small dataset.

### 2. OES-only input (Config A) performs poorly across the board

No model achieves R² > 0.31 on Config A. Most models produce negative R² (worse than predicting the mean), indicating that 11 PCA components from 20 samples cause severe overfitting. The curse of dimensionality is clearly evident — even after PCA, the OES representation has too many features relative to the sample size.

### 3. CNN is the exception — OES adds value when learned end-to-end

CNN Config C (OES + Params) achieves the **best OES-utilising performance** at R² = 0.69, and even CNN Config A (OES only) reaches R² = 0.30. This suggests that the CNN's convolutional layers learn a more effective spectral representation than PCA. The CNN bypasses PCA and learns its own feature extraction, which appears better suited for this data.

### 4. PLS Config C shows OES can complement discharge parameters

PLS with combined input (Config C, R² = 0.63) significantly outperforms PLS Config A (R² = −0.60), though it falls short of PLS Config B (R² = 0.90). This suggests PLS can extract some useful OES information when combined with discharge parameters, but the OES signal is not strong enough to improve upon discharge-only prediction for linear models.

### 5. Tree-based models struggle with this dataset

XGBoost produces identical R² = −0.11 across all three configs, suggesting it collapses to a trivial prediction regardless of input. RF performs modestly (best R² = 0.38 on Config B) but lags behind linear models. With only 20 samples, tree-based ensemble methods cannot build meaningful splits.

### 6. Neural networks overfit severely on OES input

MLP achieves R² = −0.85 (Config A) and R² = −1.13 (Config C), the worst results in the entire study. Even with dropout 0.4, weight decay, and early stopping, 20 samples are insufficient for MLP to generalise when OES features are present. Only Config B (4 features) yields a reasonable MLP result (R² = 0.57).

## Interpretation for the Research Narrative

The Phase 1 results suggest the relationship **B ≈ C >> A** for most models — discharge parameters are the dominant predictor and OES adds limited value in this 20-sample regime. However, this does **not** necessarily mean OES is uninformative. The more likely explanation is:

1. **Sample size is too small** for high-dimensional OES to show its value. PCA retains 11 components for 20 samples (p/n ratio ~0.55), leaving little room for generalisation.
2. **The 4 discharge parameters are highly structured** (each experimental group varies exactly one parameter), making prediction straightforward for regularised linear models.
3. **CNN's relative success with OES** hints that with more data or better spectral feature extraction, OES could become a useful predictor.

## Recommendations for Phase 2

1. **Prioritise expanding the dataset** — the current 20 samples fundamentally limit what can be learned from high-dimensional OES data.
2. **Investigate domain-specific OES feature engineering** — instead of blind PCA, select known diagnostic wavelengths (OH 308 nm, O 777 nm, H-alpha 656 nm) to create a small, physically meaningful feature set that may perform better with limited data.
3. **Focus on Ridge/PLS as baselines** — they are the best-performing models and most stable; use them as the benchmark for any improved approach.
4. **Explore CNN architecture further** — it is the only model that successfully extracts useful information from raw OES spectra; consider if a deeper or different architecture could improve further.
5. **Consider whether real-time OES monitoring adds value beyond discharge settings** — if the goal is real-time H₂O₂ prediction, and discharge parameters alone achieve R² = 0.90, the practical question becomes whether OES monitoring is justified by marginal improvement or whether it offers value in detecting anomalies/drift that discharge settings alone cannot capture.

