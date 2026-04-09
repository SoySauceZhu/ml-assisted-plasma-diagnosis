# Final Report — Phase 1: Baseline ML Pipeline

## Overall Logical Progression (总体行文逻辑)

This report follows a **problem-driven** structure: establish context → define the challenge → present methodology → report findings → lay groundwork for subsequent phases.

1. **Context & Motivation** — Why CO2 plasma discharge for H2O2 production matters, and why real-time OES-based prediction is valuable.
2. **Problem Definition** — The extreme curse of dimensionality: 20 samples vs. 701 OES wavelengths + 4 discharge parameters.
3. **Proposed Pipeline** — Preprocessing → Dimensionality Reduction (PCA) → Model Selection → Evaluation Strategy (LOOCV).
4. **Experimental Design** — Three input configurations (OES-only, Params-only, OES+Params) to disentangle the predictive value of each input type.
5. **Results & Analysis** — Quantitative comparison across 7 models × 3 configs, with physical interpretation.
6. **Conclusion & Transition** — Key findings from Phase 1 and implications for Phase 2 (hyperparameter tuning) and Phase 3 (domain-knowledge feature engineering).

---

## Report Sections / Part Structure

| Section | Title | Purpose |
|---------|-------|---------|
| 1 | Introduction | Context, motivation, objectives |
| 2 | Background | Plasma chemistry, OES fundamentals, dataset description |
| 3 | Methodology | Pipeline architecture, PCA, model selection, LOOCV evaluation |
| 4 | Results & Discussion | Model comparison, key observations, physical interpretation |
| 5 | Conclusion | Summary of findings and transition to next phases |

---

## Content of Each Section

### Section 1: Introduction

**Content:**
- Background on CO2 greenhouse gas conversion and plasma chemistry
- The promise of H2O2 as a green oxidising agent from CO2 discharge
- **Research objective**: Build an ML pipeline that takes OES spectra + discharge parameters and predicts H2O2 yield rate in real time
- **Phase 1 scope**: Establish a baseline; evaluate multiple models and input configurations; identify the dominant predictor

**Key message**: Real-time OES monitoring can potentially replace slow offline titration, enabling closed-loop process control in plasma-based green chemistry.

---

### Section 2: Background

**Content:**

#### 2.1 CO2 Plasma Discharge and H2O2 Synthesis
- Nanosecond pulsed CO2 bubble discharge converts CO2 to useful chemicals (H2, CO, H2O2)
- H2O2 is a valuable green oxidising agent; direct synthesis via plasma is an emerging approach
- Experimental setup at XJTU (Gao's research group), dataset openly available

#### 2.2 Optical Emission Spectroscopy (OES)
- OES captures light emission from excited species in plasma (atoms, molecules, radicals)
- Provides non-intrusive, real-time window into plasma chemistry
- Key spectral regions for this system:
  - OH band (306–312 nm) — relates to H2O2 formation
  - O atomic line (777 nm)
  - H-beta (486 nm), H-alpha (656 nm)
  - N2 band (337 nm), CO2 band (398–412 nm)
- 701 wavelength points (200–900 nm, 1 nm resolution) per measurement

#### 2.3 Dataset Description
- **Source**: Gao et al., XJTU (open dataset)
- **Samples**: 20 discharge experiments
- **Features**:
  - 701 OES intensities (I_200 to I_900)
  - 4 discharge parameters: frequency (Hz), pulse_width (ns), rise_time (ns), flow_rate (sccm)
- **Target**: H2O2 rate (range: 0.02–0.83)
- **Experimental design**: 4 groups × 5 levels each (each group varies one parameter)
- **Critical constraint**: n = 20, p = 701+4 → severe curse of dimensionality

---

### Section 3: Methodology

**Content:**

#### 3.1 Data Preprocessing
- **Baseline correction**: Subtract mean spectrum of pulse_width=0 samples (near-zero H2O2 reference, rate=0.02) to remove background/instrument drift
- **Standardisation**: StandardScaler applied to OES and discharge features independently before PCA
- **Outlier check**: Inspect for anomalous spectral spikes (verified as real emission lines: OH 308 nm, O 777 nm)

#### 3.2 Dimensionality Reduction — PCA
- PCA fitted on 701 OES wavelengths (scaled)
- Cumulative explained variance threshold: 95%
- Result: **k = 11 principal components** capture 95% variance
  - PC1 alone explains 43%
  - First 5 PCs explain ~74%
- Diagnostic plots: scree plot, cumulative variance, loading plots (annotated with known spectral lines), 2D score plot

#### 3.3 Experimental Design: Three Input Configurations

| Config | Input | Research Question |
|--------|-------|-------------------|
| **A** | OES → PCA (11 components) | Is the OES spectrum alone sufficient? |
| **B** | Discharge params only (4 features) | Can discharge settings predict yield without OES? |
| **C** | OES → PCA (11) + Discharge params | Do OES and params provide complementary information? |

Note: CNN uses raw OES (701 dims) for Config A and C (not PCA), as conv layers perform their own feature extraction.

#### 3.4 Model Selection
Seven models evaluated, chosen to span linear → non-linear → deep learning:

| Model | Rationale |
|-------|-----------|
| **Ridge Regression** | Linear baseline with L2 regularisation; robust for small n |
| **Partial Least Squares (PLS)** | Standard chemometrics method; simultaneous dimensionality reduction + regression |
| **Support Vector Regression (SVR)** | RBF kernel for non-linearity; regularised by margin |
| **XGBoost** | Gradient boosting; handles non-linearity but prone to overfit on n=20 |
| **Random Forest** | Ensemble of decision trees; fixed conservative hyperparameters |
| **MLP (PyTorch)** | Fully-connected network; [32, 16] hidden sizes, dropout 0.4, early stopping |
| **CNN 1D (PyTorch)** | 1D conv on raw OES; [16, 32] channels, kernel=7, global avg pooling |

#### 3.5 Evaluation Strategy
- **Leave-One-Out Cross-Validation (LOOCV)**: Only viable CV strategy with n=20 (20 folds, each with 19 train / 1 test)
- **Metrics**: R², RMSE, MAE
- Train-vs-test error comparison to detect overfitting

---

### Section 4: Results & Discussion

**Content:**

#### 4.1 PCA Analysis Results
- k = 11 components (95% variance threshold)
- PC1 dominated by continuum/baseline variation
- PC2–PC4 capture key emission bands (OH, atomic lines)
- Score plot shows clustering by experimental groups; partial separation by H2O2 rate
- **Interpretation challenge**: Even 11 components for 20 samples (p/n ≈ 0.55) leaves limited room for generalisation

#### 4.2 Model Comparison (LOOCV)

**Summary Table (R² values):**

| Model | Config A (OES) | Config B (Params) | Config C (OES+Params) |
|-------|:---:|:---:|:---:|
| Ridge | −0.31 | **0.90** | −0.17 |
| PLS | −0.60 | **0.90** | 0.63 |
| SVR | 0.05 | 0.62 | 0.09 |
| XGBoost | −0.11 | −0.11 | −0.11 |
| Random Forest | 0.04 | 0.38 | 0.24 |
| MLP | −0.85 | 0.57 | −1.13 |
| CNN | 0.30 | N/A | **0.69** |

#### 4.3 Key Observations

**Observation 1: Discharge parameters dominate (Config B >> Config A for most models)**
- Ridge and PLS achieve R² ≈ 0.90 with just 4 parameters
- This suggests the experimental groups are well-separated by discharge settings alone
- The curse of dimensionality makes OES (even after PCA) unreliable with only 20 samples

**Observation 2: CNN is the exception — end-to-end OES learning outperforms PCA**
- CNN Config C (R² = 0.69) is the best OES-utilising model
- CNN Config A (R² = 0.30) still beats most other models on OES-only input
- Convolutional layers learn more effective spectral representations than linear PCA

**Observation 3: Tree-based models collapse on this dataset**
- XGBoost: identical R² = −0.11 across all configs (trivial prediction)
- RF: modest at best (R² = 0.38, Config B)
- 20 samples insufficient for meaningful tree splits

**Observation 4: Neural networks severely overfit on high-dimensional OES**
- MLP: worst performer on Config A and C (R² = −0.85 and −1.13)
- Only MLP Config B (4 features) yields reasonable results (R² = 0.57)

**Observation 5: PLS Config C shows OES provides complementary information**
- PLS Config C (R² = 0.63) significantly outperforms PLS Config A (R² = −0.60)
- But Config C << Config B → OES contribution is real but insufficient to close the gap

#### 4.4 Physical Interpretation
- The strong Config B performance reflects the experimental design: each group varies exactly one parameter, creating easily separable classes
- OES failure is likely due to: (1) insufficient sample size for the high dimensionality, and (2) PCA being a blind dimensionality reduction that ignores plasma chemistry
- CNN's relative success suggests the raw spectrum contains useful information — it just needs the right architecture to extract it

---

### Section 5: Conclusion

**Content:**

#### 5.1 Summary of Phase 1 Findings
1. **Discharge parameters are the dominant predictor** — linear models with just 4 parameters achieve R² ≈ 0.90
2. **OES alone is insufficient with PCA** — negative R² for most models on Config A; the curse of dimensionality is severe
3. **CNN shows promise** — the only model family where raw OES contributes meaningfully (R² = 0.69, Config C)
4. **Tree-based methods and MLP are unsuitable** for this dataset size without substantial modifications

#### 5.2 Implications for Subsequent Phases
- **Phase 2 (Hyperparameter Tuning)**: Optuna-based tuning of MLP and CNN could improve Config C performance; non-linear models may close the gap with Config B
- **Phase 3 (Domain-Knowledge Feature Engineering)**: Instead of PCA, manually select known diagnostic wavelengths (OH 309, O 777, H-beta 486) to create a small, physically meaningful feature set — this directly addresses the core weakness identified in Phase 1
- **Phase 4 (Interpretability)**: Identify which OES features are most informative; simplify to a minimal feature set that achieves competitive R²

#### 5.3 Broader Implication
- This phase establishes that **real-time H2O2 prediction is feasible** — discharge parameters alone already reach R² = 0.90
- The remaining research question is whether domain-knowledge OES features can improve upon this baseline while maintaining physical interpretability

---

## Appendix Contents (Phase 1 Report)

- A: Full LOOCV results table (R², RMSE, MAE for all model × config combinations)
- B: PCA diagnostic plots (scree, cumulative variance, loadings, scores)
- C: Predicted vs. actual scatter plots for all model × config combinations
- D: Summary heatmaps and bar charts
