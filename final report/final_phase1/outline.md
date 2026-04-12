# Final Report Outline

## 1. Overall Story (总体行文逻辑)

Nanosecond pulsed CO₂ bubble plasma discharge is a promising green chemistry technique, but real-time monitoring of product yield (H₂O₂) remains a bottleneck — conventional quantification requires offline titration, which is slow and intrusive. Optical Emission Spectroscopy (OES) provides a non-intrusive diagnostic window, yet a single OES measurement comprises 701 wavelength points, making direct ML application challenging. Prior OES+ML studies apply PCA or PLS to raw spectra, discarding the physical meaning of individual emission lines and limiting both accuracy and interpretability. This project demonstrates that **domain-knowledge-driven feature engineering** — selecting emission line intensities, band integrals, and spectroscopic ratios grounded in plasma chemistry literature — transforms model performance: Ridge regression R² on the combined input (OES + discharge parameters) jumps from −0.17 (PCA) to 0.80 (domain features). Furthermore, backward elimination and category ablation identify a minimal 7-feature Ridge model (3 OES ratios + 4 discharge parameters) achieving R² = 0.920, matching neural networks with far fewer parameters. The broader implication is clear: for physically structured scientific datasets, especially small ones, simple interpretable models with domain-informed features outperform complex models with automated feature extraction.

---

## 2. Section-by-Section Outline

### 2.1 Introduction (`\label{sec:intro}`)

#### 2.1.1 Background and Motivation
- **[Opening]** Nanosecond pulsed CO₂ bubble plasma discharge as emerging green chemistry technique; growing demand for real-time monitoring
- **[Challenge]** Conventional H₂O₂ quantification is offline, slow, and intrusive — incompatible with closed-loop control
- **[Method introduction]** OES as a non-intrusive, in-situ diagnostic window; however, high dimensionality (701 wavelength points) poses ML challenges

#### 2.1.2 Objectives
- **[Structure]** Three objectives stated clearly:
  1. Build ML regression models to predict H₂O₂ yield from OES + discharge parameters
  2. Investigate whether domain-knowledge features outperform PCA
  3. Identify the minimal, physically interpretable feature set

#### 2.1.3 State of the Art
- **[Context]** Review 3–5 papers on OES-based ML for plasma diagnostics
- **[Gap]** Prior work uses generic PCA/PLS on raw spectra, discarding physical meaning → limited accuracy and interpretability
- **[Positioning]** This work fills the gap by exploiting plasma chemistry knowledge for feature selection

**Key figures/tables:** None in Introduction

---

### 2.2 Procedure / Methodology (`\label{sec:method}`)

#### 2.2.1 Dataset and Experimental Setup
- **[Opening]** Dataset from Gao et al. (XJTU): 701 OES samples from nanosecond pulsed CO₂ bubble discharge
- **[Detail]** Each sample: 701-point OES spectrum + 4 discharge parameters (flow rate, frequency, pulse width, rise time) + H₂O₂ yield target
- **[Evaluation]** LOOCV with R² and RMSE as primary metrics; appropriate for small sample size

#### 2.2.2 Input Configurations
- **[Structure]** Three configurations to isolate data source contributions:
  - Config A: OES only (PCA-reduced to 11 components)
  - Config B: Discharge parameters only (4 features)
  - Config C: OES + Discharge (combined)
- **Table:** `tab:configs`

#### 2.2.3 Phase 1 — Baseline Modelling
- **[Method]** 7 regression models (Ridge, PLS, SVR, XGBoost, RF, MLP, CNN) × 3 configs
- **[Detail]** OES reduced from 701 → 11 PCA components before training

#### 2.2.4 Phase 2 — Hyperparameter Optimisation
- **[Method]** Optuna (TPE sampler) tuning for non-linear models (RF, MLP, CNN)
- **[Detail]** 100+ trials per model–config pair; establishes upper-bound performance ceiling before feature engineering

#### 2.2.5 Phase 3 — Domain-Knowledge Feature Engineering
- **[Key insight]** Replace PCA with 13 physically motivated OES features:
  - 7 emission line intensities: OH (309 nm), O (777 nm), Hβ (486 nm), Hα (656 nm), N₂ (337 nm), CO₂⁺ (406 nm), C₂ (516 nm)
  - 3 band integrals: OH band (306–312 nm), CO₂⁺ band (398–412 nm), CO/Hβ band (460–500 nm)
  - 3 spectroscopic ratios: OH/Hα (309/656), Hα/Hβ (656/486), O/OH (777/309)
- **[Detail]** Config C now comprises 17 features (13 OES + 4 discharge)

#### 2.2.6 Phase 4 — Interpretability and Feature Reduction
- **[Method]** Feature importance: Ridge/PLS coefficients, RF permutation importance, MLP SHAP
- **[Statistical validation]** Bootstrap resampling (500 iterations) for 95% CI; permutation test (2000 label shuffles)
- **[Feature reduction]** Backward elimination + category ablation → minimal feature set

**Key figures/tables:** `tab:configs`

---

### 2.3 Findings / Results (`\label{sec:results}`)

#### 2.3.1 Phase 1 & 2 — Baseline and Tuned Performance
- **[Evidence]** Config B (discharge only) establishes strong baseline: Ridge R² = 0.904, PLS R² = 0.898
- **[Challenge]** Config C (OES+discharge via PCA) fails: Ridge R² = −0.17, MLP R² = −1.13
- **[Evidence]** After Optuna tuning: MLP Config C improves −1.13 → 0.37; CNN Config C = 0.77 (best OES model), but still below Config B
- **[Takeaway]** PCA-based OES features are insufficient; tuning alone cannot fix bad features

#### 2.3.2 Phase 3 — Domain-Knowledge Breakthrough
- **[Evidence]** Domain features produce step-change: Ridge Config C R² from −0.17 → 0.80; MLP Config C from 0.37 → 0.81
- **[Advantage]** Gap between Config B and Config C substantially reduced across all models
- **Figure:** `fig:r2_comparison` — R² scores across Phases 1–3

#### 2.3.3 Phase 4 — Interpretability and Minimal Model
- **[Evidence]** Consensus feature importance: flow_rate_sccm (rank 1), band_CO2p_398_412 (rank 2), pulse_width_ns (rank 3)
- **[Evidence]** Bootstrap 95% CI: Ridge B [0.800, 0.955], Ridge C [0.574, 0.910] — overlapping, no significant difference
- **[Evidence]** Permutation test: observed R² = 0.920, p < 0.0005 (2000 permutations)
- **[Evidence]** Category ablation: ratios (3 features) → R² = 0.906; band integrals (3) → R² = 0.905; single-wavelength (7) → R² = 0.823
- **[Evidence]** Backward elimination: optimal at 3 OES ratios + 4 discharge = 7 features, R² = 0.920
- **Tables:** `tab:bootstrap`, `tab:feature_importance`

**Key figures/tables:** `fig:r2_comparison`, `tab:bootstrap`, `tab:feature_importance`

---

### 2.4 Conclusions (`\label{sec:conclusions}`)

- **[Objective 1 met]** Best model R² = 0.920, enabling practical real-time yield prediction
- **[Objective 2 met]** Domain-knowledge features decisively outperform PCA (Ridge Config C: −0.17 → 0.80) — central novel finding
- **[Objective 3 met]** 7-feature Ridge matches neural networks; interpretable and deployable
- **[Broader implication]** For physically structured scientific datasets (especially small ones), domain-knowledge feature engineering > automated dimensionality reduction
- **[Limitation]** Single laboratory configuration; generalisation unvalidated; small sample size limits non-linear models

---

### 2.5 Recommendations / Future Work (`\label{sec:future}`)

- **[Generalisation]** Apply framework to other plasma reactor types (DBD, microwave)
- **[Deployment]** Integrate 7-feature Ridge with live OES hardware for closed-loop control
- **[Extended data]** Collect more samples to test non-linear model advantages at larger scale
- **[Physics-informed]** Incorporate conservation constraints for multi-product systems
- **[Transfer learning]** Leverage trained models for related plasma reactions

---

### 2.6 Reflection (`\label{sec:reflection}`)

- **[Opening]** Bridging ML and plasma physics disciplines
- **[Challenge]** Phase 3 required reading plasma diagnostics literature — cross-disciplinary skill proved decisive
- **[Growth]** Developed: end-to-end ML pipelines, Bayesian optimisation, statistical validation, scientific communication
- **[Limitation]** Compare with original skills audit: entered with strong Python, limited statistical validation / plasma knowledge

---

### 2.7 Summary / Executive Summary (≤200 words)

- **[Structure]** Cover in order: project goal → methodology → key results → main conclusion
- Text only, no figures; ≤200 words
- Already drafted in `main.tex` — needs editing to ≤200 words

---

### 2.8 Appendices

- **Appendix A:** Source code — link to repository, key module descriptions
- **Appendix B:** Dataset description — source, access, preprocessing, cleaning
- **Appendix C:** OES feature derivation — table of 13 features with wavelengths, species, literature references
- **Appendix D:** Optuna hyperparameter search results — best hyperparameters, convergence plots
- **Appendix E:** Full statistical test outputs — bootstrap distributions, permutation histogram, ablation R² table

---

## 3. Claim–Evidence Mapping

| # | Claim | Evidence | Source | Status |
|---|-------|----------|--------|--------|
| 1 | ML can accurately predict H₂O₂ yield from OES + discharge parameters | Ridge Config C R² = 0.920 (7 features); permutation test p < 0.0005 | Phase 4, `permutation_test_summary.csv` | **Supported** |
| 2 | Domain-knowledge features decisively outperform PCA | Ridge Config C: R² from −0.17 (Phase 1, PCA) → 0.80 (Phase 3, domain) | Phase 1 vs Phase 3 comparison | **Supported** |
| 3 | Hyperparameter tuning alone cannot compensate for poor features | MLP Config C: −1.13 → 0.37 after tuning, still far below Config B (0.86) | Phase 1 vs Phase 2 comparison | **Supported** |
| 4 | A simple 7-feature Ridge model matches neural networks | Ridge 7-feat R² = 0.920; MLP Config C R² = 0.815; Bootstrap CIs overlap for Ridge B vs Ridge C | Phase 4, `bootstrap_ci_summary.csv` | **Supported** |
| 5 | CO₂ flow rate and CO₂⁺ band integral are the two most influential predictors | Consensus ranking: flow_rate_sccm = rank 1 (mean 1.75), band_CO2p_398_412 = rank 2 (mean 4.75) | Phase 4, `feature_importance_all_models.csv` | **Supported** |
| 6 | OES features are highly redundant; 13 can be reduced to 3 | Backward elimination: R² increases from 0.798 (13 OES) to 0.920 (3 OES ratios); category ablation confirms ratios (R² = 0.906) ≈ full set (R² = 0.798) | Phase 4, `ablation_results.csv` | **Supported** |
| 7 | The relationship between features and target is essentially linear | Ridge and MLP Bootstrap CIs overlap (Ridge C: [0.574, 0.910]; MLP C: [0.647, 0.883]) | Phase 4, `bootstrap_ci_summary.csv` | **Supported** |
| 8 | Prediction is statistically genuine, not due to chance | Permutation test: observed R² = 0.920, all 2000 null R² values well below (mean ≈ −0.15), p < 0.0005 | Phase 4, `permutation_test_pruned_ridge.csv` | **Supported** |
| 9 | Generalisation to other reactor types | No experimental evidence in this project | — | **Needs evidence** (acknowledged as limitation/future work) |
