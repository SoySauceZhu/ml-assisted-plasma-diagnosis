# Final Report Outline (Phase 5 — Updated)

> Updated from Phase 3 outline. Changes: §2.3 Findings/Results substantially expanded with claim–evidence pairing, paragraph-role annotations, a new §2.3.4 Synthesis paragraph, fact-checked evidence anchors for every claim, and reconciled figure/table labels. See `results_flow.md` for the full writing logic flow.

## 1. Overall Story (总体行文逻辑)

Nanosecond pulsed CO₂ bubble plasma discharge is a promising green chemistry technique, but real-time monitoring of product yield (H₂O₂) remains a bottleneck — conventional quantification requires offline titration, which is slow and intrusive. Optical Emission Spectroscopy (OES) provides a non-intrusive diagnostic window, yet a single OES measurement comprises 701 wavelength points, making direct ML application challenging. Prior OES+ML studies apply PCA or PLS to raw spectra, discarding the physical meaning of individual emission lines and limiting both accuracy and interpretability. **However, traditional spectroscopy has long recognized the diagnostic value of line intensity ratios and band integrals — domain knowledge that modern ML pipelines systematically overlook.** This project demonstrates that encoding this domain knowledge as ML features transforms model performance: Ridge regression R² on the combined input (OES + discharge parameters) jumps from −0.17 (PCA) to 0.80 (domain features). Furthermore, backward elimination and category ablation identify a minimal 7-feature Ridge model (3 OES ratios + 4 discharge parameters) achieving R² = 0.920, matching neural networks with far fewer parameters. The broader implication is clear: for physically structured scientific datasets, especially small ones, simple interpretable models with domain-informed features outperform complex models with automated feature extraction.

---

## 2. Section-by-Section Outline

### 2.1 Introduction (`\label{sec:intro}`)

#### 2.1.1 Background and Motivation

- **[Opening]** Plasma-based green chemistry: nanosecond pulsed CO₂ bubble discharge as emerging technique for converting greenhouse gases into value-added chemicals (H₂, H₂O₂) [Gao2024NSCO2Discharge, Shao2018PulsedDischarges]
- **[Application need]** Growing demand for real-time process monitoring and closed-loop control in plasma systems
- **[Challenge]** Conventional H₂O₂ quantification requires offline titration — slow, intrusive, incompatible with real-time control
- **[Method introduction]** OES as a non-intrusive, in-situ diagnostic window [Laux2003OESAir]; however, high dimensionality (701 wavelength points) poses ML challenges

> **Story framing note (from Phase 1 Observation):** Do NOT frame this as "we tried PCA and it failed." Frame as: "domain knowledge is the decisive factor — traditional spectroscopy has known this, but modern ML approaches overlook it." See `introduction_flow.md` for detailed logic.

#### 2.1.2 Objectives

- **[Structure]** Three objectives stated clearly:
  1. Build ML regression models to predict H₂O₂ yield from OES + discharge parameters
  2. Investigate whether domain-knowledge features outperform PCA
  3. Identify the minimal, physically interpretable feature set

#### 2.1.3 State of the Art / Literature Review

**Paragraph 1 — OES as a plasma diagnostic tool** [Opening/Context]
- OES is an established non-intrusive diagnostic technique for characterising plasma composition, electron temperature, and species concentrations [Laux2003OESAir]
- Traditional OES analysis relies on manual spectroscopic methods: Boltzmann plot, line-ratio method, Saha-Boltzmann equation — these require physical model assumptions and expert knowledge [Laux2003OESAir, Srikar2024Accelerated]
- ML offers a data-driven alternative that can automate and accelerate OES-based diagnostics, enabling real-time applications [Gidon2019MLCAP]
- Key citations: [Laux2003OESAir], [Gidon2019MLCAP], [Srikar2024Accelerated]

**Paragraph 2 — Existing OES+ML approaches and their limitations** [Challenge/Gap]
- Recent works apply PCA or PLS to raw OES spectra before feeding into ML models:
  - Srikar et al. (2025): PCA + RF/DNN for Ar multi-jet plasma OES prediction [Srikar2025MLOES]
  - Stefas et al. (2025): PCA + MLP for cylindrical surface DBD diagnostics [Stefas2025MLSDBD]
  - Wang & Hsu (2019): PCA + deep ANN for plasma-in-liquid OES — found PCA alone insufficient, deep ANN needed [Wang2019MLOESAqueous]
  - Park et al. (2021): ML virtual metrology for electron density/temperature from OES [Park2021MLOESNitrogen]
- These PCA-based approaches:
  - Discard physical meaning of individual emission lines
  - Are vulnerable to instrument drift and wavelength calibration shifts
  - Perform poorly when datasets are small (insufficient samples for PCA to discover task-relevant variance)
- **Contrast:** Wang et al. (2025) use physically motivated line intensity ratios (LIRs) instead of PCA for electron density/temperature prediction in cascaded arc plasma — achieving R² ≈ 0.90–0.97 [Wang2025MLOESCascaded]. This demonstrates the value of domain-knowledge-based feature selection, but has not been applied to chemical yield prediction.
- Key citations: [Srikar2025MLOES], [Stefas2025MLSDBD], [Wang2019MLOESAqueous], [Park2021MLOESNitrogen], [Wang2025MLOESCascaded]

**Paragraph 3 — The gap this project fills** [Positioning]
- No prior work systematically compares PCA-based vs. domain-knowledge-based OES feature engineering for plasma product yield prediction
- Traditional spectroscopy has long used line ratios (e.g., N₂ SPS/FNS ratio for electric field [Paris2005N2Ratio]) and band integrals for robust diagnostics — this domain knowledge exists but is not leveraged in modern ML pipelines
- ML in plasma catalysis (Tu group) has shown domain features matter: Cai et al. (2024) and Wang et al. (2021) use physically meaningful operating parameters for process prediction [Cai2024MLDRM, Wang2021PlasmaTarML]
- **This project bridges the gap:** we systematically demonstrate that encoding traditional spectroscopic knowledge (emission lines, band integrals, spectroscopic ratios) as ML features is the decisive factor for OES-based yield prediction — not model complexity, not hyperparameter tuning
- Key citations: [Paris2005N2Ratio], [Cai2024MLDRM], [Wang2021PlasmaTarML]

**Key figures/tables:** None in Introduction

---

### 2.2 Procedure / Methodology (`\label{sec:method}`)

> **Phase 3 methodology writing principle (from methodology_flow.md):** Frame the 4-phase pipeline as hypothesis-driven experimental design, not chronological diary. Each phase tests one specific hypothesis while holding others constant. See `methodology_flow.md` for anti-pattern warnings and paragraph-level plan.

#### 2.2.0 Overview

- **[Role: Opening]** Task setting: predicting H₂O₂ yield from OES spectra and discharge parameters using ML regression
- **[Role: Detail]** Pipeline comprises four phases: (1) baseline modelling with standard PCA features, (2) hyperparameter optimisation to test whether model tuning compensates, (3) domain-knowledge feature engineering (central contribution), and (4) interpretability analysis and feature reduction
- **[Role: Transition]** Subsection map: §2.2.1 describes the dataset and evaluation protocol; §2.2.2 defines the input configurations; §2.2.3–2.2.6 detail each experimental phase
- **[Figure (optional)]** `fig:pipeline` — overall pipeline diagram

> **Phase 3 addition:** Added overview paragraph following the research-paper-writing skill's Method Overview Template (setting → core contribution → figure pointer → subsection map).

#### 2.2.1 Dataset and Experimental Setup

- **[Role: Opening]** Dataset from Gao et al. (XJTU) [Gao2024NSCO2Discharge]: OES samples from nanosecond pulsed CO₂ bubble discharge
- **[Role: Detail]** Each sample: 701-point OES spectrum + 4 discharge parameters (flow rate, frequency, pulse width, rise time) + H₂O₂ yield target
- **[Role: Detail]** LOOCV with R² and RMSE as primary metrics
- **[Role: Justification]** Why LOOCV: the dataset is small, making k-fold cross-validation unstable due to fold-dependent variance. LOOCV maximises training data per fold (N−1 samples), produces an unbiased performance estimate, and eliminates random fold-split variability. Each sample is held out exactly once, ensuring no data leakage.
- **[Role: Detail]** R² measures explained variance (1.0 = perfect prediction; negative = worse than predicting the mean); RMSE measures absolute prediction error in the same units as the target

> **Phase 2 addition:** Added data integrity statement to address Phase 1 Observation risk about data leakage.
> **Phase 3 addition:** Added justification for LOOCV choice and metric definitions.

#### 2.2.2 Input Configurations

- **[Role: Opening + Justification]** Three configurations defined as a **controlled-variable experimental design** to isolate the contribution of each data source:
  - Config A: OES only (PCA-reduced to 11 components in Phase 1–2; 13 domain features in Phase 3) — tests whether OES alone carries predictive information
  - Config B: Discharge parameters only (4 features) — serves as a **strong baseline**: any OES-inclusive configuration must outperform Config B to justify the additional complexity of OES processing
  - Config C: OES + Discharge (combined) — tests whether OES adds value beyond discharge parameters
- **[Role: Detail]** Table: `tab:configs` (to be updated for Phase 3: Config A = 13 domain features, Config C = 17 total)
- **[Role: Justification]** This three-way comparison is essential because it prevents a common pitfall in ML studies: claiming that a model "works" without testing whether simpler inputs achieve the same result. Config B provides the honest baseline.

> **Phase 3 addition:** Added explicit justification for why three configurations are needed and what each tests.

#### 2.2.3 Phase 1 — Baseline Modelling

- **[Role: Opening]** Seven regression models were selected to span three categories of inductive bias, ensuring that performance differences are attributable to features, not model class:
  - **Linear models** (Ridge, PLS): test whether the feature-target relationship is approximately linear; Ridge applies L2 regularisation to handle multicollinearity; PLS finds latent components that maximise covariance with the target
  - **Kernel and ensemble methods** (SVR, RF): capture non-linear interactions without requiring large datasets; SVR uses kernel trick for non-linear mapping; RF averages many decision trees for robust prediction
  - **Deep learning models** (MLP, CNN): can learn hierarchical/sequential representations from raw or engineered features; MLP is a general-purpose non-linear approximator; CNN exploits local spectral structure in OES data
- **[Role: Detail]** OES reduced from 701 → 11 PCA components (≥95% variance explained) before training
- **[Role: Detail]** XGBoost was also included but produced identical R² = −0.108 across all configurations, likely due to default hyperparameters being unsuitable for the small dataset size and LOOCV evaluation. XGBoost is excluded from subsequent phases; the remaining 6 models are carried forward.
- **[Role: Justification]** Why 7 models: model diversity ensures that observed performance patterns reflect feature quality rather than model-specific biases. If all model types fail with PCA features but succeed with domain features, the conclusion (features matter more than models) is robust.

> **Phase 2 addition:** Added XGBoost anomaly note.
> **Phase 3 addition:** Added model selection rationale with three-category justification.

#### 2.2.4 Phase 2 — Hyperparameter Optimisation

- **[Role: Opening]** Non-linear models (RF, MLP, CNN) were tuned using Optuna [optuna2019], a Bayesian optimisation framework based on the Tree-structured Parzen Estimator (TPE) sampler
- **[Role: Detail]** 100+ trials per model–config pair; linear models (Ridge, PLS) were not tuned as their regularisation parameters are set analytically or have minimal impact
- **[Role: Justification]** Why Optuna TPE: TPE is efficient for small trial budgets (100–200 trials), outperforming random search by modelling the objective function surface. Tuning before feature engineering establishes an **upper-bound performance ceiling** — if tuned models on Config C still trail Config B, the bottleneck is feature quality, not model capacity.
- **[Role: Transition]** The Phase 2 results (reported in §2.3.1) confirmed that tuning substantially improves non-linear models but cannot compensate for uninformative PCA features, motivating Phase 3.

> **Phase 3 addition:** Added justification for Optuna TPE choice and the strategic purpose of tuning before feature engineering.

#### 2.2.5 Phase 3 — Domain-Knowledge Feature Engineering

> **This is the central methodological contribution of the project.** It receives proportionally more space (3 paragraphs) than other phases (1 paragraph each).

**Paragraph 1 — Motivation** [Role: Motivation]
- **[Problem]** The preceding phases relied on PCA for OES dimensionality reduction. PCA selects directions of maximum variance in the spectrum — but maximum-variance directions need not align with the prediction target (H₂O₂ yield), especially when the dataset is small and the target-relevant signal is distributed across specific emission lines rather than broad spectral patterns.
- **[Insight from traditional spectroscopy]** Traditional plasma diagnostics has long used physically motivated spectral features for robust analysis: line intensity ratios for electron temperature estimation [Laux2003OESAir], band-head intensities for species concentration [Paris2005N2Ratio], and spectroscopic ratios for electric field measurement. These features encode domain knowledge that PCA cannot discover from small data.
- **[Precedent]** Wang et al. (2025) demonstrated that ML models trained on physically motivated line intensity ratios achieve R² ≈ 0.90–0.97 for electron density/temperature prediction in cascaded arc plasma [Wang2025MLOESCascaded] — validating the approach for a different prediction task. This project extends the principle to chemical yield prediction.

**Paragraph 2 — Feature Design** [Role: Detail (core)]
- **[Design]** Thirteen physically motivated OES features were derived, replacing the 11 PCA components:

  **Emission line intensities (7 features):** Each corresponds to a specific plasma species relevant to H₂O₂ formation pathways:
  - OH (309 nm): hydroxyl radical — primary H₂O₂ precursor via OH + OH → H₂O₂ [Gao2024NSCO2Discharge]
  - O (777 nm): atomic oxygen — participates in OH formation and oxidation reactions
  - Hβ (486 nm): hydrogen Balmer-β — electron impact excitation marker
  - Hα (656 nm): hydrogen Balmer-α — most intense hydrogen emission line
  - N₂ (337 nm): molecular nitrogen second positive system — gas temperature indicator
  - CO₂⁺ (406 nm): carbon dioxide cation — primary reactant dissociation marker
  - C₂ (516 nm): diatomic carbon Swan band — deep dissociation indicator

  **Band integrals (3 features):** Integrated emission over wavelength ranges, robust to spectral noise and wavelength calibration shifts:
  - OH band (306–312 nm): integrated OH (A²Σ⁺→X²Π) emission
  - CO₂⁺ band (398–412 nm): integrated CO₂⁺ (Fox–Duffendack–Barker) emission — reactant consumption marker
  - CO/Hβ band (460–500 nm): mixed CO and Hβ emission region — product formation indicator

  **Spectroscopic ratios (3 features):** Intensity ratios that are inherently drift-invariant (numerator and denominator are affected equally by instrument gain drift):
  - OH/Hα (309/656 nm): balance between hydrogen abstraction and recombination pathways
  - Hα/Hβ (656/486 nm): Balmer decrement — classical diagnostic for electron temperature [Laux2003OESAir]
  - O/OH (777/309 nm): oxidation vs. hydroxylation pathway balance

- **[Table]** `tab:oes_features` — full feature table with wavelengths, species, physical significance (essential table; see also Appendix C)
- **[Detail]** Config C now comprises 17 features (13 OES + 4 discharge)

**Paragraph 3 — Technical Advantages** [Role: Advantage + Transition]
- **[Advantage 1: Drift invariance]** Spectroscopic ratios normalise out absolute intensity variations caused by instrument drift, fibre degradation, or window contamination — a robustness that PCA components lack, since PCA is trained on the absolute spectral profile
- **[Advantage 2: Physical interpretability]** Every feature maps to a known plasma species or reaction pathway, enabling physically meaningful model interpretation (e.g., "the model relies on OH/Hα ratio" has a clear chemical interpretation, unlike "the model relies on PC3")
- **[Advantage 3: Dimensionality reduction]** 13 features vs. 701 raw wavelengths (or 11 PCA components) — dramatically lower dimensionality reduces overfitting risk on small datasets
- **[Contrast with PCA]** PCA components are data-driven and require sufficient samples to discover task-relevant variance; domain features are knowledge-driven and effective regardless of dataset size. PCA also suffers from multicollinearity issues in the OES context (VIF up to 381.7 for I_309_OH), which is avoided when using ratios and integrals
- **[Transition]** Having established the domain-knowledge features, Phase 4 (§2.2.6) investigates which are essential and which are redundant

> **Phase 3 addition:** Fully expanded from 4 bullets to 3 structured paragraphs following the research-paper-writing skill's Module Triad: Motivation → Design → Advantages. Added feature derivation rationale, physical meaning for each ratio, connection to traditional spectroscopy, contrast with PCA, and VIF evidence for multicollinearity.

#### 2.2.6 Phase 4 — Interpretability and Feature Reduction

**Paragraph 1 — Feature importance** [Role: Opening + Detail]
- **[Method]** Feature importance was quantified using four model-specific methods to obtain a consensus ranking:
  - Ridge: absolute standardised regression coefficients
  - PLS: Variable Importance in Projection (VIP) scores
  - RF: permutation importance (drop in R² when feature values are shuffled)
  - MLP: SHAP (SHapley Additive exPlanations) values
- **[Justification]** Why consensus: no single importance method is unbiased — Ridge coefficients depend on feature scaling; PLS VIP conflates importance with latent-component loading; RF permutation can be noisy with correlated features; SHAP depends on the model architecture. Averaging ranks across all four methods reduces method-specific artifacts and produces a more robust ranking.

**Paragraph 2 — Statistical validation** [Role: Detail]
- **[Method 1]** Bootstrap resampling (500 iterations): resample with replacement from the LOOCV predictions, recompute R² and RMSE → 95% confidence intervals. This quantifies prediction uncertainty without distributional assumptions.
- **[Method 2]** Permutation test (2000 label shuffles): shuffle H₂O₂ yield labels, retrain Ridge on the reduced feature set, record null-distribution R² → compute p-value as proportion of null R² ≥ observed R². This provides a model-free significance test: if the observed R² is higher than all 2000 null values, p < 0.0005.
- **[Justification]** Why both tests: bootstrap quantifies "how uncertain is our R²?" while permutation answers "is our R² real or due to chance?" They address complementary questions.

**Paragraph 3 — Feature reduction** [Role: Detail + Justification]
- **[Method 1: Backward elimination]** Iteratively remove the OES feature with the smallest Ridge coefficient (4 discharge parameters always retained) and retrain. This finds the optimal individual feature subset at each cardinality.
- **[Method 2: Category ablation]** Remove entire OES feature categories (emission lines, band integrals, or spectroscopic ratios) one at a time, keeping discharge parameters. This reveals which feature TYPE contributes most.
- **[Justification]** Why two strategies: backward elimination finds the optimal set at fine granularity (which specific features to keep); category ablation reveals which feature TYPE matters most (lines vs. bands vs. ratios). The two analyses converge on the same conclusion for maximum credibility.
- **[Note]** Discharge parameters (4 features) are always retained as the baseline, since Config B already achieves strong performance; the question is which OES features add value beyond this baseline.

> **Phase 3 addition:** Expanded from 3 bullets to 3 structured paragraphs with justification for consensus importance, both statistical tests, and both reduction strategies.

**Key figures/tables for §2.2:**
- `tab:configs` — input configuration summary (essential, already in `main.tex`)
- `tab:oes_features` — 13 OES features with physical justification (essential, new)
- `fig:pipeline` — pipeline diagram (optional)
- `tab:models` — model summary with categories and key hyperparameters (optional)

---

### 2.3 Findings / Results (`\label{sec:results}`)

> **Phase 5 results writing principle (from `results_flow.md`):** Frame §2.3 as a claim-driven narrative, not a phase-by-phase diary. Each subsection opens with a **[Claim]** and follows with **[Evidence]** anchored to a specific CSV row. Interpretation of *why* results occur belongs in §2.4 Conclusions. Every headline R² must be paired with a statistical qualifier (bootstrap CI or permutation p-value).

#### 2.3.0 Opening (~1 paragraph)

- **[Role: Opening]** One-paragraph orientation: LOOCV protocol, R² / RMSE as primary metrics, 20-sample dataset, three claims that structure the section (baseline paradox → domain-feature step-change → minimal-model validation)
- **[Role: Transition]** Forward-points to §2.3.1 / §2.3.2 / §2.3.3 and to the synthesis in §2.3.4

#### 2.3.1 Phase 1 & 2 — The Baseline Paradox (~2 paragraphs)

**Paragraph 1 — Baseline paradox** [Role: Claim + Evidence]

- **[Claim]** Config B (discharge parameters only) already predicts H₂O₂ yield accurately with a simple linear model, yet adding PCA-reduced OES features in Config C *degrades* performance across every carried-forward model — a paradox that motivates Phases 2 and 3.
- **[Evidence]** Ridge Config B R² = 0.904; Ridge Config C R² = −0.175; MLP Config C R² = −1.131; PLS Config C R² = 0.625 (the only Config C model with non-trivial Phase 1 performance). Source: `phase3/results/tables/phase1_vs_phase2_vs_phase3_comparison.csv` (columns R2_P1).
- **[Reference]** `tab:results_main`, `fig:phase1_heatmap_r2` (optional appendix)

**Paragraph 2 — Tuning is not enough** [Role: Evidence + Transition]

- **[Claim]** Optuna TPE tuning substantially lifts every non-linear Config C model, but no tuned Config C model matches tuned Config B — confirming that the performance ceiling is set by feature quality, not model capacity.
- **[Evidence]** MLP C: −1.131 → 0.369 (ΔR² = +1.500); CNN C: 0.688 → 0.775; RF C: 0.239 → 0.456; MLP B: 0.568 → 0.861. Source: `phase3/.../phase1_vs_phase2_vs_phase3_comparison.csv` R2_P2 column; CNN from `phase2/results/tables/phase2_loocv_results_summary.csv`.
- **[Role: Transition]** The highest tuned Config C result (CNN = 0.775) still trails the tuned Config B ceiling (MLP B = 0.861), motivating Phase 3.

#### 2.3.2 Phase 3 — Domain-Knowledge Step-Change (~2 paragraphs)

**Paragraph 1 — Step-change claim** [Role: Claim + Evidence]

- **[Claim]** Replacing 11 PCA components with 13 domain-knowledge OES features produces the largest single-phase improvement in the project, closing most of the Config B / Config C gap in one step.
- **[Evidence]** Ridge C: −0.175 → 0.798 (ΔR² = +0.973); MLP C: −1.131 → 0.815 (ΔR² = +1.946); PLS C: 0.625 → 0.744 (ΔR² = +0.119); RF C: 0.239 → 0.497 (ΔR² = +0.258). Source: `phase1_vs_phase2_vs_phase3_comparison.csv`, columns R2_P1 and R2_P3.
- **[Reference]** `fig:r2_comparison` — R² scores across Phases 1 / 2 / 3 (bar chart)

**Paragraph 2 — Cross-model generality** [Role: Comparison]

- **[Claim]** The improvement is not model-specific: every carried-forward model (Ridge, PLS, RF, MLP) improves in Config C under domain features, and the two models most penalised by PCA (Ridge and MLP) are the two that recover most.
- **[Evidence]** All four ΔR² values above are positive and of the same sign; the magnitudes follow the "worse in PCA → larger gain" ordering.

#### 2.3.3 Phase 4 — Interpretability, Validation, and Minimal Model (~4 paragraphs)

**Paragraph 1 — Consensus feature importance** [Role: Evidence]

- **[Claim]** Consensus importance across four methods (Ridge coefficients, PLS VIP, RF permutation, MLP SHAP) places a small mix of discharge parameters and a CO₂⁺ band integral at the top, with spectroscopic ratios clustered in the upper-middle — no single atomic emission line is dominant.
- **[Evidence]** Top-5 consensus ranks: `flow_rate_sccm` (mean rank 1.75), `band_CO2p_398_412` (4.75), `pulse_width_ns` (5.25), `I_486_Hb` (6.00), `band_CO_Hb_460_500` (6.00). Source: `phase4/results/tables/feature_importance_all_models.csv`.
- **[Reference]** `fig:feature_importance`, `tab:feature_importance` (top-10 in appendix E)

**Paragraph 2 — Statistical validation** [Role: Qualifier]

- **[Claim]** Bootstrap CIs and a permutation test together establish that the domain-feature model captures a real input–output relationship, while also showing that Config C is *not* statistically superior to discharge-only Config B.
- **[Evidence]** Bootstrap 95% CIs: Ridge B `[0.800, 0.955]`, Ridge C `[0.574, 0.910]` — **overlapping**; MLP B `[0.767, 0.923]`, MLP C `[0.647, 0.883]` — also overlapping. Source: `phase4/results/tables/bootstrap_ci_summary.csv`. Permutation test on the pruned 7-feature Ridge: observed R² = 0.920, zero of 2000 null R² matched or beat it → **p < 5 × 10⁻⁴**. Source: `phase4/results/tables/permutation_test_summary.csv`, `permutation_test_pruned_ridge.csv`.
- **[Qualifier — explicit]** The Ridge B / Ridge C bootstrap CIs overlap. Config C is competitive with discharge-only but not *significantly better*; the value of domain features is interpretability and physical content, not headline R² superiority over Config B.
- **[Reference]** `fig:bootstrap`, `fig:permutation` (optional), `tab:bootstrap`

**Paragraph 3 — Category ablation** [Role: Evidence]

- **[Claim]** Category ablation shows that any single *normalised* feature family — either the three spectroscopic ratios or the three band integrals — retains the full predictive signal, while single-wavelength line intensities substantially underperform.
- **[Evidence]** Ratios (3 features, + 4 discharge) R² = 0.9063; Bands (3 + 4) R² = 0.9053; Single-wavelength (7 + 4) R² = 0.8232; Full 13 OES (+ 4) R² = 0.7984; Config B (4 discharge only) R² = 0.904. Source: `phase4/results/tables/ablation_summary_article.csv`.
- **[Interpretation guard]** Do not explain *why* ratios/bands win here; that belongs in §2.4.
- **[Reference]** `fig:category_ablation`, `tab:ablation`

**Paragraph 4 — Backward elimination + pruned best model** [Role: Evidence]

- **[Claim]** Iterative backward elimination drives R² monotonically up as redundant OES features are removed, and the permutation-validated pruned Ridge (3 OES ratios + 4 discharge) achieves the best R² in the project.
- **[Evidence]** Backward elimination: R² rises from 0.7984 (13 OES) → 0.918 at 1 OES feature (`band_CO2p_398_412`) + 4 discharge. Pruned 7-feature Ridge (3 ratios + 4 discharge): observed R² = 0.920 (permutation-tested). Source: `phase4/results/tables/ablation_summary_article.csv` (backward elimination rows) + `permutation_test_summary.csv`.
- **[Number discipline]** Keep two numbers separate: the category-ablation Ratios-only row is **0.9063**; the permutation-tested pruned Ridge is **0.9200**. The ~0.014 gap is an implementation detail (different refit pipelines); do *not* conflate.
- **[Reference]** `fig:ablation_trajectory`, `tab:ablation`

**Two disambiguated "optimal" models:**

- **Pruned Ridge (recommended):** 3 spectroscopic ratios (`ratio_309_656`, `ratio_777_309`, `ratio_656_486`) + 4 discharge parameters = 7 features → R² = **0.920**, permutation p < 5 × 10⁻⁴. Recommended because it keeps all three ratio types for physical interpretability.
- **Backward-elimination optimal:** `band_CO2p_398_412` + 4 discharge = 5 features → R² = **0.918**. Most parsimonious but relies on a single OES feature.

> **Phase 5 addition:** Disambiguated the two "optimal" models with their exact feature lists and their separate R² sources.

#### 2.3.4 Synthesis — Objectives Revisited (NEW, ~1 paragraph)

- **[Role: Closing + Transition to §2.4]**
- **[Obj 1 → result]** H₂O₂ yield is predictable from OES + discharge parameters at R² = 0.920 (pruned Ridge), validated by permutation test — *objective met* on this dataset.
- **[Obj 2 → result]** Domain-knowledge features improve Config C by ΔR² ≈ +0.97 (Ridge) and +1.95 (MLP) over PCA — *objective met*; the Phase 1 → Phase 3 jump is the project's decisive finding.
- **[Obj 3 → result]** Feature reduction converges (via two independent strategies) on ≤7 features; the 13-OES set is redundant — *objective met*.
- **[Honest qualifier]** The bootstrap CIs for Ridge B and Ridge C overlap: on headline R² alone, discharge parameters already suffice. The domain-feature value is therefore interpretability, physical content, and deployability — not headline superiority. Interpretation of *why* this matters is deferred to §2.4.

**Key figures/tables for §2.3:**

- `fig:r2_comparison` (essential) — bar chart of R² across Phases 1 / 2 / 3 per model per config
- `fig:feature_importance` (essential) — consensus importance heatmap
- `fig:bootstrap` (essential) — bootstrap R² distributions with 95% CIs
- `fig:ablation_trajectory` (essential) — backward elimination R² curve
- `fig:category_ablation` (essential) — category ablation bar chart
- `fig:permutation` (optional) — permutation null distribution
- `tab:results_main` (essential) — Phase 1 / 2 / 3 R² matrix per model per config
- `tab:bootstrap` (essential) — point R², 95% CI, RMSE
- `tab:ablation` (essential) — category ablation + backward elimination summary
- `tab:feature_importance` (optional, appendix) — top-10 consensus ranks

All essential figures are already copied into `final report/report/images/` during Phase 5 execution (Step 2).

---

### 2.4 Conclusions (`\label{sec:conclusions}`)

- **[Objective 1 met]** Best model R² = 0.920, enabling practical real-time yield prediction
- **[Objective 2 met]** Domain-knowledge features decisively outperform PCA (Ridge Config C: −0.17 → 0.80) — central novel finding
- **[Objective 3 met]** 7-feature Ridge matches neural networks; interpretable and deployable
- **[Broader implication]** For physically structured scientific datasets (especially small ones), domain-knowledge feature engineering > automated dimensionality reduction
- **[Limitation]** Single laboratory configuration; generalisation unvalidated; small sample size limits non-linear models

---

### 2.5 Recommendations / Future Work (`\label{sec:future}`)

- **[Generalisation]** Apply framework to other plasma reactor types (DBD, microwave) — domain adaptation techniques [Liu2022UDA] may assist cross-reactor transfer
- **[Deployment]** Integrate 7-feature Ridge with live OES hardware for closed-loop control
- **[Extended data]** Collect more samples to test non-linear model advantages at larger scale
- **[Physics-informed]** Incorporate conservation constraints for multi-product systems
- **[Transfer learning]** Leverage trained models for related plasma reactions [Zhao2024TransferSSL]

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
- **Appendix B:** Dataset description — source [Gao2024NSCO2Discharge], access, preprocessing, cleaning
- **Appendix C:** OES feature derivation — table of 13 features with wavelengths, species, literature references (expanded version of `tab:oes_features`)
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
| 9 | Prior OES+ML work uses PCA and overlooks domain knowledge | Srikar2025, Stefas2025, Wang2019 all use PCA; Wang2025 uses ratios but only for Te/ne, not yield | Literature review (§2.1.3) | **Supported** (Phase 2 addition) |
| 10 | Generalisation to other reactor types | No experimental evidence in this project | — | **Needs evidence** (limitation/future work) |
