# Final Report Outline (Phase 3 — Updated)

> Updated from Phase 2 outline. Changes: §2.2 Methodology substantially expanded with justification notes, paragraph-role annotations, and expanded §2.2.5 (domain features). See `methodology_flow.md` for the full writing logic flow.

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

> **Phase 5 writing principle (from `results_flow.md`):** §2.3 resolves the four hypotheses posed in §2.2 in order. The narrative centre of gravity is §2.3.2 (Phase 3 step-change); Phase 4 supplies interpretability and statistical confirmation. Frame Phase 1/2 as *hypothesis tests rejected*, never as "failures". All figures live in `final report/report/images/`. See `results_flow.md` for the 9-beat forward story (R0–R9), anti-pattern warnings, and the verified quantitative fact sheet.

#### 2.3.0 Section opener (R0)
- **[Role: Roadmap]** One-paragraph roadmap listing the four hypotheses from §2.2 and signalling that the headline finding is the Phase 3 step-change plus its Phase 4 confirmation.

#### 2.3.1 Phase 1 & 2 — Baseline Ceiling and the PCA Dead End
- **[Evidence, R1]** Config B (discharge only) establishes a strong baseline that any OES-inclusive model must beat: Ridge R² = **0.904**, PLS R² = **0.898**.
- **[Evidence + Interpretation, R2 — H1 rejected]** Under PCA, adding OES actively **degrades** Config C: Ridge R² = **−0.175**, MLP R² = **−1.131**. Structural failure, not a cutoff artefact — 11 PCA components already preserve ≥95% of the spectral variance.
- **[Evidence + Interpretation, R3 — H2 rejected, Phase 2 handled inline]** Optuna tuning lifts non-linear models substantially but cannot clear the Config B ceiling from OES: MLP Config C −1.131 → **0.369**; CNN Config C 0.688 → **0.775** (best tuned OES-only model, still below Ridge B 0.904); MLP Config B 0.568 → 0.861. The ceiling is set by feature quality, not model capacity.
- **[Takeaway]** PCA-based OES features do not carry the prediction signal, and hyperparameter tuning cannot compensate — motivating a feature-level intervention in Phase 3.

> **Phase 5 note — Phase 2 handling (from user Plan):** Phase 2 receives **no dedicated figure**. R3 is reported inline via numbers or a compact 3-row table. The paragraph is intentionally short: it is a bridging beat, not an independent finding.

**Figures / tables for §2.3.1**
- `pca_cumulative_variance.png` (R2) — 11 PCs ≥95% variance, pre-empts "too few PCs" objection
- `phase1_model_comparison_bar.png` (R1) — Config B dominates, Config C lags across 7 models
- `phase1_predicted_vs_actual_grid.png` (R2, optional) — visual scatter collapse for Config C non-linear models
- *No figure for Phase 2*; short inline table listing MLP C, CNN C, MLP B tuning deltas if needed

---

#### 2.3.2 Phase 3 — The Domain-Knowledge Step-Change ← **narrative centre**
- **[Evidence + Interpretation, R4 — H3 supported]** Replacing 11 PCA components with 13 physically motivated OES features collapses the Config B / Config C gap across every model class:
  - Ridge Config C: −0.175 → **0.798** (ΔR² ≈ +0.973)
  - MLP Config C: −1.131 → **0.815** (ΔR² ≈ +1.946)
  - PLS Config C: 0.625 → **0.744** (ΔR² ≈ +0.119)
- **[Key point]** The same OES spectra, re-encoded with physical knowledge, unlock the predictive signal that PCA had obscured. This is the central finding of the project and earns the longest paragraph in §2.3 (mirroring the proportional emphasis of §2.2.5 in the Methodology).

**Figures / tables for §2.3.2**
- `phase1_vs_phase2_vs_phase3_comparison.png` — **centrepiece figure**; the step-change made visually unmissable

---

#### 2.3.3 Phase 4 — Interpretability, Uncertainty, and the Minimal Model

**R5 — Feature importance (consensus ranking)**
- **[Evidence]** 4-model consensus (Ridge coefficients, PLS VIP, RF permutation, MLP SHAP) identifies:
  - Rank 1: **flow_rate_sccm** (mean rank 1.75)
  - Rank 2: **band_CO2p_398_412** (4.75)
  - Rank 3: **pulse_width_ns** (5.25)
  - Rank 4 (tie): I_486_Hb and band_CO_Hb_460_500 (6.00)
- **[Interpretation]** The two leading OES features are **bands**, not single-wavelength lines — physically intuitive, since band integrals are drift-invariant.

**R6 — Honest uncertainty via bootstrap**
- **[Evidence]** Bootstrap 95% CIs (500 iterations):
  - Ridge B [0.800, 0.955]
  - Ridge C [0.574, 0.910]
  - MLP  B [0.767, 0.923]
  - MLP  C [0.647, 0.883]
- **[Interpretation]** Ridge B and Ridge C CIs overlap — at the 13-feature stage, adding OES does **not yet deliver a statistically significant improvement**. Ridge C and MLP C CIs also overlap — the feature–target relationship is essentially linear at this feature set. These overlaps must be named, not hidden (anti-pattern AP4).

**R7 — Permutation significance on the pruned model**
- **[Evidence]** Permutation test on the pruned 7-feature Ridge (3 OES ratios + 4 discharge):
  - Observed R² = **0.9200**
  - 2000 label-shuffle permutations → no null R² ≥ observed
  - **p < 0.0005**
- **[Handling]** No figure — per Plan, the existing permutation figure is inadequate. Report via inline sentences or a compact 3-row inline table.

**R8 — Feature reduction converges on a minimal model**
- **[Category ablation, Ridge Config C]**:
  - All 13 OES → R² = 0.798
  - Single-wavelength only (7) → R² = 0.823
  - Band integrals only (3) → R² = **0.905**
  - Ratios only (3) → R² = **0.906**
  - Discharge only (Config B) → R² = 0.904
- **[Backward elimination trajectory]** R² rises **monotonically** from 0.798 (13 OES) to **0.918** at 1 OES feature (band_CO2p_398_412), then drops back to 0.904 (= Config B) when the last OES feature is removed — confirming OES contributes real (if modest) information.

**Feature reduction — two optimal models (disambiguated):**
- **Category-ablation optimal**: 3 OES ratios (OH/Hα, O/OH, Hα/Hβ) + 4 discharge = **7 features → R² = 0.920** (permutation-confirmed, p < 0.0005). **Recommended model** — retains all three ratio types for drift invariance and physical interpretability.
- **Backward-elimination optimal**: 1 OES feature (band_CO2p_398_412) + 4 discharge = 5 features → R² = 0.918. More parsimonious but relies on a single OES feature.
- Both models outperform the full 13-OES model (R² = 0.798), demonstrating severe feature redundancy.

> **Phase 2 addition:** Disambiguated the two "optimal" models to address Phase 1 Observation risk.
> **Phase 5 note:** The ratios-only category-ablation R² (0.906) and the permutation-test observed R² (0.920) refer to **different fits** — the permutation test model was refit on the reduced set. Make this distinction explicit in the prose.

**R9 — Synthesis paragraph**
- **[Transition]** One short paragraph closing the evidence chain: PCA fails (R1–R2) → tuning cannot rescue (R3) → domain features succeed (R4) → minimal interpretable subset is sufficient (R5–R8). Ridge is chosen for deployment by Occam's razor (R6 overlap), not by capability gap. Hand off to §2.4 Conclusions.

**Figures / tables for §2.3.3**
- `fig1_feature_importance_heatmap.pdf` (R5) — consensus importance across 4 methods
- `fig5_bootstrap_r2_distributions.pdf` (R6) — bootstrap R² distributions and 95% CIs
- `rank_on_model_config.png` (R8) — overall model × config ranking, locating the pruned Ridge
- *No permutation figure*; inline report
- Tables: `tab:bootstrap`, `tab:feature_importance`, `tab:ablation` (new or merged)

**Overall key figures/tables for §2.3:** `pca_cumulative_variance.png`, `phase1_model_comparison_bar.png`, `phase1_predicted_vs_actual_grid.png`, `phase1_vs_phase2_vs_phase3_comparison.png` (centrepiece), `fig1_feature_importance_heatmap.pdf`, `fig5_bootstrap_r2_distributions.pdf`, `rank_on_model_config.png`; `tab:bootstrap`, `tab:feature_importance`, `tab:ablation`.

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
