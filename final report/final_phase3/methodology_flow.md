# Methodology Writing Logic Flow (行文思路)

## 1. Pre-Writing Questions (research-paper-writing skill)

Following the Method Writing Guide's pre-writing framework: identify all modules, then for each answer **How / Why needed / Why it works**.

### 1.1 Module Inventory

| # | Module | Subsection |
|:---:|--------|-----------|
| M0 | Overview (setting + pipeline summary) | §2.2 opening |
| M1 | Dataset and evaluation protocol | §2.2.1 |
| M2 | Input configurations (controlled-variable design) | §2.2.2 |
| M3 | Phase 1 — Baseline modelling (PCA + 7 models) | §2.2.3 |
| M4 | Phase 2 — Hyperparameter optimisation (Optuna) | §2.2.4 |
| M5 | Phase 3 — Domain-knowledge feature engineering | §2.2.5 |
| M6 | Phase 4 — Interpretability and feature reduction | §2.2.6 |

### 1.2 Three-Question Analysis per Module

| Module | How does it run? | Why do we need it? | Why does it work? |
|--------|-----------------|-------------------|-------------------|
| **M1: Dataset + Evaluation** | 701-point OES spectra + 4 discharge params + H₂O₂ yield target; LOOCV evaluation with R² and RMSE | Small dataset requires max training data per fold; LOOCV gives unbiased estimate without stratification assumptions | LOOCV exhaustively tests every sample as holdout — no information leakage, no sampling variance from fold splits |
| **M2: Input Configs** | Three configs: A (OES only), B (discharge only), C (combined) | Controlled-variable design isolates whether OES contributes beyond discharge parameters alone | By comparing A vs B vs C, we can attribute performance changes to specific data sources rather than confounding them |
| **M3: Baseline (PCA)** | 7 models × 3 configs; OES reduced from 701 → 11 PCA components (≥95% variance) | Establishes performance floor with standard dimensionality reduction; reveals that PCA-based OES features are uninformative | PCA captures maximum variance directions — but these directions may not align with the prediction target, especially on small data |
| **M4: Optuna HPO** | Bayesian optimisation (TPE sampler, 100+ trials) for RF, MLP, CNN | Tests whether model complexity can compensate for poor features; establishes that tuning alone is insufficient | TPE efficiently explores hyperparameter space; if tuned models still underperform Config B, the bottleneck is features, not models |
| **M5: Domain Features** | 13 features: 7 emission lines + 3 band integrals + 3 spectroscopic ratios, replacing 11 PCA components | PCA discards physical meaning; domain features encode species-specific information from plasma chemistry | Each feature maps to a known plasma species or reaction pathway; ratios are drift-invariant; band integrals are noise-robust — properties PCA cannot discover from small data |
| **M6: Interpretability** | Multi-model consensus importance → bootstrap CI → permutation test → backward elimination + category ablation | Validates that predictions are genuine (not overfitting); identifies minimal deployable feature set | Consensus across 4 model types reduces method-specific bias; permutation test provides frequentist p-value; ablation isolates individual feature-category contributions |

---

## 2. Backward Reasoning (先反向思考)

| Question | Answer |
|---|---|
| **What must the reader understand after reading Methodology?** | The complete experimental pipeline: dataset → 3 input configurations → 4 phases (baseline → tuning → domain features → interpretability) → evaluation protocol. Enough detail to reproduce every experiment. |
| **What is the narrative arc?** | NOT "we tried A, then B, then C." Instead: **systematic hypothesis-driven investigation.** Each phase tests one hypothesis while holding others constant: Phase 1 tests PCA features; Phase 2 tests whether model tuning compensates; Phase 3 tests whether domain features are superior; Phase 4 validates and reduces. |
| **What makes our methodology novel vs. standard ML?** | (1) Domain-knowledge feature engineering replacing PCA — physically motivated, not data-driven; (2) Controlled-variable input configuration design (A/B/C); (3) Multi-model consensus feature importance (4 methods averaged); (4) Dual feature reduction (backward elimination + category ablation); (5) Rigorous statistical validation (bootstrap + permutation) |
| **What must be justified (not just described)?** | (a) Why LOOCV, not k-fold? → small dataset, max training data per fold; (b) Why 7 models? → span linear/non-linear/deep to test whether model complexity matters; (c) Why these 13 OES features? → each linked to specific plasma species relevant to H₂O₂ formation; (d) Why consensus importance? → no single method is unbiased; (e) Why both backward elimination AND category ablation? → different granularity of feature reduction |

---

## 3. Forward Story (正向写作逻辑)

| Step | Logic | Content summary | Key justification |
|:---:|---|---|---|
| **M0** | Overview | Task setting (OES-based H₂O₂ yield prediction); pipeline has 4 phases; point to pipeline figure (if any); subsection map | Sets reader's expectations for what follows |
| **M1** | Dataset | Gao et al. (XJTU) dataset: 701-point OES + 4 discharge params + H₂O₂ yield; LOOCV evaluation; R² and RMSE metrics | Justify LOOCV: small sample size makes k-fold unstable; LOOCV is exact leave-one-out without random splits |
| **M2** | Configs | Three configs as controlled-variable design: A (OES only), B (discharge only), C (combined) | Config B serves as a strong baseline — any OES-inclusive config must beat it to prove OES adds value |
| **M3** | Baseline | 7 models spanning linear (Ridge, PLS), non-linear (SVR, RF), and deep (MLP, CNN); PCA reduces OES from 701 → 11 components | Model diversity tests whether performance ceiling is limited by features or model class; XGBoost anomaly noted and excluded |
| **M4** | Tuning | Optuna (TPE, 100+ trials) tunes RF, MLP, CNN | If tuned models on Config C still trail Config B, feature quality — not model tuning — is the bottleneck |
| **M5** | Domain features | 13 physically motivated OES features replace PCA: 7 emission lines + 3 band integrals + 3 spectroscopic ratios | Central methodological contribution. Each feature traced to plasma chemistry; ratios encode relative species populations (drift-invariant); band integrals capture broadband emission (noise-robust) |
| **M6a** | Importance | Multi-model consensus: Ridge coeff, PLS VIP, RF permutation importance, MLP SHAP → averaged ranks | No single importance method is unbiased — averaging reduces artifacts |
| **M6b** | Validation | Bootstrap (500 iterations) for 95% CI; permutation test (2000 shuffles) for statistical significance | Bootstrap quantifies uncertainty without distributional assumptions; permutation test gives model-free p-value |
| **M6c** | Reduction | Backward elimination (remove least important OES feature iteratively) + category ablation (remove entire feature categories) | Two complementary strategies: backward elimination finds the optimal individual feature set; category ablation reveals which feature TYPE (lines vs bands vs ratios) matters most |

---

## 4. Anti-Pattern Warnings for Methodology Writing

### Anti-pattern 1: Chronological diary
> **DO NOT** write: "In Phase 1 we tried PCA features. They didn't work well. So in Phase 2 we tuned hyperparameters. That still wasn't good enough. Then in Phase 3 we tried domain features."
>
> **DO** write: "The experimental pipeline comprises four phases, each testing a specific hypothesis: (1) whether standard PCA-based OES features carry predictive information, (2) whether model complexity compensates for feature quality, (3) whether domain-knowledge features outperform automated feature extraction, and (4) which features are essential and which are redundant."

**Why:** The chronological framing implies the researcher had no plan and stumbled into the solution. The hypothesis-driven framing shows systematic experimental design.

### Anti-pattern 2: Method dump without justification
> **DO NOT** write: "We used Ridge regression, PLS, SVR, XGBoost, Random Forest, MLP, and CNN."
>
> **DO** write: "Seven regression models were selected to span three categories of inductive bias: linear models (Ridge, PLS) that test whether the feature-target relationship is approximately linear; kernel and ensemble methods (SVR, RF) that capture non-linear interactions; and deep learning models (MLP, CNN) that can learn hierarchical representations. This diversity ensures that performance differences are attributable to features, not model class."

**Why:** Listing models without rationale reads as "we threw everything at the wall." Explaining the design logic shows the reader that model selection was intentional.

### Anti-pattern 3: Results in Methodology
> **DO NOT** write in the Methodology section: "Ridge Config C yielded R² = −0.17 with PCA features."
>
> **DO** write: "All R² and RMSE results are reported in Section 3 (Findings)."

**Why:** Methodology describes WHAT was done and WHY. Numbers go in Results. Mixing them makes both sections harder to read and creates redundancy.

### Anti-pattern 4: Burying the contribution
> **DO NOT** treat Phase 3 (domain-knowledge feature engineering) as "just another step in the pipeline, equal to Phase 1 or Phase 2."
>
> **DO** signal it explicitly: give §2.2.5 more space (2–3 paragraphs vs 1 paragraph for other phases); open with a motivation paragraph explaining WHY domain features are expected to be superior; provide a detailed table of all 13 features with physical justification.

**Why:** The reader must understand that Phase 3 is the central methodological contribution, not a minor variation. Proportional space allocation signals importance.

### Anti-pattern 5: Unmotivated feature selection
> **DO NOT** write: "We selected 13 OES features based on plasma chemistry literature."
>
> **DO** write: "Each of the 13 features was selected because it corresponds to a specific plasma species or reaction pathway relevant to H₂O₂ formation. For example, the OH emission line at 309 nm tracks hydroxyl radical concentration — the primary precursor to H₂O₂ in plasma-liquid systems [Gao2024]. The Hα/Hβ ratio (656/486 nm) encodes the Balmer decrement, a classical diagnostic for electron temperature [Laux2003]. These features are not arbitrary: they are the same quantities that spectroscopists have used for decades."

**Why:** The physical justification is the entire argument for why domain features outperform PCA. Without it, the reader has no reason to believe the features weren't cherry-picked post-hoc.

---

## 5. Paragraph-Level Plan for Methodology Section

| Para # | Subsection | Role | First sentence message | Content | Citations | Tables/Figures |
|:---:|---|---|---|---|---|---|
| 1 | §2.2 Overview | Opening | This section describes the experimental pipeline... | Task setting (OES → H₂O₂ yield); 4-phase structure; subsection map (§2.2.1–2.2.6) | — | `fig:pipeline` (optional) |
| 2 | §2.2.1 | Opening + Detail | The dataset was provided by Gao et al... | Dataset source, structure (701-point OES, 4 discharge params, H₂O₂ target), sample count | [Gao2024] | — |
| 3 | §2.2.1 | Justification | Model performance was evaluated using LOOCV... | Why LOOCV (small dataset, unbiased, no fold variance); R² and RMSE definitions; data integrity (no leakage) | — | — |
| 4 | §2.2.2 | Detail + Justification | Three input configurations were defined... | Config A/B/C as controlled-variable design; Config B as discharge-only baseline; purpose of isolating OES contribution | — | `tab:configs` |
| 5 | §2.2.3 | Opening + Detail | Seven regression models spanning three categories... | Model selection rationale (linear / non-linear / deep); PCA reduction (701 → 11); brief note on XGBoost anomaly and exclusion | — | — |
| 6 | §2.2.4 | Detail + Justification | Non-linear models were tuned using Optuna... | Optuna TPE sampler; 100+ trials; purpose: test if tuning compensates for feature quality | [optuna2019] | — |
| 7 | §2.2.5 | Motivation | The preceding phases relied on PCA for OES dimensionality reduction... | Motivation: PCA discards physical meaning; traditional spectroscopy uses line ratios and band integrals for robust diagnostics; modern ML overlooks this domain knowledge | [Laux2003], [Paris2005] | — |
| 8 | §2.2.5 | Detail (core) | Thirteen physically motivated OES features were derived... | Enumerate: 7 emission lines (species correspondence); 3 band integrals (wavelength ranges, robustness); 3 spectroscopic ratios (physical meaning) | [Gao2024], [Laux2003] | `tab:oes_features` |
| 9 | §2.2.5 | Advantage + Transition | These domain-knowledge features offer three technical advantages... | (1) Drift-invariant (ratios normalize); (2) physically interpretable (traceable to species); (3) dramatically lower dimensionality (13 vs 701/11). Transition: "Section 2.2.6 evaluates which of these features are essential." | [Wang2025] | — |
| 10 | §2.2.6 | Opening + Detail | Feature importance was quantified using four model-specific methods... | Ridge coeff, PLS VIP, RF permutation, MLP SHAP → consensus ranking by averaged ranks | — | — |
| 11 | §2.2.6 | Detail | Statistical validation comprised two complementary tests... | Bootstrap (500 iterations, 95% CI) for uncertainty quantification; permutation test (2000 shuffles) for significance | — | — |
| 12 | §2.2.6 | Detail + Justification | Feature reduction was performed via two complementary strategies... | Backward elimination (iterative removal) + category ablation (remove entire feature types: lines/bands/ratios). Justify: backward finds optimal individual set; ablation reveals which TYPE matters | — | — |

**Total: 12 paragraphs** (2 for dataset/evaluation, 1 for configs, 1 for baseline, 1 for tuning, 3 for domain features, 3 for interpretability, 1 overview)

---

## 6. Figures and Tables Plan

| Label | Type | Content | Location | Essential? |
|---|---|---|---|---|
| `fig:pipeline` | Figure | Overall pipeline diagram: Data → 3 Configs → Phase 1–4 → Results | §2.2 Overview (Para 1) | **Optional** — useful but not strictly necessary for a BEng report; the subsection map in the overview paragraph can substitute |
| `tab:configs` | Table | Input configuration summary: Config A/B/C, feature composition, purpose | §2.2.2 (Para 4) | **Essential** — already exists in `main.tex`; needs updating for Phase 3 (Config A: 13 domain features, Config C: 17 features) |
| `tab:oes_features` | Table | All 13 OES features: feature name, type (line/band/ratio), wavelength(s), corresponding plasma species, physical meaning, literature reference | §2.2.5 (Para 8) | **Essential** — this is the central methodological table; without it, the reader cannot assess feature selection quality. Could also go in Appendix C with a summary in-text. |
| `tab:models` | Table | 7 regression models with category (linear/non-linear/deep), brief description, key hyperparameters tuned | §2.2.3 (Para 5) | **Optional** — could be a compact table or just described in text. Useful if space permits. |

### Table: `tab:oes_features` (Draft Structure)

| Feature | Type | Wavelength (nm) | Plasma Species | Physical Significance |
|---------|------|:---:|----------------|----------------------|
| I_309_OH | Line | 309 | OH radical | Primary H₂O₂ precursor; OH + OH → H₂O₂ |
| I_777_O | Line | 777 | Atomic oxygen (O) | Oxidative species; participates in OH formation |
| I_486_Hb | Line | 486 | Hydrogen (Hβ) | Balmer series; electron impact excitation marker |
| I_656_Ha | Line | 656 | Hydrogen (Hα) | Balmer series; most intense H line |
| I_337_N2 | Line | 337 | Molecular nitrogen (N₂) | Second positive system; gas temperature indicator |
| I_406_CO2p | Line | 406 | CO₂⁺ ion | CO₂ dissociation product; primary reactant marker |
| I_516_C2 | Line | 516 | C₂ (Swan band) | Carbon recombination; deep dissociation indicator |
| band_OH_306_312 | Band | 306–312 | OH (A²Σ⁺→X²Π) | Integrated OH emission; robust to spectral noise |
| band_CO2p_398_412 | Band | 398–412 | CO₂⁺ (Fox–Duffendack–Barker) | Integrated CO₂⁺ emission; reactant consumption marker |
| band_CO_Hb_460_500 | Band | 460–500 | CO + Hβ overlap | Mixed CO/H emission; product formation region |
| ratio_309_656 | Ratio | 309/656 | OH/Hα | Hydrogen abstraction pathway balance |
| ratio_656_486 | Ratio | 656/486 | Hα/Hβ | Balmer decrement; classical Te diagnostic |
| ratio_777_309 | Ratio | 777/309 | O/OH | Oxidation vs hydroxylation pathway balance |

---

## 7. Methodology Writing Checklist

Before writing prose, verify:

- [ ] Every subsection opens with motivation (WHY) before design (WHAT)
- [ ] Phase 3 (§2.2.5) gets proportionally more space (3 paragraphs) than other phases (1 paragraph each)
- [ ] `tab:oes_features` includes physical justification for every feature — no feature is listed without a species/pathway explanation
- [ ] No R² values or performance numbers appear in the Methodology section — all go in §2.3
- [ ] XGBoost anomaly is briefly noted and explained in §2.2.3
- [ ] LOOCV is justified, not just described (small dataset + no leakage)
- [ ] The 4-phase structure is framed as hypothesis-driven, not chronological
- [ ] The overview paragraph provides a subsection map (§2.2.1 covers..., §2.2.2 covers..., etc.)
- [ ] `tab:configs` is updated to reflect Phase 3 feature counts (Config A: 13, Config C: 17)
- [ ] Citations are used: [Gao2024] for dataset, [optuna2019] for Optuna, [Laux2003] and [Paris2005] for OES diagnostics, [Wang2025] for line-ratio ML validation
