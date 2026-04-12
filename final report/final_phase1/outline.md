# Final Report Outline
## Machine Learning Assisted Real-Time Plasma Diagnosis: Domain-Knowledge-Driven Feature Engineering Enables H2O2 Yield Prediction

---

## Overall Logical Progression (总体行文逻辑)

The report follows a problem-driven narrative: starting from an industrial need (real-time plasma product monitoring), establishing that existing data-driven approaches are insufficient, then showing step-by-step how domain knowledge transforms model performance, and concluding with actionable insights and forward-looking recommendations.

```
Industrial Need
    → Problem Framing (why OES + ML is promising but hard)
        → Methodology (4-phase iterative pipeline)
            → Results (progressive improvement, key discovery in Phase 3)
                → Conclusions (domain knowledge > data-driven automation)
                    → Recommendations (generalisation, deployment, future research)
                        → Reflection (professional development)
```

---

## Report Structure

### Front Matter (not in word count)
- Title Page (use Canvas template)
- Acknowledgements (optional)
- Summary / Executive Summary (≤200 words, text only)
- Table of Contents, List of Figures, List of Tables, List of Abbreviations

---

### 1. Introduction / Scope (~600 words)

**Purpose:** Set context, define objectives, position work relative to state of the art.

**Content:**
- **Background:** Nanosecond pulsed CO2 bubble plasma discharge as a route to green chemistry (CO2 → H2O2). Importance for climate change mitigation and plasma-based water treatment.
- **Problem Statement:** H2O2 yield currently requires offline titration — slow, disruptive, not suitable for real-time control. Optical Emission Spectroscopy (OES) offers non-intrusive in-situ diagnostics but its high dimensionality (701 wavelength points) makes direct ML use challenging.
- **Objectives:**
  1. Build ML models to predict H2O2 yield from OES and discharge parameters.
  2. Investigate whether domain-knowledge feature engineering outperforms automated dimensionality reduction (PCA).
  3. Identify the minimal, most informative feature set for a deployable real-time predictor.
- **State of the Art:** Briefly review existing OES-based ML studies; note that prior work uses generic PCA without exploiting plasma chemistry knowledge. Distinguish this work's contribution.
- **Scope:** Dataset from XJTU (Gao et al.), 701 samples, supervised regression task, software/ML project.

---

### 2. Procedure / Methodology (~900 words)

**Purpose:** Describe the 4-phase iterative pipeline clearly enough to allow reproduction.

**Content:**

#### 2.1 Dataset and Experimental Setup
- Source: XJTU open dataset — nanosecond pulsed CO2 bubble discharge experiments.
- Input features: 701-point OES spectrum + 4 discharge parameters (voltage, current, frequency, pulse width / rise time / flow rate).
- Target: H2O2 yield (continuous regression target).
- Train/test split strategy: Leave-one-out cross-validation (LOOCV) due to small sample size.

#### 2.2 Input Configurations
| Config | Features Used |
|:------:|---------------|
| A | OES (PCA-reduced, 11 components) only |
| B | 4 discharge parameters only |
| C | OES + discharge parameters combined |

#### 2.3 Phase 1 — Baseline Modelling
- Models: Ridge, PLS, SVR, XGBoost, Random Forest, MLP, 1D-CNN.
- OES dimensionality reduction: PCA (11 components, ~95% variance retained).
- Evaluation metric: R² and RMSE via LOOCV.

#### 2.4 Phase 2 — Hyperparameter Optimisation
- Framework: Optuna (Bayesian optimisation / TPE sampler).
- Scope: Non-linear models (RF, MLP, CNN), >100 trials per model–config pair.
- Goal: Establish fair performance ceiling before feature engineering.

#### 2.5 Phase 3 — Domain-Knowledge Feature Engineering
- Motivation: PCA components are statistically optimal but physically opaque.
- Approach: Manual selection of 13 physically meaningful OES features based on plasma chemistry literature:
  - Key emission lines: OH (309 nm), O (777 nm), Hβ (486 nm), Hα (656 nm), N₂ (337 nm), CO₂⁺ (406 nm), C₂ (516 nm).
  - Band integrals: OH band (306–312 nm), CO₂⁺ band (398–412 nm), CO/Hβ band (460–500 nm) — robust to spectral drift.
  - Ratio features: OH/Hα (309/656), Hα/Hβ (656/486), O/OH (777/309) — physically interpretable diagnostics.
- Combined with 4 discharge parameters → 17-feature Config C.

#### 2.6 Phase 4 — Interpretability and Feature Reduction
- Feature importance: Absolute standardised coefficients (Ridge/PLS), permutation importance (RF), gradient-based attribution (MLP).
- Statistical validation: Bootstrap resampling (500 iterations) for R² confidence intervals; permutation test for model significance (p < 0.00005).
- Feature reduction: Backward elimination + category ablation to find minimal feature set.

---

### 3. Findings / Results (~900 words)

**Purpose:** Present key results with evidence; compare phases; demonstrate contribution.

**Content:**

#### 3.1 Phase 1 & 2 Results — Baseline and Tuning
- Config B (discharge only) achieved R² ≈ 0.90 with linear models — strong baseline.
- Config A/C (OES via PCA) performed poorly: Ridge Config C R² = −0.17, MLP Config C R² = −1.13 before tuning.
- After Optuna tuning: MLP Config C improved to R² = 0.33; CNN Config C was best OES-only model but still substantially below Config B.
- Key insight: PCA features fail to transfer useful OES information.

#### 3.2 Phase 3 Results — Domain-Knowledge Breakthrough
- **Headline result:** Ridge Config C R² jumped from −0.17 → 0.80 using 13 hand-crafted OES features.
- MLP Config C: 0.33 → 0.80. Similar trend across all models.
- Domain-knowledge features closed the gap between Config B and Config C — OES now contributes meaningfully.
- Figure: R² comparison chart across all phases, models, and configs.

#### 3.3 Phase 4 Results — Interpretability and Minimal Model
- Feature importance consensus (Table): CO₂ flow rate and CO₂⁺ band integral (398–412 nm) are top predictors across all models — physically coherent with CO₂ being the main reactant.
- Bootstrap 95% CI: Ridge Config B [0.800, 0.955], Ridge Config C [0.574, 0.910] — overlap confirms no statistically significant difference between linear and neural network models.
- Permutation test: p < 0.00005 — model captures a genuine input–output relationship.
- **Best minimal model:** Ridge regression with 3 OES ratio features + 4 discharge parameters (7 features total), R² = 0.920.
- A simple linear model matches MLP and CNN performance with far less complexity.
- Figure: Feature importance heatmap; R² vs. feature count (ablation curve).

#### 3.4 Comparison with Prior Work
- Previous OES-based ML studies used PCA without physical grounding → this work demonstrates the decisive advantage of domain-knowledge features.
- Achieving R² = 0.92 with 7 features is state-of-the-art for this dataset and task.

---

### 4. Conclusions (~400 words)

**Purpose:** Synthesise findings; evaluate objective achievement; contextualise contribution.

**Content:**
- **Objective 1 achieved:** ML models successfully predict H2O2 yield; best model R² = 0.92.
- **Objective 2 achieved:** Domain-knowledge feature engineering decisively outperforms PCA (R² −0.17 → 0.80 for Ridge Config C).
- **Objective 3 achieved:** Minimal 7-feature Ridge model matches complex neural networks — suitable for real-time deployment.
- **Novel contribution:** First systematic demonstration that plasma-chemistry-informed feature selection transforms OES-based yield prediction from impractical to high-performing.
- **Broader context:** Simple, interpretable models can match deep learning when features are physically grounded — a general lesson for small-dataset scientific ML.
- **Limitations:** Small dataset (701 samples from one lab); generalisability to other plasma systems untested.

---

### 5. Recommendations / Future Work (~300 words)

**Purpose:** Look forward technically; show engineering vision.

**Content:**
- **Generalisation:** Test domain-knowledge feature engineering on datasets from different plasma systems and geometries to assess transferability.
- **Real-time deployment:** Integrate the 7-feature Ridge model with live OES hardware for closed-loop discharge control; evaluate latency and robustness.
- **Larger datasets:** Collect more samples across wider operating ranges to test non-linear model benefits.
- **Physics-informed ML:** Incorporate mass conservation constraints (three-phase balance) into multi-output models predicting multiple plasma products simultaneously.
- **Extended feature set:** Explore time-resolved OES features and rotational/vibrational temperature estimates as additional physically grounded inputs.
- **Transfer learning:** Use models trained on this dataset to initialise models for related plasma reactions (e.g., N₂O, ozone generation).

---

### 6. Reflection (~200 words)

**Purpose:** Professional development narrative.

**Content:**
- **Technical skills developed:** ML pipeline design, hyperparameter optimisation with Optuna, statistical validation (bootstrap, permutation testing), plasma spectroscopy literature review, Python-based data science.
- **Professional skills developed:** Independent research planning across 4 iterative phases, scientific communication (bench inspection poster & presentation), technical report writing to IET standards.
- **Comparison with original skills audit:** Started with ML fundamentals but limited domain knowledge; developed ability to bridge physics and data science — a transferable cross-disciplinary skill.
- **Preparation for industry/further study:** The project demonstrates the value of domain expertise in applied ML — directly relevant to engineering roles in process control, sensor fusion, and industrial AI.

---

### Back Matter (not in word count)
- **References** (IEEE format — journals, conference papers, online datasets, software)
- **Appendices:**
  - A: Full source code (Python ML pipeline)
  - B: Dataset description and feature definitions
  - C: Detailed hyperparameter search results (Optuna trials)
  - D: Full statistical test outputs (bootstrap distributions, permutation test)
  - E: Complete OES feature derivation (emission line assignments with literature sources)

---

## Word Count Allocation Summary

| Section | Target Words |
|---------|:------------:|
| Introduction / Scope | ~600 |
| Methodology | ~900 |
| Findings / Results | ~900 |
| Conclusions | ~400 |
| Recommendations | ~300 |
| Reflection | ~200 |
| **Total (main body)** | **~3,300** |

*Buffer of ~700 words retained for figures, tables, and captions within the 4,000-word limit.*
