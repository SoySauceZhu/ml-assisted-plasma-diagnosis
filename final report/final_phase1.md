# Plan

## Background

I'm going to write the final report for my FYP project (BEng final report).
The report requirement and marking description is shown in 'final report' folder.
Follow the official guidance (final report/Final report Guidance 2026.pdf), the suggestion and the instruction below, help me write the paper.

## Task
The task in first phase is articulate the outline of the report article.

## Instruction
By a famous suggestion for paper writing, the first step of a paper is as followed:
> 梳理论文的story，写一个Introduction的写作思路，并整理comparison experiments和ablation studies的过程和结果。

Please generate a detailed "outline.md" to write the outline, in which you should include:
- The overall logical story of the paper (总体行文逻辑)
- structure of each sections/part. Use bullet points
- content of each section/part

Then generate a brief "experiments.md" to summarize the experiments and method.
- You can choose to use or not use the chronological, four-phase, structure or reorganize these experiments logically.
- Write intro/description for each experiment and show the result in brief words.

## Notice

Even though then final report guidance indicates that this report should be a engineering technical report, but my project is a pure research project.
You have discretion to choose if to write the final paper with "research-style content and technical-report-style format". 

# Action

## Prerequisites

Before starting, read and understand:
1. `final report/Final report Guidance 2026.pdf` — official report requirements and format
2. `final report/marking descriptors Final Report_2026.pdf` — marking criteria
3. `final report/report/main.tex` — the existing LaTeX draft (contains the current report structure, data, and results)
4. Load the `research-paper-writing` skill for section-level writing guidance

## Step 1: Generate `outline.md`

Create `final report/final_phase1/outline.md` with the following structure:

### 1.1 Overall Story (总体行文逻辑)

Write a concise narrative arc (5–8 sentences) that captures the logical story of the paper:
- **Problem**: Real-time H₂O₂ yield prediction from plasma OES is needed but offline methods are slow and intrusive.
- **Existing gap**: Prior OES+ML approaches use PCA/PLS on raw spectra, discarding physical meaning → poor performance on small datasets.
- **Key insight**: Domain-knowledge-driven feature engineering (selecting emission lines, band integrals, spectroscopic ratios from plasma chemistry literature) dramatically improves prediction.
- **Evidence**: Ridge Config C R² jumps from −0.17 (PCA) to 0.80 (domain features); a minimal 7-feature Ridge model achieves R² = 0.92.
- **Implication**: For physically structured scientific datasets, simple interpretable models with domain-informed features outperform complex models with automated feature extraction.

### 1.2 Section-by-Section Outline

For **each** section below, provide:
- Section title and label
- 3–7 bullet points describing content
- The role of each paragraph (opening / challenge / method / advantage / evidence / limitation)
- Key figures/tables to include (reference existing ones in `main.tex` where applicable)

Sections to cover:
1. **Introduction** (Background & Motivation → Objectives → State of the Art)
2. **Procedure / Methodology** (Dataset → Input Configs → Phase 1–4 methods)
3. **Findings / Results** (Phase 1&2 baseline → Phase 3 breakthrough → Phase 4 interpretability & minimal model)
4. **Conclusions** (Objectives met → broader implication → limitations)
5. **Recommendations / Future Work**
6. **Reflection**
7. **Summary / Executive Summary** (≤200 words)
8. **Appendices** (Code, Dataset, OES features, Optuna results, Statistical tests)

### 1.3 Key Arguments and Claim–Evidence Mapping

For each major claim in the paper, list:
- The claim statement
- The supporting evidence (experiment phase, metric, table/figure)
- Status: supported / needs more evidence

Focus especially on claims in the Introduction and Conclusions.

## Step 2: Generate `experiments.md`

Create `final report/final_phase1/experiments.md` with the following structure:

### 2.1 Experimental Overview

Provide a brief paragraph summarizing the overall experimental methodology:
- Dataset: 701 OES samples, 4 discharge parameters, H₂O₂ yield target
- Evaluation: LOOCV with R² and RMSE
- 7 models × 3 input configurations × 4 iterative phases

### 2.2 Experiment Descriptions

For each experiment/phase, write:
- **Title** and purpose (1–2 sentences)
- **Method** description (what was done, which models, which features)
- **Key results** in brief (quantitative: R², RMSE values)
- **Takeaway** (what this phase proved or disproved)

Organize as follows (you may reorganize logically rather than chronologically if it better serves the story):

1. **Baseline Modelling (Phase 1)**: PCA-reduced OES + discharge params → 7 models. Key result: Config B (discharge only) R² ≈ 0.90; Config C (OES+discharge via PCA) R² = −0.17 for Ridge. PCA-based OES features are insufficient.

2. **Hyperparameter Optimisation (Phase 2)**: Optuna TPE tuning for RF, MLP, CNN. Key result: MLP Config C improved from −1.13 to 0.33; CNN Config C best OES model but still below Config B. Upper-bound established: tuning alone cannot fix bad features.

3. **Domain-Knowledge Feature Engineering (Phase 3)**: Replace PCA with 13 physically motivated OES features (7 emission lines + 3 band integrals + 3 spectroscopic ratios). Key result: Ridge Config C R² from −0.17 → 0.80; MLP Config C from 0.33 → 0.80. Central finding of the project.

4. **Interpretability & Feature Reduction (Phase 4)**: Feature importance ranking, bootstrap CI, permutation test, category ablation, backward elimination. Key result: 7-feature Ridge model R² = 0.92; permutation test p < 0.00005. Minimal interpretable model matches neural networks.

### 2.3 Comparison Experiments Summary Table

Create a summary table with columns: Phase | Model | Config | R² | RMSE | Key Change

### 2.4 Ablation Studies

Summarize:
- **Category ablation**: removing emission lines / band integrals / ratios one group at a time
- **Backward elimination**: iterative feature removal to find minimal set
- Results: which features are essential, which are redundant

## Output Format

- Write both files in Markdown
- Use clear headings and bullet points
- Include quantitative results wherever possible (exact R², RMSE, p-values from `main.tex`)
- Keep language concise and technical
- Reference specific tables/figures from `main.tex` by their labels (e.g., Table~\ref{tab:bootstrap})

# Observation

## Phase 1 Task Completion Evaluation

### Task Requirement Checklist

| Requirement | Status | Notes |
|---|:---:|---|
| Overall logical story (总体行文逻辑) | Done | 7-sentence narrative arc covering problem → gap → insight → evidence → implication |
| Section-by-section outline with bullet points | Done | 8 sections, each with subsection-level bullets |
| Paragraph roles annotated (opening/challenge/method/evidence/limitation) | Done | All subsections tagged with paragraph roles |
| Content of each section/part | Done | Detailed content descriptions with quantitative references |
| `experiments.md` with experiment descriptions | Done | 4 phases, each with purpose/method/results/takeaway |
| Comparison experiments summary table | Done | 12-row table covering key results across all phases |
| Ablation studies summary | Done | Category ablation + backward elimination + optimal reduced model |
| Quantitative results from actual data | Done | All R², RMSE, p-values pulled from CSV files, consistent with `main.tex` |

**Verdict: All instruction requirements met.**

---

### Adversarial Review (research-paper-writing skill — 5-Dimension Self-Review)

#### 1. Contribution Clarity — Pass with notes

**Strength:** The paper story has a clear, non-trivial insight — domain-knowledge features transform OES-based prediction from failure (R² = −0.17) to success (R² = 0.80/0.92). This is a genuinely surprising result, not a predictable incremental gain.

**Risk to address in Phase 2 writing:**
- The outline currently reads as "incremental patching of a naive baseline" (Phase 1 → Phase 2 → Phase 3 → Phase 4). The Introduction guide explicitly warns: *"Do not first present a naive solution and then describe our improvement over it."* The story should be reframed from "we tried PCA and it failed, so we tried domain features" to "domain knowledge is the decisive factor — here is why and the evidence."
- **Recommendation:** In Introduction, use **Technical-Challenge Version 2** (existing task + our insight seen in traditional methods) — frame domain-knowledge feature engineering as a classical spectroscopy principle that modern ML approaches have overlooked, not as "our fix after PCA failed."

#### 2. Writing Clarity — Pass with gaps

**Strength:** The outline provides clear paragraph roles and the story flows logically from problem → evidence → conclusion.

**Gaps identified:**
- **State of the Art (§2.1.3)** in `outline.md` only says "Review 3–5 papers" with no specifics. The `main.tex` also has `\todo{Briefly review 3–5 papers}`. This is the weakest section — without concrete prior work citations, the "gap" claim is unsupported. **Priority for Phase 2: literature review is essential before writing.**
- **Terminology note:** The outline uses "Config A/B/C" consistently, which is good. But the Conclusion section in `main.tex` switches between "7-feature" and "minimal" model — needs standardization.
- `references.bib` is almost entirely placeholder (`TODO`). The paper cannot proceed to prose writing without real citations.

#### 3. Experimental Strength — Strong

**Strengths:**
- The R² improvement from −0.17 → 0.80 (and to 0.92 after reduction) is not marginal — it is a qualitative regime change.
- Results are consistent across multiple models (Ridge, PLS, MLP all improve with domain features).
- Permutation test (p < 0.0005) and bootstrap CIs provide rigorous statistical validation.
- The backward elimination trajectory (R² monotonically increasing as features are removed) is a compelling and unusual result that strengthens the "redundancy" narrative.

**Weakness:**
- Single dataset from one lab. Acknowledged as limitation but cannot be resolved within this project.
- XGBoost results are anomalous (identical R² = −0.108 across all configs) — likely a bug or data issue. The outline and experiments.md correctly omit it from later phases, but this should be briefly acknowledged in the report (not silently dropped).

#### 4. Evaluation Completeness — Pass

**Strengths:**
- Ablation studies are thorough: both category ablation (3 groups) and backward elimination (full 13→0 trajectory).
- Multiple evaluation dimensions: R², RMSE, MAE, bootstrap CI, permutation test, feature importance consensus, VIF analysis.
- Claim–Evidence Mapping table (9 claims, 8 supported, 1 explicitly marked as "needs evidence") demonstrates disciplined claim management.

**Minor gap:**
- `experiments.md` §4.3 mentions the permutation-tested 7-feature model uses 3 ratios, but the backward elimination trajectory shows the optimal point is actually at **1 OES feature** (band_CO2p_398_412, R² = 0.918). The 7-feature model (R² = 0.920) comes from category ablation (3 ratios + 4 discharge). Both are valid "optimal" models from different selection methods — this distinction should be made explicit in the report to avoid reviewer confusion.

#### 5. Method Design Soundness — Pass

**Strengths:**
- LOOCV is appropriate for n=701 (actually n=number of experimental conditions, which appears small).
- The four-phase iterative structure is a sound experimental design that progressively isolates variables.
- Feature selection is grounded in plasma chemistry literature, not arbitrary.

**Risk:**
- The outline does not explicitly discuss potential **data leakage** or **target leakage** risks. For a BEng report this may not be critical, but for research credibility, a brief statement confirming that LOOCV prevents leakage would strengthen the Methodology section.

---

### Story Coherence Assessment (Introduction Logic Map)

Mapping the outline against the Introduction guide's logic:

| Logic Step | Content in Outline | Completeness |
|---|---|:---:|
| L1: What task | H₂O₂ yield prediction from OES | Clear |
| L2: Target metrics | R², RMSE | Clear |
| L3: SOTA fails | PCA/PLS on raw spectra → poor performance | Stated but **no concrete citations** |
| L4: Root technical issue | PCA discards physical meaning of emission lines | Clear |
| L5: Our solution | Domain-knowledge feature engineering (13 features) | Clear |
| L6: Why it works | Features grounded in plasma chemistry; ratios/bands are drift-robust | Clear |
| L7: Additional contributions | Minimal 7-feature model; statistical validation framework | Clear |

**Introduction template recommendation:** Use **Version 3** (general to specific setting) for Part A — start from "plasma diagnostics is important" then narrow to "OES-based H₂O₂ prediction." Combined with **Technical-Challenge Version 2** for Part B — domain knowledge has roots in traditional spectroscopy but modern ML approaches overlook it.

---

### Summary of Identified Risks for Phase 2

| Priority | Risk | Action |
|:---:|---|---|
| **High** | No real literature citations (`references.bib` is all TODO) | Complete literature review before writing prose |
| **High** | Story reads as incremental patching; needs reframing | Restructure Introduction to lead with insight, not with failure |
| Medium | XGBoost anomaly unexplained | Add brief note in Methodology or Results |
| Medium | Two "optimal" models (1-feature vs 3-ratio) need disambiguation | Clarify in Phase 4 Results subsection |
| Low | Data leakage / LOOCV validity not explicitly stated | Add 1–2 sentences in Methodology |
| Low | Summary in `main.tex` exceeds 200 words | Edit down before submission |

---

### Overall Assessment

Phase 1 deliverables (`outline.md` and `experiments.md`) are **complete and data-grounded**. All quantitative values are verified against the actual experimental CSVs. The claim–evidence mapping is disciplined with 8/9 claims fully supported. The outline provides sufficient structure to proceed to prose writing.

The most critical preparation for Phase 2 is **completing the literature review** and **reframing the Introduction** to avoid the "incremental patching" anti-pattern identified by the research-paper-writing skill. The story should open with the insight (domain knowledge matters) rather than the failure (PCA doesn't work).