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