# Plan

## Background

Phase 1,2,3 has generated the writing logic flow of Introduction and Methodology section of my final year report. Also, a refined outline is generated.

In phase 4, we are gonna write the first two sections: Intro and Method.

## Tasks

- Acquire enough context about previous phases's instruction markdown, final report guidance and marking descriptor, Introduction and Methodology writing logic flow, facts in my experiments, reference/citation in bibtex, and look at corresponding sections in the outline.
- Write Introduction section and Methodology section for my final report. Write in 'final\ report/report/main.tex'. 


## Tips
- As mentioned in phase1 instruction markdown, the report guidance asks for a technical report, but my work is a research project. So our content/writing logic should be like research paper style, but we should also meet the requirements of format or something detail.
- Use IEEE citation style
- As mentioned in phase3's observation. No citation is listed in bibtex about how to choose the 13 features manually. You can put citations aside in this phase. Just complete the sections following the writing flow instruction. I'll add some references manually later.
- You can look at 'phase3_feature_engineering.md' to see how to justify the use of each features.


# Action

## Prerequisites

Before executing, the agent must read and internalise the following context files (in this order):

1. **Phase 3 outline (latest version)** → `final report/final_phase3/outline.md`
   - Contains: the most refined outline with §2.1 Introduction and §2.2 Methodology fully structured, paragraph-role annotations, justification bullets
   - Purpose: **primary blueprint** — every paragraph to be written follows this outline

2. **Introduction writing logic flow** → `final report/final_phase2/introduction_flow.md`
   - Contains: Literature Map, Backward/Forward Logic Flow (L1–L10), Anti-Pattern Warnings, Paragraph-Level Plan (8 paragraphs)
   - Purpose: governs the narrative arc, paragraph sequencing, and anti-pattern guardrails for §1 Introduction

3. **Methodology writing logic flow** → `final report/final_phase3/methodology_flow.md`
   - Contains: Pre-Writing Questions, Backward Reasoning, Forward Story (M0–M6c), 5 Anti-Pattern Warnings, Paragraph-Level Plan (12 paragraphs), Figures/Tables Plan
   - Purpose: governs the narrative arc, paragraph sequencing, and anti-pattern guardrails for §2 Methodology

4. **Experiments summary** → `final report/final_phase1/experiments.md` or result folders of each experiment phases (i.e. `phase4/results/tables`)
   - Contains: all 4 phases' experiment descriptions, quantitative result tables, comparison summary, ablation studies
   - Purpose: source of facts — verify any claim before writing; but remember NO results (R² values) go in Methodology

5. **Feature engineering justification** → `phase3_feature_engineering.md`
   - Contains: rationale for each of the 13 OES features, species correspondence, physical meaning
   - Purpose: detailed justification for §2.2.5 (Domain-Knowledge Feature Engineering)

6. **References** → `final report/report/references.bib`
   - Purpose: available citation keys; use IEEE-style `\cite{key}` commands
   - Note (from Plan Tips): some feature justification citations are missing — leave `\cite{TODO}` placeholders where needed; the user will add references manually later

7. **Current main.tex** → `final report/report/main.tex`
   - Purpose: understand existing LaTeX structure, what's already drafted, what needs replacing
   - The existing draft sections (§1 and §2) are **placeholder/skeleton text** — they will be overwritten

8. **Report guidance & marking descriptors** (PDFs):
   - `final report/Final report Guidance 2026.pdf`
   - `final report/marking descriptors Final Report_2026.pdf`
   - Purpose: understand format requirements, word count expectations, marking criteria
   - Note (from Plan Tips): content is research paper style but must meet technical report format requirements

9. **Phase 2 observation (risks)** → `final report/final_phase2.md` (Observation section)
   - Purpose: check for any risks relevant to prose writing (e.g., Stefas2025 is a preprint)

10. **Phase 3 observation (risks)** → `final report/final_phase3.md` (Observation section)
    - Purpose: check adversarial notes about feature justification depth, results boundary, methodology length

11. **Skill available:**
    - research-paper-writing: `.claude/skills/research-paper-writing`
    - Purpose: consult for writing quality, paragraph flow, and anti-pattern avoidance

---

## Step 1: Write Introduction Section (§1)

Rewrite `\section{Introduction}` in `main.tex` (lines 155–178), replacing the existing placeholder/skeleton with polished prose. Follow the introduction_flow.md paragraph plan and the Phase 3 outline §2.1.

### Structure and paragraph mapping:

| Para # | Subsection | Role | Content (from introduction_flow.md) |
|:---:|---|---|---|
| 1 | §1.1 Background | Opening | Plasma-based green chemistry: nanosecond pulsed CO₂ bubble discharge; growing demand for real-time monitoring |
| 2 | §1.1 Background | Challenge | Conventional H₂O₂ quantification (offline titration) vs. OES as non-intrusive alternative; 701-point dimensionality challenge |
| 3 | §1.2 Objectives | Structure | Three objectives stated clearly (predict yield, compare PCA vs domain features, find minimal feature set) |
| 4 | §1.3 Literature | Context | OES as established plasma diagnostic; traditional spectroscopic methods (Boltzmann plot, line-ratio); ML as data-driven alternative |
| 5 | §1.3 Literature | Gap/Challenge | Existing OES+ML approaches (Srikar, Stefas, Wang2019, Park) all use PCA; limitations of PCA-based approaches |
| 6 | §1.3 Literature | Contrast | Wang2025 uses physically motivated LIRs → R² ≈ 0.90–0.97; demonstrates domain-knowledge value but only for Te/ne, not yield |
| 7 | §1.3 Literature | Positioning | Gap: no systematic PCA vs domain-knowledge comparison for yield prediction; traditional spectroscopy knowledge exists but is overlooked; this project bridges the gap |

### Writing rules for Introduction:

1. **Follow the outline §2.1 and introduction_flow.md strictly** — do not invent new structure
2. **Use IEEE citation style**: `\cite{key}` with keys from references.bib
3. **Do NOT report our own results** in the Introduction — only cite literature results (e.g., Wang2025's R² ≈ 0.90–0.97 is acceptable as it's a citation)
4. **Frame the story as**: "domain knowledge is the decisive factor" — NOT "we tried PCA and it failed" (anti-pattern from introduction_flow.md)
5. **Keep each paragraph focused**: one main idea per paragraph, clear topic sentence
6. **Note about Stefas2025**: it is a preprint — mention this in text when citing (Phase 2 risk)
7. **Approximate length**: 600–900 words for the entire Introduction section

---

## Step 2: Write Methodology Section (§2)

Rewrite `\section{Procedure / Methodology}` in `main.tex` (lines 180–249), replacing the existing placeholder/skeleton with polished prose. Follow the methodology_flow.md 12-paragraph plan and the Phase 3 outline §2.2.

### Structure and paragraph mapping:

| Para # | Subsection | Role | Content (from methodology_flow.md) |
|:---:|---|---|---|
| 1 | §2 Overview | Opening | Task setting (OES → H₂O₂ yield); 4-phase hypothesis-driven structure; subsection map |
| 2 | §2.1 Dataset | Opening + Detail | Gao et al. dataset: 701-point OES, 4 discharge params, H₂O₂ target, sample count |
| 3 | §2.1 Evaluation | Justification | Why LOOCV (small dataset, unbiased, no fold variance); R² and RMSE definitions; data integrity |
| 4 | §2.2 Configs | Detail + Justification | Config A/B/C as controlled-variable design; Config B as baseline; purpose of isolation |
| 5 | §2.3 Baseline | Opening + Detail | 7 models spanning 3 categories (linear/non-linear/deep); PCA 701→11; XGBoost anomaly note |
| 6 | §2.4 Tuning | Detail + Justification | Optuna TPE, 100+ trials; purpose: test if tuning compensates for feature quality |
| 7 | §2.5 Domain Features | Motivation | PCA discards physical meaning; traditional spectroscopy uses ratios/bands; modern ML overlooks this |
| 8 | §2.5 Domain Features | Detail (core) | 13 features: 7 lines + 3 bands + 3 ratios; species correspondence; physical meaning |
| 9 | §2.5 Domain Features | Advantage + Transition | Drift-invariant, interpretable, lower dimensionality; contrast with PCA; transition to §2.6 |
| 10 | §2.6 Interpretability | Opening + Detail | 4 importance methods → consensus ranking |
| 11 | §2.6 Validation | Detail | Bootstrap (500 iterations, 95% CI); permutation test (2000 shuffles) |
| 12 | §2.6 Reduction | Detail + Justification | Backward elimination + category ablation; two complementary strategies |

### Writing rules for Methodology:

1. **Follow the methodology_flow.md 12-paragraph plan strictly** — this is the primary guide
2. **NO R² values or performance numbers** in Methodology — all go in §3 Results
3. **Frame 4 phases as hypothesis-driven**, not chronological (anti-pattern #1)
4. **Justify every method choice** — don't just list methods (anti-pattern #2)
5. **Give §2.5 (Domain Features) proportionally more space** (3 paragraphs vs 1 for other phases) — this is the central contribution (anti-pattern #4)
6. **For standard methods** (PCA, Ridge, Optuna, Bootstrap): cite, don't re-derive formulas. Only detail YOUR contributions (13 OES features)
7. **Include `tab:configs`** — update the existing table if needed (Config A should reflect Phase 3: 13 domain features)
8. **Include `tab:oes_features`** — create the 13-feature table with columns: Feature, Type, Wavelength, Plasma Species, Physical Significance. Use the draft from methodology_flow.md §6
9. **Use `\cite{TODO}` placeholders** where feature justification citations are missing (user will add later)
10. **Approximate length**: 1000–1500 words for the entire Methodology section

### Table updates required:

- **`tab:configs`**: Update Config A description to "13 domain-knowledge OES features" (not "PCA-reduced"); update Config C to "17 features (13 OES + 4 discharge)". Consider adding a note that Phases 1–2 used PCA while Phase 3+ used domain features.
- **`tab:oes_features`** (NEW): Create from the draft in methodology_flow.md §6. Place in §2.5 or reference Appendix C.

---

## Step 3: Quality Checks

After writing both sections, verify:

### Content checks:
- [ ] Introduction follows the "domain knowledge is decisive" narrative — NOT "PCA failed"
- [ ] Introduction §1.3 cites all key papers: Gao2024, Laux2003, Srikar2025, Stefas2025, Wang2019, Park2021, Wang2025, Paris2005, Cai2024, Wang2021PlasmaTarML
- [ ] Methodology contains ZERO R² values or performance numbers
- [ ] §2.5 Domain Features has 3 paragraphs (more than any other subsection)
- [ ] Every method choice has a "why" — LOOCV, 7 models, 3 configs, Optuna, 13 features, consensus importance
- [ ] `tab:oes_features` lists all 13 features with physical justification
- [ ] `tab:configs` is updated for Phase 3 feature counts
- [ ] `\cite{TODO}` placeholders are used where citations are missing (not invented keys)

### Style checks:
- [ ] IEEE citation style (`\cite{key}`)
- [ ] Consistent use of LaTeX formatting: `\textbf{}`, `\textsubscript{}`, `\si{}`
- [ ] No orphan TODO markers from the skeleton remain (except deliberate `\cite{TODO}`)
- [ ] Paragraphs have clear topic sentences
- [ ] Transitions between subsections are smooth
- [ ] No first-person ("I") in Introduction or Methodology — use passive voice or "this project"

### LaTeX checks:
- [ ] All `\label{}` and `\ref{}` commands are consistent
- [ ] Tables compile correctly (no missing `\\` or `&`)
- [ ] No undefined citation keys (use only keys from references.bib or `TODO`)

---

## Deliverables

| # | Deliverable | Location | Description |
|---|---|---|---|
| 1 | Rewritten Introduction (§1) | `final report/report/main.tex` §1 | ~7 paragraphs, ~600–900 words |
| 2 | Rewritten Methodology (§2) | `final report/report/main.tex` §2 | ~12 paragraphs, ~1000–1500 words, includes updated `tab:configs` and new `tab:oes_features` |

---

## Quality Criteria

1. **Introduction** tells the "domain knowledge is decisive" story with proper literature positioning — not a PCA failure narrative
2. **Methodology** is hypothesis-driven (not chronological diary) with every method choice justified
3. §2.5 (Domain Features) is clearly signalled as the central methodological contribution with proportional space
4. No results leak into Methodology — clean separation between method description and findings
5. All anti-patterns from introduction_flow.md and methodology_flow.md are avoided
6. Writing quality matches the research-paper-writing skill standards: clear paragraph flow, strong topic sentences, no method dumps
