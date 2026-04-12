# Plan

## Background

Phase 1,2 successfully generate the outline, reference bibtex and the logic flow of introduction section.

In this phase, we are gonna generate the logic flow of method section and also an improved outline for this section.

## Task

- Acquire enough context of my research, including data, settings, experiments, results, etc. Also look at documents and deliverables of previous phases.
- Generate the writing logic flow 写作思路行文逻辑 of "method section", put in final_phase3 folder
- Update/improve the outline, focuses on the method section.

## Deliverables
- updated outline.md
- writing flow of method section as methodology_flow.md


# Action

## Prerequisites

Before executing, the agent must read and internalise the following context files (in this order):

1. **Phase 1 experiments summary** → `final report/final_phase1/experiments.md`
   - Contains: all 4 phases' experiment descriptions, quantitative result tables, comparison summary, ablation studies
   - Purpose: provides the raw material (methods, settings, numbers) for the methodology section

2. **Phase 2 outline (current version)** → `final report/final_phase2/outline.md`
   - Contains: current §2.2 Procedure/Methodology outline (6 subsections: 2.2.1–2.2.6)
   - Purpose: starting point for the updated outline; identifies what's already structured

3. **Phase 2 introduction_flow.md** → `final report/final_phase2/introduction_flow.md`
   - Contains: Literature Map, Backward/Forward Logic Flow, Template Selection, Anti-Pattern Warning, Paragraph-Level Plan
   - Purpose: **structural template** — the methodology_flow.md should follow a similar level of rigour and structure

4. **Experimental data CSVs** (all in `phase1-4/results/tables/`):
   - `loocv_results_summary.csv` (Phase 1 baseline results)
   - `phase1_vs_phase2_comparison.csv`, `phase2_loocv_results_summary.csv` (Phase 2 tuning results)
   - `phase1_vs_phase2_vs_phase3_comparison.csv`, `phase3_loocv_results_summary.csv` (Phase 3 domain feature results)
   - `bootstrap_ci_summary.csv`, `feature_importance_all_models.csv` (Phase 4 statistical validation)
   - `ablation_results.csv`, `ablation_backward_elimination_full.csv`, `ablation_summary_article.csv` (Phase 4 feature reduction)
   - `permutation_test_summary.csv`, `permutation_test_pruned_ridge.csv` (Phase 4 permutation test)
   - Purpose: verify all quantitative claims; extract exact numbers for methodology detail

5. **Phase 2 observation (risks carried forward)** → `final report/final_phase2.md` (Observation section)
   - Purpose: check if any Phase 2 risks pertain to methodology writing

6. **Report main.tex** → `final report/report/main.tex`
   - Purpose: understand existing LaTeX structure, section labels, figure/table references

7. **References** → `final report/report/references.bib`
   - Purpose: available citation keys for methodology references (especially [optuna2019], [Gao2024NSCO2Discharge])
  
8. skill available:
   - research-paper-writing: ".claude/skills/research-paper-writing"

---

## Step 1: Construct Methodology Logic Flow (`methodology_flow.md`)

Create `final report/final_phase3/methodology_flow.md` with the following sections:

### Section 1 — Methodology Backward Reasoning (先反向思考)

Answer these questions to establish the writing logic:

| Question | Answer to derive |
|---|---|
| **What must the reader understand after reading Methodology?** | The complete experimental pipeline: dataset → configurations → 4 phases → evaluation protocol. Enough detail to reproduce. |
| **What is the narrative arc of Methodology?** | NOT chronological trial-and-error. Frame as: systematic investigation with controlled variables — each phase tests one hypothesis while holding others constant. |
| **What makes our methodology novel vs. standard ML?** | (1) Domain-knowledge feature engineering replacing PCA; (2) ablation-based feature reduction; (3) multi-model consensus importance; (4) rigorous statistical validation (bootstrap + permutation) |
| **What must be justified (not just described)?** | Why LOOCV (not k-fold)? Why these 7 models? Why these 13 OES features? Why these specific statistical tests? |

### Section 2 — Forward Story (正向写作逻辑)

Define the logical flow as a sequence of steps (similar to L1–L10 in introduction_flow.md):

| Step | Logic | Content summary |
|:---:|---|---|
| M1 | Dataset | Describe the dataset source, structure, target variable |
| M2 | Evaluation protocol | Justify LOOCV; define R² and RMSE metrics |
| M3 | Input configurations | Three configs (A/B/C) as controlled-variable design |
| M4 | Phase 1 — Baseline | 7 models × 3 configs with PCA; establish baseline |
| M5 | Phase 2 — Tuning | Optuna HPO for non-linear models; test if tuning suffices |
| M6 | Phase 3 — Domain features | 13 features replacing PCA; the central methodological contribution |
| M7 | Phase 4a — Importance | Multi-model consensus feature ranking |
| M8 | Phase 4b — Statistical validation | Bootstrap CI + permutation test |
| M9 | Phase 4c — Feature reduction | Backward elimination + category ablation |

### Section 3 — Anti-Pattern Warnings for Methodology

Identify and document writing anti-patterns specific to methodology:

1. **Anti-pattern: Chronological diary** — "First we tried X, then we tried Y." → Instead: frame each phase as a hypothesis test with clear rationale stated before results.
2. **Anti-pattern: Method dump** — listing model names/parameters without justification. → Instead: explain WHY each model was chosen (e.g., Ridge for linearity detection, CNN for raw-spectrum learning).
3. **Anti-pattern: Results in Methodology** — reporting R² values in the method section. → Instead: methodology only describes WHAT was done and WHY; all numbers go in Results.
4. **Anti-pattern: Burying the contribution** — treating domain-feature engineering as "just another phase." → Instead: clearly signal Phase 3 as the central methodological contribution.

### Section 4 — Paragraph-Level Plan for Methodology

Create a paragraph-by-paragraph plan (similar to the 8-paragraph plan in introduction_flow.md). For each paragraph specify:
- Paragraph number
- Role (Opening / Detail / Justification / Transition)
- First sentence message
- Content summary
- Key citations (if any)
- Tables/Figures referenced

The methodology section should contain approximately **10–14 paragraphs** covering:
- Dataset description (1–2 paragraphs)
- Evaluation protocol (1 paragraph)
- Input configurations (1 paragraph)
- Phase 1 baseline (1–2 paragraphs)
- Phase 2 tuning (1 paragraph)
- Phase 3 domain features (2–3 paragraphs — this is the core contribution)
- Phase 4 interpretability (2–3 paragraphs)

### Section 5 — Figures and Tables Plan

List all figures and tables that should appear in or be referenced from the Methodology section:

| Label | Type | Content | Location |
|---|---|---|---|
| `tab:configs` | Table | Input configuration summary (A/B/C) | §2.2.2 |
| `tab:oes_features` | Table | 13 OES features with wavelengths, species, physical meaning | §2.2.5 |
| `fig:pipeline` | Figure | Overall pipeline diagram (optional) | §2.2 opening or §2.2.5 |

Assess whether each is essential or optional.

---

## Step 2: Update Outline (`outline.md`)

Copy `final report/final_phase2/outline.md` to `final report/final_phase3/outline.md`, then improve the §2.2 Methodology section with the following enhancements:

### 2a. Add justification notes

For each subsection (2.2.1–2.2.6), add a **[Justification]** bullet explaining WHY the method choice was made (not just WHAT was done). Specifically:

- §2.2.1: Why LOOCV? (small dataset, maximises training data per fold, unbiased estimate)
- §2.2.2: Why three configurations? (controlled-variable design to isolate OES vs. discharge contributions)
- §2.2.3: Why these 7 models? (span linear → non-linear → deep learning; cover different inductive biases)
- §2.2.4: Why Optuna TPE? (efficient for small search budgets; cite [optuna2019])
- §2.2.5: Why these 13 features? (each linked to specific plasma chemistry species/reactions; cite [Laux2003OESAir], [Paris2005N2Ratio])
- §2.2.6: Why consensus importance? (no single importance method is unbiased — averaging reduces method-specific artifacts)

### 2b. Expand §2.2.5 (Domain Features) — this is the core

The current outline has 4 bullets for §2.2.5. Expand to include:
- Feature derivation rationale (which plasma species each line/band/ratio corresponds to)
- Physical meaning of each ratio (e.g., OH/Hα reflects hydrogen abstraction pathway; Hα/Hβ encodes electron temperature via Balmer decrement)
- Connection to traditional spectroscopic practice (cite [Paris2005N2Ratio] for line ratios, [Laux2003OESAir] for OES diagnostics)
- Contrast with PCA: domain features are drift-invariant, physically interpretable, and require no minimum sample size

### 2c. Add paragraph-role annotations

For each subsection, annotate with `[Role: ...]` tags indicating the paragraph's role in the narrative:
- `[Role: Opening]` — introduces the subsection's purpose
- `[Role: Detail]` — technical specifics
- `[Role: Justification]` — explains WHY
- `[Role: Transition]` — connects to next subsection

### 2d. Add Phase 2 risk resolution notes (if applicable)

Check the Phase 2 Observation for any remaining risks that pertain to methodology. Add resolution notes where relevant.

---

## Deliverables Checklist

| # | Deliverable | Path | Status |
|---|---|---|---|
| 1 | `methodology_flow.md` | `final report/final_phase3/methodology_flow.md` | ☐ |
| 2 | Updated `outline.md` | `final report/final_phase3/outline.md` | ☐ |

---

## Quality Criteria

The deliverables should satisfy:

1. **methodology_flow.md** mirrors the structural rigour of `introduction_flow.md` (backward reasoning, forward story, anti-patterns, paragraph plan, figures/tables plan)
2. **outline.md** §2.2 has justification for every method choice — no "method dump" without rationale
3. §2.2.5 (Domain Features) is clearly signalled as the central methodological contribution
4. Anti-pattern warnings are specific and actionable (not generic advice)
5. All quantitative claims in the outline can be traced to experimental data CSVs
6. No results (R² values, performance numbers) appear in the Methodology outline — these belong in §2.3 Results

# Observation

## Phase 3 Task Completion Evaluation

### Task Requirement Checklist

| Requirement | Status | Notes |
|---|:---:|---|
| Acquire context of research (data, experiments, results, previous phases) | Done | Read: experiments.md, Phase 2 outline, introduction_flow.md, references.bib, main.tex, feature_importance CSV, VIF CSV, Phase 2 observation; also loaded research-paper-writing skill method guide |
| Generate methodology writing logic flow → methodology_flow.md | Done | 7-section document: pre-writing questions, backward reasoning, forward story, 5 anti-patterns, 12-paragraph plan, figures/tables plan, writing checklist |
| Update/improve outline focusing on method section | Done | §2.2 expanded from 6 subsections (bullet-only) to 7 subsections (§2.2.0–2.2.6) with paragraph-role annotations, justification bullets, and 3-paragraph expansion of §2.2.5 |

**Verdict: All task requirements and deliverables met.**

---

### Phase 2 Risk Traceability (methodology-relevant risks only)

| Priority | Phase 2 Risk | Phase 3 Resolution | Location |
|:---:|---|---|---|
| Medium | No citation for physical justification of the 13 OES features (OH → H₂O₂ pathway) | Each of the 13 features now has a species correspondence and physical significance in outline §2.2.5 Paragraph 2; `tab:oes_features` draft provided in methodology_flow.md §6 | outline.md §2.2.5, methodology_flow.md §6 |
| Low | Stefas2025 is a preprint | Not addressed — deferred to prose writing phase (note in text when cited) | — |
| Low | LaTeX compilation of references.bib not tested | Not addressed — deferred to prose writing phase | — |
| Low | Summary in main.tex exceeds 200 words | Not addressed — deferred to prose writing phase | — |

---

### Quality Criteria Assessment

| # | Criterion | Met? | Notes |
|:---:|---|:---:|---|
| 1 | methodology_flow.md mirrors introduction_flow.md structure | Yes | Both have: backward reasoning, forward story, anti-patterns, paragraph plan, figures/tables plan |
| 2 | outline §2.2 has justification for every method choice | Yes | LOOCV, 3 configs, 7 models, Optuna TPE, 13 features, consensus importance — all justified |
| 3 | §2.2.5 signalled as central contribution | Yes | Explicit note + 3 paragraphs (vs 1 for other phases) + dedicated `tab:oes_features` |
| 4 | Anti-patterns are specific and actionable | Yes | 5 anti-patterns with concrete DO/DON'T examples (not generic "write clearly") |
| 5 | Quantitative claims traceable to CSVs | Yes | Feature importance ranks verified against `feature_importance_all_models.csv`; VIF values verified against `feature_correlation_vif.csv` |
| 6 | No R² values in Methodology outline | Partial | §2.3 Results section retains numbers (correct). §2.2 Methodology has no R² values in the method description itself, but §2.2.5 Paragraph 1 references Wang2025's R² ≈ 0.90–0.97 as literature context — acceptable since this is a citation, not our own result |

---

### Brief Adversarial Notes

1. **Feature physical justification depth:** The `tab:oes_features` draft in methodology_flow.md provides one-line physical significance per feature. For prose writing, some features need deeper justification — particularly OH (309 nm) as H₂O₂ precursor. The Phase 2 observation already flagged the need for 1–2 citations on OH radical pathways to H₂O₂ formation. This remains unresolved and should be addressed during prose writing.

2. **Results boundary:** The outline §2.2.4 contains a transition sentence that alludes to Phase 2 results ("confirmed that tuning... cannot compensate"). This is borderline — the sentence describes the strategic rationale for Phase 3 rather than reporting numbers, so it is acceptable as a forward pointer. During prose writing, keep this sentence qualitative (no R² values).

3. **Methodology length:** The 12-paragraph plan may produce a lengthy Methodology section (approximately 3–4 pages). For a BEng report this is acceptable given the 4-phase structure, but if space becomes tight, Phase 1 and Phase 2 can be compressed into a single subsection.

---

### Summary

Phase 3 successfully delivers the methodology writing blueprint. The key improvement over Phase 2 is the transformation of §2.2 from a flat bullet list into a structured, justified, role-annotated outline ready for prose writing. The methodology_flow.md provides the anti-pattern guardrails and paragraph plan needed to write the actual section. The most important remaining gap is deeper physical justification for OES feature selection (OH → H₂O₂ pathway citations), which should be addressed in the prose writing phase.
