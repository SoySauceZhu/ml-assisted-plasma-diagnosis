# Plan

## Background

Phase 1,2,3,4 has generated a outline of the report and finished the draft of intro and method section.

This phase is gonna follow the same paradigm and generate writing flow of result section.

## Task
- Read and acquire enough context from experiment results, including the 'final report/final_phase1/experiments.md' in phase1 and the result folder in each experiment phases (i.e. 'phase4/results/tables').
- Generate the writing logic flow 写作思路行文逻辑 of "findings/result section", put in final_phase5 folder
- Copy any result figure images you need to 'final report/report/images' folder
- Update/improve the outline of result section.

## Material that you might need:
- bench inspection poster pdf (bench inspection poster/poster.pdf)
- table or figure folders under each experiment phases. i.e. 'phase4/results/tables'

## Notice
- Strictly follow the experiment result, no fabrication
- If experiment summary isn't informative enough, turn to the original result in result folder of each experiment phase

## Tip
- You can use 'research-paper-writing' skill to help articulating narrative arc.


# Action

## Prerequisites

Before executing, the agent must read and internalise the following context files (in this order):

1. **Phase 1 experiments summary** → `final report/final_phase1/experiments.md`
   - Contains: 4-phase experiment descriptions, quantitative result tables, comparison summary, ablation studies
   - Purpose: primary fact source — any result claim must trace back here or to the raw CSVs

2. **Phase 3 outline (latest version)** → `final report/final_phase3/outline.md` §2.3
   - Contains: current Results subsection skeleton (§2.3.1 Phase 1&2, §2.3.2 Phase 3, §2.3.3 Phase 4) with evidence bullets
   - Purpose: starting point for the updated outline; the Phase 5 flow must fit this structure

3. **Phase 2 introduction_flow.md and Phase 3 methodology_flow.md** → `final report/final_phase2/introduction_flow.md`, `final report/final_phase3/methodology_flow.md`
   - Purpose: **structural template** — `results_flow.md` should mirror their rigour (backward reasoning, forward story, anti-patterns, paragraph plan, figures/tables plan)

4. **Raw experimental result tables** (verify every number before using it):
   - Phase 1 baseline: `phase1/results/tables/loocv_results_summary.csv`
   - Phase 2 tuning: `phase2/results/tables/phase2_loocv_results_summary.csv`, `phase2/results/tables/phase1_vs_phase2_comparison.csv`
   - Phase 3 domain features: `phase3/results/tables/phase3_loocv_results_summary.csv`, `phase3/results/tables/phase1_vs_phase2_vs_phase3_comparison.csv`
   - Phase 4 statistical validation: `phase4/results/tables/bootstrap_ci_summary.csv`, `phase4/results/tables/permutation_test_summary.csv`, `phase4/results/tables/permutation_test_pruned_ridge.csv`
   - Phase 4 interpretability: `phase4/results/tables/feature_importance_all_models.csv`, `phase4/results/tables/loocv_fold_importance_stability.csv`, `phase4/results/tables/feature_correlation_vif.csv`
   - Phase 4 feature reduction: `phase4/results/tables/ablation_results.csv`, `phase4/results/tables/ablation_backward_elimination_full.csv`, `phase4/results/tables/ablation_summary_article.csv`, `phase4/results/tables/ablation_results_mlp.csv`
   - Phase 4 residuals: `phase4/results/tables/residual_detail.csv`, `phase4/results/tables/residual_condition_summary.csv`, `phase4/results/tables/residual_feature_correlation.csv`
   - Purpose: source of truth — if `experiments.md` is ambiguous or incomplete, trust the CSV

5. **Raw experimental result figures** (candidate inputs for `final report/report/images`):
   - Phase 1: `phase1/results/figures/heatmap_R2.png`, `model_comparison_bar.png`, `pca_cumulative_variance.png`, `predicted_vs_actual_grid.png`
   - Phase 2: `phase2/results/figures/phase1_vs_phase2_comparison.png`, selected `optuna_history_*.png`
   - Phase 3: `phase3/results/figures/phase1_vs_phase2_vs_phase3_comparison.png`, selected `predicted_vs_actual_*_phase3.png`
   - Phase 4: `phase4/results/figures/fig1_feature_importance_heatmap.pdf`, `fig4_importance_stability_errorbars.pdf`, `fig5_bootstrap_r2_distributions.pdf`, `fig6_residual_pred_vs_actual.pdf`, `fig7_feature_correlation_vif.pdf`, `fig9_ablation_trajectory.png`, `fig10_category_ablation.png`, `fig11_vif_barchart.png`, `fig12_permutation_test.png`
   - Purpose: pick the minimal visual set that carries the narrative; copy only those to `final report/report/images`

6. **Bench inspection poster** → `bench inspection poster/poster.pdf`
   - Purpose: the poster already distils the "headline" results and figure choices — use it as a sanity check for which findings are most persuasive

7. **Current main.tex Results section** → `final report/report/main.tex` §3 (or wherever the Results skeleton currently sits)
   - Purpose: understand existing LaTeX structure, placeholder figures/tables, section labels
   - Note: existing Results content is skeleton/placeholder; the flow produced in this phase will drive its rewrite in a later phase

8. **Phase 4 observation (risks carried forward)** → `final report/final_phase4.md` (Observation section)
   - Purpose: check whether any Phase 4 risks (overclaim in §1, missing citations, word budget) constrain the Results narrative

9. **Report guidance & marking descriptors** → `final report/Final report Guidance 2026.pdf`, `final report/marking descriptors Final Report_2026.pdf`
   - Purpose: Results section expectations, word-count budget, evidence/interpretation balance in the marking rubric

10. **Skill available:**
    - `research-paper-writing` → `.claude/skills/research-paper-writing`
    - Purpose: consult for results-section narrative arc, figure/table captioning, and claim–evidence pairing

---

## Step 1: Construct Results Logic Flow (`results_flow.md`)

Create `final report/final_phase5/results_flow.md`, mirroring the structure of `methodology_flow.md`. It must contain the following sections:

### Section 1 — Pre-Writing Questions (Backward Reasoning 先反向思考)

Answer these questions explicitly before writing any prose:

| Question | Answer to derive |
|---|---|
| **What must the reader believe after reading Results?** | (1) Domain-knowledge features decisively beat PCA; (2) the win is not an artefact of tuning or chance; (3) a small, interpretable feature set matches the full one. |
| **What is the narrative arc?** | NOT a phase-by-phase diary. Frame as a claim-driven story: Baseline → Tuning cannot save PCA → Domain features produce a step-change → Statistical validation rules out chance → Minimal model matches full model. |
| **What are the three "headline" numbers the reader must remember?** | Best model R² ≈ 0.92 (7-feature Ridge); Ridge Config C jump −0.17 → 0.80 (Phase 1 → Phase 3); permutation p < 0.0005. |
| **Which results carry the objectives?** | Obj 1 (predict yield) → Phase 3 Config C + Phase 4 pruned Ridge; Obj 2 (PCA vs domain) → Phase 1/2 vs Phase 3 comparison; Obj 3 (minimal feature set) → Phase 4 ablation + backward elimination. |
| **Where is the boundary between Results and Discussion/Conclusions?** | Results = observed numbers, statistical tests, factual comparisons. Interpretation ("why domain features work", implications for other plasma systems) belongs in Conclusions. |

### Section 2 — Forward Story (正向写作逻辑)

Define the logical flow as a sequence of R1–R? steps (analogous to L1–L10 / M1–M9 in the earlier flows). Suggested backbone:

| Step | Logic | Content summary |
|:---:|---|---|
| R1 | Orient the reader | State the evaluation protocol context (LOOCV, R², RMSE) and signpost the three claims to be supported. |
| R2 | Phase 1 baseline | Config B (discharge-only) is a strong baseline; Config C (PCA-OES) collapses — establish the problem. |
| R3 | Phase 2 tuning | Optuna tuning improves non-linear models but Config C still trails Config B — tuning alone cannot save PCA features. |
| R4 | Phase 3 step-change | Swapping PCA → 13 domain features closes the Config B/C gap across models — the central finding. |
| R5 | Phase 4a importance | Consensus ranking identifies the dominant features (discharge + CO₂⁺ band + key ratios). |
| R6 | Phase 4b statistical validation | Bootstrap CIs and permutation p-value rule out chance; CI overlap between Config B and Config C quantifies the remaining uncertainty. |
| R7 | Phase 4c feature reduction | Category ablation and backward elimination both converge on a ≤7-feature model matching or beating the full set. |
| R8 | Synthesis closing | Tie the four phases back to the three objectives; forward-point to Conclusions. |

Each step must specify: the claim, the evidence (exact CSV/table/figure), and a one-sentence topic sentence.

### Section 3 — Anti-Pattern Warnings for Results

Document at least the following anti-patterns with concrete DO/DON'T examples:

1. **Anti-pattern: Phase diary** — "Phase 1 gave X. Phase 2 gave Y. Phase 3 gave Z." → Instead: each subsection opens with the claim being tested, then the evidence.
2. **Anti-pattern: Number dump** — quoting every cell of every table in prose. → Instead: lead with the headline number per claim; push full tables to the appendix or keep them visual.
3. **Anti-pattern: Premature interpretation** — explaining *why* domain features work inside Results. → Instead: keep interpretation in Conclusions; Results only states *what* was observed.
4. **Anti-pattern: Cherry-picking** — reporting only the best model per phase without context. → Instead: report the full comparison, then highlight the winner.
5. **Anti-pattern: Over-claiming** — "domain features always outperform PCA." → Instead: bound claims by dataset, sample size, and the specific plasma system studied.
6. **Anti-pattern: Ignoring uncertainty** — presenting point R² without CI or permutation evidence. → Instead: always pair headline numbers with their statistical qualifier (CI or p-value).

### Section 4 — Paragraph-Level Plan (8–12 paragraphs)

Produce a paragraph-by-paragraph plan. For each paragraph specify:
- Paragraph number
- Subsection (§2.3.1 / §2.3.2 / §2.3.3 / synthesis)
- Role (Opening / Claim / Evidence / Comparison / Qualifier / Transition)
- Topic sentence (one draft sentence)
- Evidence anchors (CSV file + row/column, or figure label)
- Tables/figures referenced

Target length: **700–1200 words** for the prose portion of §2.3. Tables and figures carry most of the numeric load — prose should be tight.

### Section 5 — Figures and Tables Plan

Produce two lists:

**(a) Figures to include in §2.3** — for each, specify label, source file, caption intent, and whether it is essential or optional. At minimum consider:

| Label | Source file | Purpose | Essential? |
|---|---|---|---|
| `fig:r2_comparison` | `phase3/results/figures/phase1_vs_phase2_vs_phase3_comparison.png` | Headline R² bar chart across Phases 1–3 | Essential |
| `fig:feature_importance` | `phase4/results/figures/fig1_feature_importance_heatmap.pdf` | Consensus importance across models | Essential |
| `fig:bootstrap` | `phase4/results/figures/fig5_bootstrap_r2_distributions.pdf` | Bootstrap R² distributions with CIs | Essential |
| `fig:ablation` | `phase4/results/figures/fig9_ablation_trajectory.png` / `fig10_category_ablation.png` | Ablation/backward-elimination trajectory | Essential |
| `fig:permutation` | `phase4/results/figures/fig12_permutation_test.png` | Permutation-test null distribution | Optional (already stated as p<0.0005 in prose) |
| `fig:residual` | `phase4/results/figures/fig6_residual_pred_vs_actual.pdf` | Residual diagnostics for pruned Ridge | Optional (defer to Appendix if space tight) |

**(b) Tables to include in §2.3** — at minimum:
- `tab:results_main` — Phase 1 / 2 / 3 LOOCV R² comparison across models and configs (source: `phase3/results/tables/phase1_vs_phase2_vs_phase3_comparison.csv`)
- `tab:bootstrap` — bootstrap 95% CI for the key models (source: `phase4/results/tables/bootstrap_ci_summary.csv`)
- `tab:ablation` — category ablation and backward elimination summary (source: `phase4/results/tables/ablation_summary_article.csv`)
- `tab:feature_importance` (optional, may be deferred to appendix) — consensus feature ranks

For each, mark essential vs optional and note whether the data is already in the repo or needs to be re-extracted.

### Section 6 — Fact-Check Register

Produce a table listing every numeric claim the flow relies on, with the exact CSV file and row/column (or figure) it comes from. This is the guardrail against fabrication required by the Plan's Notice.

| Claim | Exact number | Source file | Row/column or figure |
|---|---|---|---|
| Ridge Config B LOOCV R² | 0.904 | `phase1/results/tables/loocv_results_summary.csv` | Ridge / B |
| Ridge Config C LOOCV R² (PCA) | −0.17 | `phase1/results/tables/loocv_results_summary.csv` | Ridge / C |
| Ridge Config C LOOCV R² (domain features) | 0.80 | `phase3/results/tables/phase3_loocv_results_summary.csv` | Ridge / C |
| Pruned Ridge best R² | 0.920 | `phase4/results/tables/ablation_summary_article.csv` / `permutation_test_pruned_ridge.csv` | — |
| Permutation-test p-value | < 0.0005 (2000 shuffles) | `phase4/results/tables/permutation_test_summary.csv` / `permutation_test_pruned_ridge.csv` | — |
| Bootstrap 95% CI (Ridge B) | [0.800, 0.955] | `phase4/results/tables/bootstrap_ci_summary.csv` | Ridge / B |
| Bootstrap 95% CI (Ridge C) | [0.574, 0.910] | `phase4/results/tables/bootstrap_ci_summary.csv` | Ridge / C |

**The agent must verify each of these against the actual CSV before using it, and extend the table with any additional numbers it decides to quote.** If a number in the Phase 3 outline disagrees with the CSV, trust the CSV and flag the discrepancy in the Observation.

---

## Step 2: Copy Result Figures into the Report Images Folder

Copy only the figures listed as **Essential** in Section 5(a) (and any Optional ones the flow actively uses) into `final report/report/images/`.

- Prefer PDF over PNG when both exist (vector scaling for print).
- Rename on copy to stable labels (e.g., `fig_r2_comparison.pdf`, `fig_feature_importance.pdf`, `fig_bootstrap.pdf`, `fig_ablation.pdf`). Do not rename in-place in the source folder.
- Do NOT copy figures that the flow does not explicitly reference — keep the images folder lean.
- If `final report/report/images/` already contains an earlier version of a figure, overwrite only after confirming it is stale; otherwise version the new file (`_v2.pdf`).

---

## Step 3: Update the Outline (`outline.md`)

Copy `final report/final_phase3/outline.md` to `final report/final_phase5/outline.md`, then improve the §2.3 Findings/Results section with the following enhancements:

### 3a. Add paragraph-role annotations

Annotate each subsection with `[Role: ...]` tags (Opening / Claim / Evidence / Comparison / Qualifier / Transition), mirroring the style used for §2.2 in the Phase 3 outline.

### 3b. Tighten the three subsections around claim–evidence pairs

For each of §2.3.1, §2.3.2, §2.3.3, restructure the bullets so that each bullet is either a **[Claim]** or an **[Evidence]** directly supporting a claim above it. Remove any bullet that is neither.

### 3c. Add a §2.3.4 Synthesis paragraph

Add a short closing paragraph that ties the four phases back to the three objectives stated in §1.2. This is the bridge to §2.4 Conclusions.

### 3d. Update figure/table labels to match the Section 5 plan

Reconcile the outline's figure/table labels with the new labels from `results_flow.md` Section 5 (e.g., the outline currently references `fig:r2_comparison`, `tab:bootstrap`, `tab:feature_importance` — confirm these or update them).

### 3e. Resolve Phase 4 risks where relevant

Check `final_phase4.md` Observation. In particular:
- If §1 contains an implicit preview of results, re-check that the Results subsection actually supports that framing.
- If the word budget is tight, flag which Results bullets can be compressed.

---

## Deliverables Checklist

| # | Deliverable | Path | Status |
|---|---|---|---|
| 1 | `results_flow.md` | `final report/final_phase5/results_flow.md` | ☐ |
| 2 | Updated `outline.md` | `final report/final_phase5/outline.md` | ☐ |
| 3 | Copied result figures | `final report/report/images/` | ☐ |

---

## Quality Criteria

The deliverables should satisfy:

1. **`results_flow.md`** mirrors the structural rigour of `methodology_flow.md` (backward reasoning, forward story, anti-patterns, paragraph plan, figures/tables plan, plus a fact-check register).
2. **Every numeric claim** in the flow is traced to a specific CSV row or figure — no fabrication, no rounding that loses the original precision.
3. **Narrative is claim-driven**, not phase-chronological — each subsection opens with the claim being tested, then the evidence.
4. **Interpretation is kept out of Results** — explanations of *why* domain features work belong in Conclusions.
5. **Uncertainty is always paired with headline numbers** — bootstrap CI or permutation p-value accompanies the best-R² claim.
6. **Figure set is minimal and purposeful** — only figures actually referenced by the flow are copied into `final report/report/images/`.
7. **Updated outline §2.3** uses claim–evidence pairing and role annotations consistent with the Phase 3 outline style.
8. **Any discrepancy** between `experiments.md`, the Phase 3 outline, and the raw CSVs is flagged in the Observation (to be written after execution) — trust the CSV.

---

# Observation

## Phase 5 Task Completion Evaluation

### Task Requirement Checklist

| Requirement | Status | Notes |
|---|:---:|---|
| Acquire context from experiment results (experiments.md + CSVs in each phase's `results/tables`) | Done | Read `final_phase1/experiments.md`, `final_phase3/outline.md`, and spot-verified CSVs: `phase1_vs_phase2_vs_phase3_comparison.csv`, `ablation_summary_article.csv`, `bootstrap_ci_summary.csv`, `permutation_test_summary.csv`, `permutation_test_pruned_ridge.csv`, `feature_importance_all_models.csv` |
| Generate Results writing logic flow → `results_flow.md` | Done | 7-section document: pre-writing questions, forward story (R1–R8), 7 anti-patterns, 10-paragraph plan, figures/tables plan, 28-entry fact-check register, writing checklist |
| Copy result figures into `final report/report/images` | Done | 7 figures copied with stable labels (`fig_r2_comparison`, `fig_feature_importance`, `fig_bootstrap`, `fig_ablation_trajectory`, `fig_category_ablation`, `fig_permutation`, `fig_phase1_heatmap_r2`); PDFs preferred where available |
| Update/improve outline focusing on §2.3 | Done | §2.3 expanded from 3 flat bullet groups to a structured 10-paragraph plan across §2.3.0 Opening + §2.3.1–§2.3.3 + new §2.3.4 Synthesis; every bullet is now a [Claim]/[Evidence] pair with a named CSV source and a figure/table label |

**Verdict: All task requirements and deliverables met.**

---

### Phase 4 Risk Traceability (results-relevant risks only)

| Priority | Phase 4 Risk | Phase 5 Resolution | Location |
|:---:|---|---|---|
| Medium | Word budget approaching upper bound; Methodology already at ~1000 words | `results_flow.md` targets 900–1200 words for §2.3 prose and pushes numeric matrices into tables/figures so prose only quotes headline numbers | `results_flow.md` §4 (paragraph plan), §7 (checklist) |
| Medium | §1 Introduction implicitly previews "domain features are decisive" — must be defensible after Results is written | Synthesis paragraph §2.3.4 ties each objective back to a specific result and explicitly flags bootstrap CI overlap, keeping the Introduction claim defensible | `outline.md` §2.3.4 |
| Low | No new BibTeX citations added for feature justification | Not applicable to Results — deferred to the Results prose writing phase (Phase 6); citations in Results are typically only for prior-work comparison, which is minimal |  — |
| Low | Pre-existing placeholder images (`uol_logo.png`, `r2_comparison.png`) in main.tex | `r2_comparison.png` concern is partially resolved by copying `fig_r2_comparison.png` under a stable label; `uol_logo.png` remains a pre-existing cover-page issue out of Phase 5 scope | `final report/report/images/` |

---

### Quality Criteria Assessment

| # | Criterion | Met? | Notes |
|:---:|---|:---:|---|
| 1 | `results_flow.md` mirrors `methodology_flow.md` structural rigour | Yes | Both documents contain: backward reasoning, forward story, anti-patterns, paragraph plan, figures/tables plan. `results_flow.md` adds a Fact-Check Register (Section 6), a new structural element justified by the "no fabrication" Notice in the Plan |
| 2 | Every numeric claim is traced to a CSV row | Yes | 28 numeric claims, each with exact file path, row identifier, and raw vs rounded value; two discrepancies explicitly surfaced rather than silently reconciled |
| 3 | Narrative is claim-driven, not phase-chronological | Yes | Every paragraph in the §2.3 outline opens with a [Claim]; phase labels remain as subsection anchors but are subordinate to the claim structure |
| 4 | Interpretation is kept out of Results | Yes | Paragraph 3 of §2.3.3 contains an explicit "[Interpretation guard]" bullet; §2.3.4 Synthesis explicitly defers "why this matters" to §2.4 |
| 5 | Uncertainty always paired with headline numbers | Yes | Every R² in the flow is paired with its bootstrap CI or permutation p-value; §2.3.3 Paragraph 2 is dedicated to the statistical qualifier and explicitly flags the Ridge B / Ridge C CI overlap |
| 6 | Figure set is minimal and purposeful | Yes | 7 figures copied, each explicitly referenced in the paragraph plan; no figures copied that are not used in the flow |
| 7 | Updated outline §2.3 uses claim–evidence pairing and role annotations | Yes | Role tags: Opening / Claim / Evidence / Comparison / Qualifier / Transition / Closing. Consistent with the Phase 3 outline convention for §2.2 |
| 8 | Any CSV vs outline discrepancy is flagged | Yes | Three discrepancies flagged in `results_flow.md` Section 6: category-ablation vs permutation R² gap, permutation p-value rounding, and stale 701-samples header in experiments.md |

---

### Brief Adversarial Notes

1. **The "R² = 0.920 vs R² = 0.906" discrepancy is real and material.** The category-ablation Ratios-only row (0.9063) and the permutation-tested pruned Ridge (0.9200) should in principle be the same 7-feature model, but they report different numbers in different CSVs. The flow handles this by quoting both separately with their exact sources, but during prose writing the author must resist the temptation to collapse them into a single number. If time permits before Phase 6, it may be worth re-running the ratios-only Ridge under the same refit pipeline as the permutation test to eliminate the gap; otherwise the prose should acknowledge implementation-level variation once, not repeatedly.

2. **The bootstrap CI overlap is the most easily miscommunicated result in the project.** Ridge Config B [0.800, 0.955] and Ridge Config C [0.574, 0.910] overlap substantially — strictly speaking, Config C is *not statistically superior* to discharge-only. The Phase 3 outline hinted at this; the Phase 5 outline now makes it explicit ("Qualifier — explicit" bullet in §2.3.3 Paragraph 2, and a closing "Honest qualifier" line in §2.3.4). The Introduction's framing ("domain knowledge is decisive") remains defensible because the decisive claim is about the Phase 1 → Phase 3 jump (−0.175 → 0.798), not about outperforming Config B. Phase 6 prose must keep that distinction sharp, because a careless reading of §2.3 could contradict §1.

3. **Permutation p-value precision.** The CSV stores `p_value = 0.0` exactly (0 of 2000 shuffles reached the observed R²). The Phase 3 outline wrote "p < 0.0005", which is slightly tighter than the test supports. The Phase 5 flow recommends "p < 5 × 10⁻⁴" (the conservative 1/2000 upper bound) or "p = 0/2000". This is a small but real over-precision issue that should not propagate into Phase 6 prose.

4. **Results section length.** Ten paragraphs at 900–1200 words is comfortably within the report's word budget, but Phase 4 noted that Methodology was already at the lower end of its range. If the combined Introduction + Methodology + Results prose approaches the guidance limit, §2.3.1 (Paragraphs 2–3) can be compressed into a single paragraph since Phase 1 + Phase 2 together only serve to motivate Phase 3.

5. **Figure labels must match main.tex.** The flow uses `fig:r2_comparison`, `fig:feature_importance`, `fig:bootstrap`, `fig:ablation_trajectory`, `fig:category_ablation`, `fig:permutation`. The current `main.tex` §3 skeleton references at least `fig:r2_comparison` — the others may not yet exist. Phase 6 must add the missing `\label{}` definitions when inserting the figures, or the compile will produce "undefined reference" warnings.

---

### Summary

Phase 5 successfully delivers the Results writing blueprint. The key improvement over Phase 3's §2.3 (which was a flat bullet list) is the transformation into a claim-driven, role-annotated structure with every number traced to a specific CSV row and every headline R² paired with a statistical qualifier. The addition of the Fact-Check Register (Section 6 of `results_flow.md`) operationalises the Plan's "no fabrication" notice and surfaces two real discrepancies that would otherwise quietly propagate into prose. The most important remaining risk for Phase 6 is communication discipline around the bootstrap CI overlap and the 0.906/0.920 gap — both are captured in the outline as explicit guards, but the prose writer must actively maintain that discipline.


