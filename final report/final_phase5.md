# Plan

## Background

Phase 1,2,3,4 has generated a outline of the report and finished the draft of intro and method section.

This phase is gonna follow the same paradigm and generate writing flow of result section.

## Requirement:

虽然实验结果都有表格数据，而且"bench inspection poster/poster.pdf"已经有了基本的结果展示。

但是有些实验展示的结果图很差，我只想展示部分实验结果的图示。我想展示的图示结果如下：
- phase1: 
    - PCA 的 variance curve。(phase1/results/figures/pca_cumulative_variance.png)
    - phase1 model comparison bar, and grid. ('phase1/results/figures/model_comparison_bar.png', 'phase1/results/figures/predicted_vs_actual_grid.png')
- phase 2:
   - 我还没有想好 是否要在phase2单独展示结果。 也没有想好 如果需要的话，应该展示什么。
- phase 3:
    - 展示对比的bar graph (phase3/results/figures/phase1_vs_phase2_vs_phase3_comparison.png)
- phase 4:
    - Feature Importance heatmap (phase4/results/figures/fig1_feature_importance_heatmap.pdf)
    - boostrap (phase4/results/figures/fig5_bootstrap_r2_distributions.pdf)
    - permutation result (the figure is not proper. Just use table or data only)
    - overall model ranking (phase4/results/figures/rank_on_model_config.png, phase4_result.txt)

## Task
- Read and acquire enough context from experiment results, including the 'final report/final_phase1/experiments.md' in phase1 and the result folder in each experiment phases (i.e. 'phase4/results/tables').
- Generate the writing logic flow 写作思路行文逻辑 of "findings/result section", put in final_phase5 folder
- Copy any result figure images you need to 'final report/report/images' folder
- Update/improve the outline of result section.

## Material that you might need:
- experiment summary: final report/final_phase1/experiments.md
- bench inspection poster pdf (bench inspection poster/poster.pdf)
- table or figure folders under each experiment phases. i.e. 'phase4/results/tables'

## Notice
- If experiment summary isn't informative enough, turn to the original result in result folder of each experiment phase
- You can use 'research-paper-writing' skill to help articulating narrative arc.
- If there is any inconsistency between phase 5 and previous phases, place a notice in observation section when instructed. I'll manually fix in later phases.

---

# Action

## Prerequisites

Before executing, the agent must read and internalise the following context files (roughly in order):

1. **Phase 5 outline (current)** → `final report/final_phase5/outline.md`
   - The latest outline already contains §2.3 Findings/Results sub-structure (Phase 1&2 / Phase 3 / Phase 4) with evidence bullets and quantitative anchors.
   - Purpose: **primary blueprint** — every results paragraph should align with this outline; the writing flow generated in this phase refines it into narrative form.

2. **Phase 1 experiment summary** → `final report/final_phase1/experiments.md`
   - Cross-phase quantitative tables, the "decisive-finding" narrative, and ablation/bootstrap/permutation summaries.
   - Purpose: authoritative source of numbers. Every R²/RMSE claim in the results section must trace back here (or to the raw CSVs).

3. **Raw result artefacts (tables + figures)** per phase:
   - `phase1/results/tables/loocv_results_summary.csv`, `phase1/results/figures/`
   - `phase2/results/tables/phase2_loocv_results_summary.csv`, `phase2/results/tables/phase1_vs_phase2_comparison.csv`
   - `phase3/results/tables/phase3_loocv_results_summary.csv`, `phase3/results/tables/phase1_vs_phase2_vs_phase3_comparison.csv`
   - `phase4/results/tables/*.csv` (bootstrap, permutation, ablation, feature importance, final ranking), `phase4_result.txt`
   - Purpose: verify any claim before writing; prefer these over the experiment summary if discrepancies appear.

4. **Bench inspection poster** → `bench inspection poster/poster.pdf`
   - Purpose: reference for how results were previously framed visually. Use as inspiration for narrative hooks, but the final report should go deeper than the poster.

5. **Previous writing flows** (for consistency of narrative arc):
   - `final report/final_phase2/introduction_flow.md`
   - `final report/final_phase3/methodology_flow.md`
   - Purpose: Results section must resolve the hypotheses posed in the Methodology and fulfil the expectations set up in the Introduction. Reuse the same "domain knowledge is decisive" framing.

6. **Current main.tex** → `final report/report/main.tex`
   - Contains already-drafted Introduction (§1) and Methodology (§2), plus a skeleton Results section.
   - Purpose: understand what has been claimed in §1/§2 so the Results can land those claims without contradiction; note existing labels/figures so new ones are named consistently.

7. **References** → `final report/report/references.bib`
   - Purpose: reuse existing citation keys; use `\cite{TODO}` placeholders only if genuinely missing.

8. **Skill available:**
   - `research-paper-writing` (in `.claude/skills/research-paper-writing`) — consult for narrative arc, topic sentences, anti-pattern avoidance in results writing.

---

## Step 1: Build a quantitative fact sheet (internal scratchpad)

Before writing any narrative, the agent should extract and cross-check all numbers that will appear in the results section. This is an internal step — no deliverable file, but these facts anchor Step 2.

Minimum numbers to verify (from CSVs, not from memory):

- **Phase 1 baseline (7 models × 3 configs)**: Ridge/PLS Config B R² (should be ≈ 0.90); Ridge/MLP Config C R² (should be strongly negative with PCA); XGBoost anomaly.
- **Phase 2 tuning delta**: per-model improvement from Optuna; particularly MLP Config C (−1.13 → ≈ 0.37) and CNN Config C (best OES-only tuned model ≈ 0.77).
- **Phase 3 domain-feature jump**: Ridge Config C −0.17 → 0.80; MLP Config C 0.37 → 0.82; closing of the Config B ↔ Config C gap.
- **Phase 4**: consensus top-3 features; bootstrap 95% CIs for Ridge B and Ridge C; permutation-test observed R² and p-value; category ablation (ratios / bands / single-wavelength); backward elimination trajectory; two optimal reduced models (3 ratios + 4 discharge = 7 features → R² = 0.920; 1 OES + 4 discharge = 5 features → R² = 0.918); final model ranking.

If any number differs between `experiments.md` and the raw CSVs, trust the CSV and flag the discrepancy in the final Observation section.

---

## Step 2: Write the Results writing flow → `final report/final_phase5/results_flow.md`

Create `results_flow.md` following the same structure used for `introduction_flow.md` and `methodology_flow.md`. Required sections:

1. **Pre-writing questions** — what claim does this section have to land? What does the reader need to believe by the end?
2. **Backward reasoning** — start from the final conclusion (a 7-feature Ridge model achieves R² = 0.920 and domain knowledge is the decisive factor) and work backwards through the evidence that supports each link.
3. **Forward story (R0–Rn)** — numbered logical beats of the results narrative, each with: the claim, the supporting evidence (table/figure/number), and the transition to the next beat.
4. **Anti-pattern warnings** (≥ 4), at minimum:
   - Do not present Phase 1 as a "failure" — frame it as a controlled experiment that rules out one hypothesis (PCA-based features carry the signal).
   - Do not turn the results section into a table-by-table walkthrough ("method dump"). Each paragraph should make one interpretive point, not just display numbers.
   - Do not reintroduce methodology details — cite subsection labels instead.
   - Do not over-claim: with n = 20 the bootstrap CIs are wide; the narrative must acknowledge that Ridge B and Ridge C CIs overlap.
   - Do not bury Phase 3 — it is the central finding and deserves proportional space, mirroring how §2.5 was emphasised in the methodology.
5. **Paragraph-level plan** — one entry per paragraph with: subsection, paragraph role (Opening / Evidence / Interpretation / Transition), one-sentence content summary, figures/tables to reference, numbers to include.
6. **Figures and tables plan** — explicit list of which figures/tables will appear in §2.3 (see Step 3 for the image shortlist) and what each one argues. Distinguish must-have vs. optional.

The flow file should be self-contained so it can serve as the sole reference when drafting the §2.3 LaTeX prose in a later phase.

---

## Step 3: Copy chosen figures into the report image folder

Target directory: `final report/report/images/` (already exists). Copy only the figures the user explicitly selected in the Plan:

| # | Source path | Destination filename | Phase | Purpose |
|---|---|---|---|---|
| 1 | `phase1/results/figures/pca_cumulative_variance.png` | `pca_cumulative_variance.png` | P1 | Motivate PCA cut-off (11 components ≥ 95% variance) |
| 2 | `phase1/results/figures/model_comparison_bar.png` | `phase1_model_comparison_bar.png` | P1 | Show Phase 1 R² across 7 models × 3 configs |
| 3 | `phase1/results/figures/predicted_vs_actual_grid.png` | `phase1_predicted_vs_actual_grid.png` | P1 | Visual diagnostic of baseline model behaviour |
| 4 | `phase3/results/figures/phase1_vs_phase2_vs_phase3_comparison.png` | `phase1_vs_phase2_vs_phase3_comparison.png` | P3 | Central figure — shows the step-change from domain features |
| 5 | `phase4/results/figures/fig1_feature_importance_heatmap.pdf` | `fig1_feature_importance_heatmap.pdf` | P4 | Consensus importance across 4 methods |
| 6 | `phase4/results/figures/fig5_bootstrap_r2_distributions.pdf` | `fig5_bootstrap_r2_distributions.pdf` | P4 | Bootstrap R² distributions + 95% CIs |
| 7 | `phase4/results/figures/rank_on_model_config.png` | `rank_on_model_config.png` | P4 | Overall final ranking of model × config combinations |

Notes:
- Phase 2 has **no dedicated figure** — per Plan, the user has not decided. The writing flow should handle Phase 2 as a short in-line narrative paragraph that references tuning results via a small table or inline numbers, not a dedicated figure.
- The permutation test should be presented as **inline numbers or a small table** (not a figure), as the existing figure is judged inadequate.
- Use the existing filenames in the destination verbatim where possible so figure paths stay predictable. If a name collision exists with files already in `images/`, prefix with `phase<N>_` as shown above.
- Before copying, verify each source path exists. If a source is missing, record the gap in the Observation section rather than silently skipping.

---

## Step 4: Update the outline's §2.3 (Results)

Edit `final report/final_phase5/outline.md` §2.3 so that:

1. Each sub-subsection (2.3.1 Phase 1&2 / 2.3.2 Phase 3 / 2.3.3 Phase 4) gains an explicit **figure/table list** using the copied filenames in Step 3.
2. Evidence bullets remain quantitative but are reordered if Step 2's backward reasoning suggests a stronger narrative order.
3. A new bullet is added under §2.3.1 clarifying that Phase 2 will be covered in a short narrative paragraph (not a dedicated figure) — consistent with the user's Plan.
4. The "two optimal reduced models" block stays intact (already disambiguated in Phase 2).
5. Any inconsistency discovered between the outline's current quantitative claims and the raw CSVs is noted inline (`> [Phase 5 note]: …`) rather than silently corrected.

Do **not** rewrite §2.1/§2.2/§2.4 in this phase — they are out of scope.

---

## Step 5: Quality checks

Before finishing, verify:

### Content
- [ ] `results_flow.md` exists and contains all 6 required subsections
- [ ] Every number in the flow has a verified source (experiments.md or a specific CSV)
- [ ] Phase 3 receives proportionally more narrative space than Phase 1, 2 or 4 sub-parts
- [ ] Bootstrap CI overlap between Ridge B and Ridge C is explicitly acknowledged (honest uncertainty)
- [ ] Both "optimal reduced models" are described and disambiguated
- [ ] Phase 2 handling (no figure, inline narrative) is explicitly decided in the flow

### Files / paths
- [ ] All 7 figures listed in Step 3 have been copied into `final report/report/images/`
- [ ] `outline.md` §2.3 references figures by their destination filenames (not source paths)
- [ ] Source figures that could not be copied (if any) are listed in the Observation section

### Consistency with §1 and §2
- [ ] Results narrative does not contradict claims already made in main.tex §1 or §2
- [ ] "Domain knowledge is decisive" framing is preserved
- [ ] No R² values are assigned to methods that weren't described in §2.2

---

## Deliverables

| # | Deliverable | Location | Description |
|---|---|---|---|
| 1 | Results writing flow | `final report/final_phase5/results_flow.md` | Full writing logic flow following the same structure as `introduction_flow.md` / `methodology_flow.md` |
| 2 | Copied figures | `final report/report/images/` | 7 figure files (see Step 3 table) |
| 3 | Updated outline §2.3 | `final report/final_phase5/outline.md` | Figure/table lists added; narrative order refined; Phase 2 handling clarified |

No LaTeX edits to `main.tex` in this phase — the actual Results prose will be written in a later phase, using `results_flow.md` as the blueprint.

---

## Quality Criteria

1. **Traceability**: every quantitative claim in `results_flow.md` can be traced to a specific CSV or table cell.
2. **Narrative arc**: the Results section tells a coherent story (hypothesis → controlled test → domain-feature breakthrough → interpretability → minimal deployable model), not a chronological table dump.
3. **Proportionality**: Phase 3 is clearly the centre of gravity, mirroring §2.5's emphasis in the Methodology.
4. **Honest uncertainty**: wide CIs with n = 20 are acknowledged, not hidden.
5. **Anti-patterns avoided**: no method dumps, no "Phase 1 failed" framing, no re-derivation of methods, no over-claim from single-point estimates.
6. **Self-containment**: `results_flow.md` is usable as the sole blueprint when a later phase writes the §2.3 prose.

---

# Observation

## Summary

Phase 5 produced the three deliverables specified in the Action: a self-contained `results_flow.md`, seven figures copied into `final report/report/images/`, and a rewritten §2.3 in `final_phase5/outline.md`. The flow follows the same 8-part structure used for `introduction_flow.md` and `methodology_flow.md` (pre-writing questions → backward reasoning → forward story R0–R9 → anti-patterns → paragraph plan → figures plan → verified fact sheet → consistency notes). The narrative centre is R4 (Phase 3 step-change) with Ridge Config C −0.175 → 0.798 and MLP Config C −1.131 → 0.815 as the headline numbers, supported visually by `phase1_vs_phase2_vs_phase3_comparison.png`. Phase 4 beats (R5–R8) split importance, bootstrap uncertainty, permutation significance, and feature reduction into interpretive paragraphs rather than table walkthroughs. Every quantitative claim was verified directly against the raw CSVs under `phase{1,2,3,4}/results/tables/` before being written into the flow or the outline — no number was lifted from memory or `experiments.md` alone.

## What went well

- **Traceability held**: the fact sheet in §7 of `results_flow.md` lists each number with its source CSV; the bootstrap CIs, permutation p-value, consensus importance ranks, and ablation rows all match the raw tables.
- **Narrative arc is hypothesis-driven**: the forward story maps R1 → H1 rejected, R3 → H2 rejected, R4 → H3 supported, R5–R8 → H4 supported. This resolves the four hypotheses posed in §2.2 in the same order, so §2.3 will read as a payoff, not a diary.
- **Honest uncertainty is built in, not bolted on**: R6 explicitly acknowledges Ridge B ↔ Ridge C and Ridge C ↔ MLP C bootstrap CI overlap. AP4 flags over-claim as an anti-pattern, and R9's synthesis justifies Ridge by Occam's razor rather than by a capability gap.
- **Phase 3 is proportionally weighted**: R4 is marked as the longest paragraph and owns the centrepiece figure, mirroring the §2.2.5 emphasis in the Methodology.
- **Plan-specified figure selection was respected**: Phase 2 was deliberately left without a dedicated figure (R3 handled inline), and the permutation test is reported as inline numbers per the user's explicit instruction that the existing figure is inadequate.

## Discrepancies flagged during execution (for manual follow-up)

1. **`phase4_result.txt` vs. `phase4_result.md`** — the Plan references `phase4_result.txt`; the actual file on disk is `phase4_result.md`. All other figures and tables it pointed to exist. No action needed beyond noting the filename.
2. **Permutation p-value reporting** — `permutation_test_summary.csv` stores `p_value = 0.0` literally (i.e., no null R² ≥ observed across 2000 shuffles). The outline and flow report this as **p < 0.0005** (the tightest bound given n = 2000). This phrasing should be preserved when drafting §2.3 prose; avoid "p = 0".
3. **Ratios-only ablation R² = 0.906 vs. pruned-Ridge permutation R² = 0.920** — these refer to *different model fits*. The category ablation keeps the 3 ratios + 4 discharge features but uses the same Ridge fit as the broader ablation sweep (R² = 0.9063 from `ablation_summary_article.csv`). The permutation test refits Ridge on just that 7-feature subset, yielding R² = 0.9200. The prose for R7/R8 must make this distinction explicit to avoid looking like a typo.
4. **XGBoost anomaly** — flagged in §2.2.3 already, but worth re-stating in R1: XGBoost R² = −0.108 identical across all configs is a model artefact (default hyperparameters unsuitable for n = 20 + LOOCV), not a finding about OES. Mention briefly and then drop from the Phase 1 narrative.

## Risks / caveats for later phases

- **n = 20 ceiling**: every claim beyond the permutation-tested pruned model sits inside overlapping bootstrap CIs. The §2.3 prose must stay disciplined about this — the Phase 3 step-change is large in *magnitude* but not formally "significant" until the feature set is pruned. AP4 in the flow enforces this, but the later writing phase will need to resist shortcutting.
- **Figure inventory is not yet curated for consistency of style**: the seven copied figures mix PNG (P1, P3, P4 ranking) and PDF (P4 importance, P4 bootstrap). LaTeX handles both, but the final report may benefit from regenerating the PNGs as PDFs for uniform vector quality — out of scope for Phase 5.
- **Redundant figures already exist in `images/`**: the directory previously contained `fig_feature_importance.pdf`, `fig_bootstrap.pdf`, `fig_category_ablation.pdf`, `fig_r2_comparison.png`, etc. These were not deleted; the later writing phase should reference the newly-copied filenames (explicit in the flow/outline) and ideally clean up the older duplicates.
- **No `main.tex` changes yet**: §2.3 in `main.tex` is still skeleton/placeholder. The prose pass will need to preserve the label `\label{sec:results}` and ensure figure paths resolve to `images/<filename>`.
- **Tables for §2.3 are not yet drafted**: `tab:bootstrap`, `tab:feature_importance`, and `tab:ablation` are referenced but not created. They can be built directly from the CSVs during the prose-writing phase.

