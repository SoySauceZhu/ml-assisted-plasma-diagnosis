# Plan

## Background

Phase 5 has prepared figure images and articulate the narrative arc and outline of result section of my report. 

This phase you are gonna write the draft for result section.

## Task
- Acquire enough context about previous phases's instruction markdown, final report guidance and marking descriptor, result section writing logic flow, facts in my experiments, reference/citation in bibtex, and look at corresponding sections in the outline.
- Write Result section for my final report. Write in 'final\ report/report/main.tex'. 

## Tips
- As mentioned in phase1 instruction markdown, the report guidance asks for a technical report, but my work is a research project. So our content/writing logic should be like research paper style, but we should also meet the requirements of format or something detail.
- Use IEEE citation style
- You can use skill'research-paper-writing'

---

# Action

## Prerequisites

Before executing, the agent must read and internalise the following context files (roughly in order):

1. **Phase 5 results writing flow** → `final report/final_phase5/results_flow.md`
   - Contains: pre-writing questions, backward reasoning (A→B→C chain), forward story R0–R9, 6 anti-pattern warnings, paragraph-level plan, figures plan, verified quantitative fact sheet, and consistency notes
   - Purpose: **primary blueprint** — §2.3 prose follows this beat-by-beat (R0 → R9), with every number anchored in §7 of the flow

2. **Phase 5 outline §2.3** → `final report/final_phase5/outline.md`
   - Contains: the rewritten §2.3 with figure lists per sub-subsection, Phase 2 "no figure" decision, permutation "inline only" decision, and the ratios-0.906 vs. pruned-Ridge-0.920 disambiguation
   - Purpose: the structural map of the Results section; the flow provides the narrative, the outline provides the section tree

3. **Phase 5 Observation** → `final report/final_phase5.md` (Observation section)
   - Purpose: 4 discrepancies flagged during Phase 5 execution — must be resolved correctly in prose:
     - `phase4_result.txt` vs. `.md` (cosmetic — no prose impact)
     - Permutation `p = 0.0` in CSV must be reported as **p < 0.0005**, never "p = 0"
     - Ratios-only ablation R² = 0.906 vs. pruned-refit Ridge R² = 0.920 are **different fits** and must be distinguished in the prose
     - XGBoost anomaly (R² = −0.108 identical across configs) is a default-hyperparameter artefact; mention briefly and drop

4. **Phase 1 experiment summary** → `final report/final_phase1/experiments.md`
   - Purpose: secondary fact source (the flow's §7 is primary — it was verified from raw CSVs); cross-check only if the flow has a gap

5. **Raw result artefacts** (reference only if a number is missing from the flow):
   - `phase1/results/tables/loocv_results_summary.csv`
   - `phase2/results/tables/phase2_loocv_results_summary.csv`, `phase1_vs_phase2_comparison.csv`
   - `phase3/results/tables/phase1_vs_phase2_vs_phase3_comparison.csv`
   - `phase4/results/tables/bootstrap_ci_summary.csv`, `permutation_test_summary.csv`, `feature_importance_all_models.csv`, `ablation_summary_article.csv`

6. **Current main.tex** → `final report/report/main.tex`
   - Contains: already-drafted §1 Introduction and §2 Methodology; skeleton/placeholder §2.3 Results to be replaced
   - Purpose: understand existing LaTeX structure, label conventions, figure include patterns, and what subsection claims §2.3 must land without contradicting §1/§2

7. **Images directory** → `final report/report/images/`
   - Contains: the 7 Phase 5 figures already copied (`pca_cumulative_variance.png`, `phase1_model_comparison_bar.png`, `phase1_predicted_vs_actual_grid.png`, `phase1_vs_phase2_vs_phase3_comparison.png`, `fig1_feature_importance_heatmap.pdf`, `fig5_bootstrap_r2_distributions.pdf`, `rank_on_model_config.png`)
   - Purpose: exact filenames to use in `\includegraphics{}` commands — do not invent new paths

8. **References** → `final report/report/references.bib`
   - Purpose: available IEEE citation keys. §2.3 should need minimal new citations; most content is our own experimental results. Cite `optuna2019` when mentioning TPE, and the dataset paper `Gao2024NSCO2Discharge` only if introducing a new quantitative comparison. Use `\cite{TODO}` only if genuinely missing

9. **Previous writing flows** (for consistency of style):
   - `final report/final_phase2/introduction_flow.md`
   - `final report/final_phase3/methodology_flow.md`
   - Purpose: match the tone and paragraph density of §1 and §2 already in `main.tex`

10. **Report guidance & marking descriptors** (PDFs):
    - `final report/Final report Guidance 2026.pdf`
    - `final report/marking descriptors Final Report_2026.pdf`
    - Purpose: word-budget awareness (main sections ≤ 4000 words total), format requirements, marking criteria (IET 10 Laws of Good Report Writing: results must be clear, honest, interpreted)

11. **Skill available:**
    - `research-paper-writing` (in `.claude/skills/research-paper-writing`) — consult for paragraph flow, topic sentences, and anti-pattern avoidance when drafting §2.3

---

## Step 1: Write §2.3 Findings / Results in `main.tex`

Replace the existing skeleton/placeholder `\section{Findings}` (or `\section{Results}`) in `main.tex` with polished prose following `results_flow.md` beat-by-beat.

### Structure and paragraph mapping (from `results_flow.md` §5)

| Para | Subsection | Beat | Role | Content (one-sentence summary) | Figures / tables |
|:---:|---|:---:|---|---|---|
| 1 | §2.3 opener | R0 | Roadmap | Resolve 4 hypotheses from §2.2 in order; headline = Phase 3 step-change + Phase 4 confirmation | — |
| 2 | §2.3.1 Phase 1&2 | R1 | Evidence (baseline) | Config B sets the R² ≈ 0.90 ceiling; any OES model must beat this | `phase1_model_comparison_bar.png` |
| 3 | §2.3.1 Phase 1&2 | R2 | Evidence + Interpretation (H1 rejected) | PCA + OES actively degrades R²; structural failure, not cutoff artefact | `pca_cumulative_variance.png`, `phase1_predicted_vs_actual_grid.png` |
| 4 | §2.3.1 Phase 1&2 | R3 | Evidence + Interpretation (H2 rejected) | Optuna tuning closes part of the gap but cannot reach Config B from OES | *(inline only — no figure)* |
| 5 | §2.3.2 Phase 3 ← **longest** | R4 | Evidence + Interpretation (H3 supported) | 13 domain OES features collapse the B/C gap across every model class | `phase1_vs_phase2_vs_phase3_comparison.png` |
| 6 | §2.3.3 Phase 4 | R5 | Evidence (importance) | Consensus top-3: flow_rate, band_CO2p, pulse_width — bands beat single wavelengths | `fig1_feature_importance_heatmap.pdf` |
| 7 | §2.3.3 Phase 4 | R6 | Evidence + Honest uncertainty | Wide CIs at n=20; Ridge B / Ridge C overlap; Ridge C / MLP C overlap → linear suffices | `fig5_bootstrap_r2_distributions.pdf` |
| 8 | §2.3.3 Phase 4 | R7 + R8 | Evidence + Interpretation (H4 supported) | Permutation significance + two-strategy convergence on pruned model | `rank_on_model_config.png` + small inline permutation statement |
| 9 | §2.3.3 closer | R9 | Synthesis + Transition | Evidence chain closes; Ridge by Occam, not capability; hand off to §2.4 | — |

### Writing rules for Results

1. **Follow `results_flow.md` strictly** — do not invent new beats or re-order them. The backward reasoning chain (A→B→C) is what makes the section hypothesis-driven.
2. **One interpretive point per paragraph** — numbers are citations, not the subject (AP2). Every paragraph needs a topic sentence that states the claim, followed by the evidence.
3. **Frame Phase 1/2 as hypothesis tests, not failures** (AP1). Use phrases like "These results reject the hypothesis that…" — never "our PCA approach did not work".
4. **Do not re-explain methodology** (AP3). Cite subsection labels (`§\ref{sec:method}` or the specific subsection) instead of repeating bootstrap iterations, Optuna details, etc.
5. **Do not over-claim on n = 20** (AP4). Explicitly state that Ridge B and Ridge C bootstrap CIs overlap and that Phase 3's step-change is large in magnitude but only formally significant after pruning. This is R6's job — do not soften it.
6. **Phase 3 gets proportionally more space** (AP5). Paragraph 5 (R4) should be the longest paragraph in §2.3 and must include the centrepiece figure.
7. **Figures carry support, not argument** (AP6). Every figure must be cited by a sentence that states the interpretive point in words; the figure supports, not replaces, the claim.
8. **Report permutation p-value correctly**: write `$p < 0.0005$`, never `$p = 0$`. The CSV stores `0.0` but the correct bound at n = 2000 permutations is `< 1/2001 ≈ 0.0005`.
9. **Disambiguate the two "0.9x" results for the pruned model**: the category-ablation ratios-only R² = 0.906 and the permutation-tested pruned-Ridge R² = 0.920 refer to *different fits*. One sentence in R7 or R8 must make this explicit (e.g., "Refitting Ridge on just the three ratios plus the four discharge parameters raises R² to 0.920…").
10. **Mention the XGBoost anomaly once and move on**: in R1, note it as "XGBoost produced R² = −0.108 identically across all configurations, a default-hyperparameter artefact unsuitable for n = 20 with LOOCV, and is excluded from subsequent analysis." Do not revisit it.
11. **IEEE citation style**: use `\cite{optuna2019}` when first mentioning TPE tuning; reuse keys from §1/§2 only where genuinely useful. §2.3 should be citation-light — it is our own evidence.
12. **No first-person** ("I", "we" → "this project", "the results", passive voice) — consistent with §1 and §2 already in `main.tex`.
13. **Approximate length**: 700–1000 words for the entire §2.3 (the flow §5 budget). The main-sections word budget is ≤ 4000; §1 ≈ 550 and §2 ≈ 1000 are already drafted, leaving roughly 1500–2000 words for §2.3 + §2.4 + §2.5 + §2.6.

### Figure inclusion template (LaTeX)

All figures live in `final report/report/images/`. Use relative paths consistent with existing `\includegraphics` commands in `main.tex`. Suggested labels:

- `fig:pca_variance` — `pca_cumulative_variance.png`
- `fig:phase1_bar` — `phase1_model_comparison_bar.png`
- `fig:phase1_grid` — `phase1_predicted_vs_actual_grid.png` (optional, drop if word budget is tight)
- `fig:phase_comparison` — `phase1_vs_phase2_vs_phase3_comparison.png` (**centrepiece**)
- `fig:importance_heatmap` — `fig1_feature_importance_heatmap.pdf`
- `fig:bootstrap_dist` — `fig5_bootstrap_r2_distributions.pdf`
- `fig:model_ranking` — `rank_on_model_config.png`

If a figure include already exists in the skeleton with a different label, keep the existing label and update the filename rather than creating a duplicate.

---

## Step 2: Create supporting tables for §2.3

The Results section needs compact summary tables to avoid paragraphs dense with numbers. Create the following tables in `main.tex` (either inline in §2.3 or in an appendix if the word budget is tight):

### `tab:phase_comparison` — cross-phase R² table (new)
Compact 3-phase × key-models × configs table. Suggested columns: Model, Config, R² (Phase 1), R² (Phase 2 tuned), R² (Phase 3 domain). Rows: Ridge, PLS, MLP, CNN (key models only — full table can go in Appendix E). Anchor for R1, R2, R3, R4.

### `tab:bootstrap` — bootstrap 95% CI summary (new)
Columns: Model, Config, R² point, 95% CI, RMSE. Rows: Ridge B/C, MLP B/C at minimum; extend to PLS and RF if space allows. Anchor for R6. Source: `phase4/results/tables/bootstrap_ci_summary.csv`.

### `tab:feature_importance` — consensus feature importance top-8 (new)
Columns: Rank, Feature, Type (discharge / OES band / OES line / OES ratio), Mean rank. Anchor for R5. Source: `phase4/results/tables/feature_importance_all_models.csv`.

### `tab:ablation` — feature reduction summary (new)
Two sub-blocks: (a) category ablation (All 13 / Single-wavelength / Bands / Ratios / Discharge-only rows with R² and # features), and (b) backward-elimination peak row (1 OES feature + 4 discharge → R² = 0.918) plus the pruned-refit Ridge row (3 ratios + 4 discharge → R² = 0.920, p < 0.0005). Anchor for R7 and R8. Source: `phase4/results/tables/ablation_summary_article.csv`.

**Table rules:**
- Use `\label{}` consistent with existing table labels in `main.tex` (check `tab:configs`, `tab:oes_features` for style).
- Keep tables compact — overflowing tables hurt readability more than they help.
- All numbers must match the fact sheet in `results_flow.md` §7 (which was verified against the raw CSVs in Phase 5).
- Full tables (e.g., complete 13-step backward elimination) belong in appendices, not §2.3.

---

## Step 3: Quality checks

After writing, verify:

### Content
- [ ] §2.3 opens with a one-paragraph roadmap (R0) listing the four hypotheses and signalling the headline finding
- [ ] Every R² / RMSE / CI / p-value in §2.3 matches the fact sheet in `results_flow.md` §7
- [ ] Phase 1 and Phase 2 are framed as *hypothesis tests rejected*, not as "failures"
- [ ] Paragraph 5 (R4, Phase 3) is clearly the longest paragraph in §2.3 and contains the centrepiece figure
- [ ] R6 explicitly acknowledges bootstrap CI overlap between Ridge B and Ridge C (and between Ridge C and MLP C)
- [ ] The permutation p-value is written as `$p < 0.0005$`, never `$p = 0$`
- [ ] The ratios-only R² = 0.906 and the pruned-refit Ridge R² = 0.920 are distinguished as different fits
- [ ] XGBoost anomaly is mentioned once and then dropped
- [ ] No R² values are assigned to methods that were not described in §2.2
- [ ] All 4 anti-patterns from `results_flow.md` §4 are avoided (AP1 framing, AP2 method dump, AP3 re-explanation, AP4 over-claim, AP5 burying Phase 3, AP6 figure-as-argument)

### LaTeX
- [ ] All `\includegraphics{}` paths point to `final report/report/images/<filename>` (or the existing relative path convention in `main.tex`)
- [ ] All `\label{}` and `\ref{}` commands resolve (no undefined references)
- [ ] Tables compile: no missing `\\`, `&`, `\hline`, or column specifiers
- [ ] No `\cite{TODO}` placeholders unless a reference is genuinely missing (§2.3 should need very few citations)
- [ ] Figure captions carry interpretive text (1–2 sentences), not just titles

### Consistency with §1 and §2
- [ ] Results narrative does not contradict claims already made in §1 or §2
- [ ] "Domain knowledge is decisive" framing is preserved from §1 and §2.2.5
- [ ] Tone matches §1 and §2: passive voice, no first-person, IEEE citation style
- [ ] Section labels (`sec:results`, `sec:method`) are consistent with the existing `main.tex` pattern

---

## Deliverables

| # | Deliverable | Location | Description |
|---|---|---|---|
| 1 | Rewritten §2.3 Findings / Results | `final report/report/main.tex` §2.3 | ~9 paragraphs, ~700–1000 words, following R0–R9 beats |
| 2 | Supporting tables | `final report/report/main.tex` §2.3 | `tab:phase_comparison`, `tab:bootstrap`, `tab:feature_importance`, `tab:ablation` (or relevant subset) |
| 3 | Figure includes | `final report/report/main.tex` §2.3 | 6–7 `\includegraphics{}` blocks pointing to files in `images/` |

No changes to §1, §2, §2.4, §2.5, §2.6, references.bib, or the images directory in this phase.

---

## Quality Criteria

1. **Hypothesis resolution**: §2.3 resolves the four hypotheses from §2.2 in order — H1 (PCA), H2 (tuning), H3 (domain features), H4 (feature reduction) — and the reader can trace each resolution to a specific beat (R1/R2 → H1, R3 → H2, R4 → H3, R5–R8 → H4).
2. **Narrative centre of gravity**: Phase 3 (R4) is visually and rhetorically dominant — longest paragraph, centrepiece figure.
3. **Honest uncertainty**: the n = 20 ceiling is acknowledged explicitly in R6; the pruned model's significance (R7) is not conflated with the Phase 3 step-change (R4).
4. **Clean separation**: §2.3 contains evidence and interpretation only; no methodology re-derivation, no claims that belong in §2.4 Conclusions.
5. **Anti-patterns avoided**: AP1–AP6 from `results_flow.md` §4 all clear.
6. **Compilable LaTeX**: the file compiles without errors; all citations, labels, and figure paths resolve.

---

# Observation

## Summary

Phase 6 delivered a complete draft of §2.3 Findings / Results in `main.tex`, replacing the previous skeleton (three thin paragraphs with placeholder figure paths) with a 9-paragraph prose section organised around the R0–R9 beats from `results_flow.md`. The section now contains an opening roadmap, three sub-subsections (Phase 1&2, Phase 3, Phase 4), seven figures all sourced from `images/`, and four supporting tables (`tab:phase_comparison`, `tab:feature_importance`, `tab:bootstrap`, `tab:ablation`). The file compiles cleanly on the first pass (`pdflatex → bibtex → pdflatex → pdflatex`, 26 pages, zero `!` errors); the only warnings remaining are pre-existing `acro:*` references from the List of Abbreviations, which are outside Phase 6's scope.

## Draft quality — what worked

- **Hypothesis-driven arc held**: the opener explicitly states that §2.3 resolves the four hypotheses from §2.2 in order, and each sub-subsection delivers a rejection (H1, H2) or confirmation (H3, H4). Phase 1 and Phase 2 are framed as controlled tests that rejected specific hypotheses, not as failures — AP1 clear.
- **Phase 3 is proportionally weighted**: §2.3.2 is the longest paragraph in the section and owns the centrepiece figure (`phase1_vs_phase2_vs_phase3_comparison.png`). The prose explicitly names Phase 3 as "the central finding of the project" and interprets the step-change through two reinforcing observations (benefit across all model families; Ridge matches tuned NNs). AP5 clear.
- **Honest uncertainty is in the prose**: R6's paragraph explicitly calls out the Ridge B ↔ Ridge C CI overlap and the Ridge C ↔ MLP C overlap, frames these as a direct consequence of `n = 20`, and uses them to justify Occam's-razor model selection in the closing synthesis (R9). AP4 clear.
- **The 0.906 vs 0.920 disambiguation is explicit**: the pruned-refit Ridge paragraph in §2.3.3 now says verbatim "Refitting Ridge on this latter 7-feature set raises R² from the category-ablation value of 0.906 to 0.920", and the `tab:ablation` footnote reinforces the permutation test attaches to the refit model only. Phase 5 observation item (3) is resolved.
- **Critical numerical fixes from the skeleton**: the skeleton had `p < 0.00005` (too strong), MLP C tuned `0.33` (should be 0.369), and MLP C Phase 3 `0.80` (should be 0.815) — all corrected. Figure paths also changed from the empty `figures/` directory to the actual `images/` directory. These were latent errors in the old skeleton that would have propagated if untouched.
- **XGBoost handled once and dropped**: the anomaly (R² = −0.108 identical across configs) is mentioned in one sentence at the end of the §2.3.1 opening paragraph and never revisited. Phase 5 observation item (4) is resolved.
- **Figure captions carry interpretation**: every figure's caption makes a one-sentence interpretive point, not just a title. AP6 clear.
- **Methodology not re-derived**: the prose cites `\S\ref{sec:method}` rather than re-explaining LOOCV, PCA, or bootstrap mechanics. AP3 clear.

## Draft quality — self-critique and weaknesses

- **Length**: the section runs long — my rough estimate is ≈1100–1300 words, slightly over the 700–1000 target in `results_flow.md`. Candidate trims: the §2.3.1 Optuna paragraph (R3) could drop to 2 sentences since it is a bridging beat; the second half of the Phase 4 synthesis is partially redundant with §2.4 Conclusions and could be compressed.
- **Density of tables**: four tables plus seven figures in one section is on the upper edge of what the page budget can absorb. The CI table (`tab:bootstrap`) and the bootstrap distribution figure (`fig:bootstrap_dist`) make overlapping points; one could be moved to Appendix E if the total report exceeds the 4000-word budget.
- **Phase 2 handling**: per the user's Plan, Phase 2 received no dedicated figure. I folded its results into the final paragraph of §2.3.1 as inline prose — this works but means the R3 beat is structurally subordinate to R2 rather than a standalone paragraph. Acceptable but worth noting.
- **Permutation test reporting**: reported as "no null R² as large as the observed value, corresponding to p < 0.0005" rather than a dedicated inline table. This is the minimal form consented in the flow but is compact enough that a reader skimming for the significance claim may miss it.
- **Figure–table overlap in §2.3.3**: Table~\ref{tab:feature_importance} and Figure~\ref{fig:importance_heatmap} both land the consensus ranking; the bootstrap table and figure do the same for CIs. In a tighter edit pass, one of each pair could be dropped — the tables are the more precise artefacts and the figures are the more scannable ones, so the choice is stylistic rather than substantive.
- **Cross-section consistency spot check**: §1 claims the project bridges a gap by demonstrating domain knowledge is decisive; §2.2.5 emphasises the 13-feature design as the central contribution; §2.3.2 lands the claim quantitatively. These three sections align. §2.4 Conclusions has not been reviewed yet — when it is updated in a later phase, the `R² = 0.920` and "7-feature Ridge" numbers must match §2.3 exactly.

## Consistency flags for later phases

1. **Existing §2.4 Conclusions** still carries skeleton values from an earlier draft (e.g., `R² = 0.80` in the §1 objectives, `best model R² = 0.920` in the outline). A later phase must reconcile §2.4's specific numbers with §2.3's verified fact sheet — particularly the distinction between the 13-feature Ridge C (0.798) and the pruned 7-feature Ridge (0.920).
2. **Appendix C (OES Feature Derivation) and Appendix E (Full Statistical Test Outputs)** are still `\todo{}` placeholders. §2.3 now references `Appendix~\ref{app:stats}` for the full cross-phase table; that appendix needs to be populated in a later phase or the reference softened.
3. **`figures/` directory is empty**; all actual figures live in `images/`. The skeleton's `figures/uol_logo.png` and `figures/r2_comparison.png` paths (still present in lines 108 and originally in the old §2.3) were only partially cleaned up — line 108's `uol_logo.png` still points at the empty directory and remains an `IfFileExists` placeholder. Not Phase 6's scope to fix, but worth flagging.
4. **Wide bootstrap CIs at n=20** must be echoed, not contradicted, in the §2.4 Conclusions. If §2.4 claims "domain features are statistically significantly better than PCA", that would overclaim relative to §2.3's R6 paragraph — the correct formulation is "domain features produce a step-change in point estimates and, after pruning, deliver a permutation-confirmed result".
5. **No new citations added in §2.3 beyond `optuna2019`** (already in `references.bib`); §2.3 is citation-light by design. If the marker expects more literature comparison in the results (e.g., "our R² = 0.920 compares to Wang2025's R² ≈ 0.90–0.97"), a later phase could add a single comparison sentence in §2.4 rather than §2.3.

## Risks / caveats

- **Not yet visually proofread**: I relied on a clean compilation as the correctness signal, but I have not opened the resulting PDF to check table alignment, figure sizing, or page breaks. Long tables like `tab:ablation` may wrap awkwardly; figures at `0.86--0.92\textwidth` may push below the page break. The user should skim the PDF pages covering §2.3 before the next phase.
- **Word budget risk**: §1 ≈ 550, §2 ≈ 1000, §2.3 ≈ 1200 = ≈ 2750 words already. Remaining budget for §2.4 Conclusions, §2.5 Future Work, and §2.6 Reflection is ≈ 1250 words — tight but workable. If §2.3 needs trimming to make room, the Optuna bridging paragraph (R3) is the first candidate.
- **No `main.tex` visual inspection before/after**: the edit was substantial (≈220 lines replaced in one Edit call). Although LaTeX compiled cleanly, subtle prose errors (e.g., inconsistent figure label casing, stray non-ASCII dashes) may only surface on careful read-through.
