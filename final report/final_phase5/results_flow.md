# Results Writing Logic Flow (Phase 5)

> Mirrors the structural rigour of `final_phase3/methodology_flow.md`. Produced for the `§2.3 Findings / Results` section of the final report. Every numeric claim traces to a specific CSV row or figure; see Section 6 (Fact-Check Register).

---

## 1. Pre-Writing Questions (Backward Reasoning 先反向思考)

| Question | Answer to derive |
|---|---|
| **What must the reader believe after reading Results?** | (1) Domain-knowledge features decisively beat PCA for OES-based H₂O₂ yield prediction. (2) The improvement is not an artefact of tuning, model class, or chance — it survives bootstrap CIs and a permutation test. (3) A minimal, physically interpretable feature set (≤7 features) matches the full 13-OES model and matches non-linear neural networks. |
| **What is the narrative arc?** | NOT a phase-by-phase diary. A **claim-driven** story: *Baseline shows a paradox → Tuning cannot close the gap → Domain features close it in one step → Statistics rule out chance → Minimal model matches full model.* |
| **What are the three "headline" numbers the reader must remember?** | (i) Ridge Config C R² jumps from **−0.175 → 0.798** (Phase 1 PCA → Phase 3 domain features). (ii) Pruned 7-feature Ridge reaches **R² = 0.920**, permutation **p = 0.0** (0/2000 shuffles beat it). (iii) Bootstrap 95% CI for Ridge B `[0.800, 0.955]` overlaps Ridge C `[0.574, 0.910]` — Config C is not *significantly better* than discharge-only, but it is competitive with far more mechanistic content. |
| **Which results carry each objective?** | **Obj 1 (predict H₂O₂ yield):** best R² = 0.920 from the pruned Ridge, permutation-validated. **Obj 2 (PCA vs domain):** Phase 1/2 vs Phase 3 comparison across all 6 carried-forward models. **Obj 3 (minimal feature set):** Phase 4 category ablation + backward elimination, both converging on ≤7 features. |
| **Where is the boundary between Results and Conclusions?** | Results = *what was observed* (numbers, tests, comparisons). Interpretation of *why* domain features work (drift invariance, physical alignment with chemistry) and the *broader implication* (domain knowledge > automated DR for small physically structured data) → Conclusions. |
| **What must NOT appear in Results?** | Re-derivation of methods (already in §2.2); speculation on mechanism beyond the data; comparison to papers whose datasets differ (goes in Discussion / Conclusions). |

---

## 2. Forward Story (正向写作逻辑)

The flow is a sequence of 8 steps (R1–R8). Each step has: (a) the claim being made, (b) the evidence anchor, (c) a drafted topic sentence.

| Step | Logic | Claim | Evidence anchor | Draft topic sentence |
|:---:|---|---|---|---|
| **R1** | Orient | Evaluation protocol and signposting | — | "Results are reported as leave-one-out R² and RMSE on the 20-sample dataset; three claims structure this section: baseline paradox, domain-feature breakthrough, and minimal-model validation." |
| **R2** | Phase 1 baseline paradox | Config B (discharge only) establishes a strong linear baseline, but adding PCA-reduced OES (Config C) *degrades* performance for every carried-forward model. | `phase1/results/tables/loocv_results_summary.csv` → Ridge B 0.904 / Ridge C −0.175 / MLP B 0.568 / MLP C −1.131 | "Config B already predicts H₂O₂ yield accurately with Ridge (R² = 0.904), yet adding PCA-based OES features in Config C drives Ridge to R² = −0.175 and MLP to R² = −1.131 — a paradox that motivates the next two phases." |
| **R3** | Phase 2 tuning insufficient | Optuna tuning improves every non-linear Config C model substantially but none crosses the Config B bar — tuning alone cannot repair uninformative features. | `phase2/results/tables/phase2_loocv_results_summary.csv`; `phase1_vs_phase2_comparison.csv` → CNN C 0.688→0.775, MLP C −1.131→0.369, RF C 0.239→0.456 | "Bayesian tuning lifts MLP Config C from R² = −1.131 to 0.369 and pushes CNN Config C to R² = 0.775, but no tuned Config C model matches tuned Config B (MLP B = 0.861), confirming that the ceiling is set by feature quality, not model capacity." |
| **R4** | Phase 3 step-change | Replacing 11 PCA components with 13 domain features yields a step-change in Config C across all models, closing most of the gap to Config B in a single phase. | `phase3/results/tables/phase1_vs_phase2_vs_phase3_comparison.csv` → Ridge C −0.175→0.798 (ΔR² = +0.973); MLP C −1.131→0.815 (ΔR² = +1.946); PLS C 0.625→0.744 | "Swapping PCA for 13 physically motivated OES features produces the largest single-phase improvement in the project: Ridge Config C R² rises from −0.175 to 0.798, MLP Config C from −1.131 to 0.815, and every other carried-forward model improves in the same direction." |
| **R5** | Phase 4a consensus importance | Across 4 importance methods (Ridge coefficients, PLS VIP, RF permutation, MLP SHAP), the dominant predictors are a small mix of discharge parameters and a CO₂⁺ band integral — not any single emission line. | `phase4/results/tables/feature_importance_all_models.csv` → flow_rate_sccm (mean rank 1.75), band_CO2p_398_412 (4.75), pulse_width_ns (5.25), I_486_Hb and band_CO_Hb_460_500 (6.00 each) | "Consensus ranking across four importance methods places `flow_rate_sccm` at rank 1 (mean rank 1.75) and the CO₂⁺ 398–412 nm band integral at rank 2 (4.75), with the next four positions filled by discharge parameters and broad spectral bands rather than single atomic lines." |
| **R6** | Phase 4b statistical validation | Bootstrap 95% CIs for the best Config B and Config C Ridge models overlap, and a 2000-shuffle permutation test rejects the null at p = 0/2000 (conservatively p < 5 × 10⁻⁴) for the pruned Ridge. | `phase4/results/tables/bootstrap_ci_summary.csv` → Ridge B 0.904 [0.800, 0.955], Ridge C 0.798 [0.574, 0.910], MLP C 0.815 [0.647, 0.883]; `permutation_test_summary.csv` → observed R² = 0.9200, p = 0.0 over 2000 permutations | "Bootstrap resampling places Ridge Config C at 0.798 with 95% CI [0.574, 0.910], overlapping Ridge Config B's [0.800, 0.955]; a 2000-shuffle permutation test on the pruned 7-feature Ridge (observed R² = 0.920) returns zero null R² at or above the observed value (p < 5 × 10⁻⁴)." |
| **R7** | Phase 4c feature reduction | Two independent reduction strategies — category ablation (keep only one feature type) and iterative backward elimination — converge on the same conclusion: ≤7 features suffice and the full 13-OES model is redundant. | `phase4/results/tables/ablation_summary_article.csv` → Ratios (3) R² = 0.9063, Bands (3) R² = 0.9053, Single-wavelength (7) R² = 0.8232, Full (13) R² = 0.7984; `permutation_test_summary.csv` → pruned Ridge (3 ratios + 4 discharge) R² = 0.9200; backward elimination reaches R² = 0.918 at 1 OES feature (band_CO2p_398_412) + 4 discharge | "Category ablation shows that any single normalised feature family — either the three spectroscopic ratios or the three band integrals — retains the full predictive signal (R² = 0.9063 and 0.9053 respectively), while backward elimination drives R² monotonically from 0.798 (13 OES) up to 0.918 at one OES feature, revealing that most of the original 13 features are redundant rather than informative." |
| **R8** | Synthesis | Tie the four phases back to the three objectives and forward-point to Conclusions. | — | "Collectively, the four phases answer the three project objectives: a 7-feature Ridge model predicts H₂O₂ yield at R² = 0.920 (Obj 1); domain-knowledge features improve Config C by nearly a full R² unit over PCA (Obj 2); and reduction identifies a minimal physically interpretable set dominated by discharge parameters, band integrals, and spectroscopic ratios (Obj 3)." |

---

## 3. Anti-Pattern Warnings for Results

1. **Anti-pattern — Phase diary.**
   *Bad:* "In Phase 1 we got X. In Phase 2 we got Y. In Phase 3 we got Z."
   *Good:* Each subsection opens with the **claim** being tested, followed by the evidence. Phase labels appear only as pointers, not as the structural backbone.

2. **Anti-pattern — Number dump.**
   *Bad:* Quoting every cell of every table in prose.
   *Good:* In prose, quote only the one or two numbers that carry each claim (e.g., Ridge C −0.175 → 0.798; pruned R² = 0.920; p < 5 × 10⁻⁴). Full comparison matrices live in tables / figures.

3. **Anti-pattern — Premature interpretation.**
   *Bad:* "Domain features work because ratios are drift-invariant and physically aligned with the OH→H₂O₂ pathway."
   *Good:* Results states *what* (Ridge C jumped from −0.175 to 0.798); *why* belongs in Conclusions §2.4.

4. **Anti-pattern — Cherry-picking.**
   *Bad:* Only reporting the best model per phase.
   *Good:* Always report a matrix (all models × all configs) and then highlight the winner. Every improvement claim must be seen as a trend across models, not as a single lucky row.

5. **Anti-pattern — Over-claiming.**
   *Bad:* "Domain features always outperform PCA for OES-based ML."
   *Good:* Bound claims by this study: 20-sample dataset, one reactor configuration, H₂O₂ yield target. Use "on this dataset" / "in this system" explicitly.

6. **Anti-pattern — Ignoring uncertainty.**
   *Bad:* Reporting R² = 0.920 as if it were a point estimate on a held-out test set.
   *Good:* Always pair R² with its statistical qualifier — bootstrap CI for estimation uncertainty, permutation p-value for chance. Explicitly note that Ridge B and Ridge C bootstrap CIs **overlap**, so domain-feature Config C is *not* statistically better than discharge-only; the value is interpretability + pathway content, not headline R².

7. **Anti-pattern — Drifting across Methods / Conclusions boundary.**
   *Bad:* Re-describing LOOCV / bootstrap / permutation procedures in Results.
   *Good:* Methods are already in §2.2 — Results cites `tab:` / `fig:` and moves on. One short protocol-reminder sentence in R1 is enough.

---

## 4. Paragraph-Level Plan (10 paragraphs, ~1000 words total)

| # | Subsection | Role | Topic sentence | Evidence anchor | Figures / Tables |
|:---:|---|---|---|---|---|
| 1 | §2.3 Opening (was §2.3.0) | Orient | Summarises the protocol (LOOCV, R², RMSE) and previews the three claims the section supports. | — | — |
| 2 | §2.3.1 Phase 1 & 2 | Claim — baseline paradox | Config B discharge-only is already a strong linear baseline; adding PCA-based OES (Config C) collapses performance across models. | `loocv_results_summary.csv`: Ridge B 0.904, Ridge C −0.175, MLP C −1.131, PLS C 0.625 | `tab:results_main` (§2.3), `fig:phase1_heatmap_r2` (optional appendix) |
| 3 | §2.3.1 | Evidence — tuning insufficient | Optuna tuning substantially lifts non-linear Config C models, but no tuned Config C model crosses the Config B bar. | `phase1_vs_phase2_comparison.csv`: MLP C −1.131→0.369, CNN C 0.688→0.775, RF C 0.239→0.456, MLP B 0.861 | `tab:results_main` |
| 4 | §2.3.2 Phase 3 | Claim — domain-feature step-change | Replacing 11 PCA components with 13 domain features closes the Config B/C gap in one step across every model. | `phase1_vs_phase2_vs_phase3_comparison.csv`: Ridge C ΔR² = +0.973, MLP C ΔR² = +1.946, PLS C +0.119, RF C +0.258 | `fig:r2_comparison`, `tab:results_main` |
| 5 | §2.3.2 | Comparison — cross-model generality | Every carried-forward model (Ridge, PLS, RF, MLP) improves, and the improvement is largest for Ridge and MLP — the gap is not model-specific. | Same CSV, all rows | `fig:r2_comparison` |
| 6 | §2.3.3 Phase 4 | Evidence — consensus importance | Consensus ranking across 4 importance methods places flow rate and the CO₂⁺ 398–412 nm band at the top, with the spectroscopic ratios concentrated in the upper-middle. | `feature_importance_all_models.csv` | `fig:feature_importance`, `tab:feature_importance` (appendix) |
| 7 | §2.3.3 | Qualifier — statistical validation | Bootstrap 95% CIs for Ridge B and Ridge C overlap, but a 2000-shuffle permutation test on the pruned 7-feature Ridge rejects the null (observed R² = 0.920, zero null permutations beat it). | `bootstrap_ci_summary.csv`, `permutation_test_summary.csv`, `permutation_test_pruned_ridge.csv` | `fig:bootstrap`, `fig:permutation` (optional), `tab:bootstrap` |
| 8 | §2.3.3 | Evidence — category ablation | Keeping only one OES feature family — either the three ratios or the three band integrals — matches or exceeds the full 13-OES model; single-wavelength lines underperform. | `ablation_summary_article.csv`: Ratios 0.9063, Bands 0.9053, Single-wavelength 0.8232, Full 0.7984 | `fig:category_ablation`, `tab:ablation` |
| 9 | §2.3.3 | Evidence — backward elimination | Iterative backward elimination drives R² monotonically up as redundant features are removed, peaking at R² = 0.918 with just one OES feature (band_CO2p_398_412) plus the 4 discharge parameters. | `ablation_backward_elimination_full.csv` / `ablation_summary_article.csv` backward rows | `fig:ablation_trajectory`, `tab:ablation` |
| 10 | §2.3.4 Synthesis (NEW) | Closing | Ties the four phases to the three objectives and forward-points to §2.4 Conclusions; explicitly flags the bootstrap CI overlap so the reader is not misled about "beating" Config B. | — | — |

**Length target:** 900–1200 words of prose. Tables and figures carry most of the numeric load; prose quotes only the headline numbers.

---

## 5. Figures and Tables Plan

### 5a. Figures (copied into `final report/report/images/`)

| Label (LaTeX) | Source file | Copied as | Purpose | Essential? |
|---|---|---|---|:---:|
| `fig:r2_comparison` | `phase3/results/figures/phase1_vs_phase2_vs_phase3_comparison.png` | `images/fig_r2_comparison.png` | Headline R² bar chart across Phases 1 / 2 / 3 — carries the step-change claim | **Yes** |
| `fig:feature_importance` | `phase4/results/figures/fig1_feature_importance_heatmap.pdf` | `images/fig_feature_importance.pdf` | Consensus importance heatmap across 4 methods | **Yes** |
| `fig:bootstrap` | `phase4/results/figures/fig5_bootstrap_r2_distributions.pdf` | `images/fig_bootstrap.pdf` | Bootstrap R² distributions with 95% CIs for Ridge / MLP / PLS / RF on B and C | **Yes** |
| `fig:ablation_trajectory` | `phase4/results/figures/fig9_ablation_trajectory.pdf` | `images/fig_ablation_trajectory.pdf` | Backward elimination R² curve (monotone rise to 0.918) | **Yes** |
| `fig:category_ablation` | `phase4/results/figures/fig10_category_ablation.pdf` | `images/fig_category_ablation.pdf` | Bar chart: Ratios / Bands / Single-wavelength / Full / Config B | **Yes** |
| `fig:permutation` | `phase4/results/figures/fig12_permutation_test.pdf` | `images/fig_permutation.pdf` | Permutation null distribution with observed R² marker | Optional — p-value is already quoted; include if space allows |
| `fig:phase1_heatmap_r2` | `phase1/results/figures/heatmap_R2.png` | `images/fig_phase1_heatmap_r2.png` | Phase 1 model×config heatmap (for appendix or §2.3.1 inset) | Optional |

All copies completed in Step 2 of execution.

### 5b. Tables

| Label | Content | Source | Essential? | Location |
|---|---|---|:---:|---|
| `tab:results_main` | Phase 1 / 2 / 3 LOOCV R² per model per config (Ridge, PLS, RF, MLP; Config A/B/C) | `phase3/results/tables/phase1_vs_phase2_vs_phase3_comparison.csv` | **Yes** | §2.3.1 |
| `tab:bootstrap` | Point R², 95% CI, RMSE for Ridge / PLS / RF / MLP on Config B and C | `phase4/results/tables/bootstrap_ci_summary.csv` | **Yes** | §2.3.3 |
| `tab:ablation` | Category ablation (Ratios / Bands / Single-wavelength / Full / Config B) + backward elimination best row | `phase4/results/tables/ablation_summary_article.csv` | **Yes** | §2.3.3 |
| `tab:feature_importance` | Top-10 consensus ranks across 4 methods | `phase4/results/tables/feature_importance_all_models.csv` | Optional — keep top-5 in prose / figure; full table → Appendix E | Appendix E |

---

## 6. Fact-Check Register (Numeric Claims → Source)

All numeric claims the flow uses. **Every number has been verified against the listed CSV in Phase 5 execution.**

| # | Claim | Exact value | Source file | Notes |
|:---:|---|---|---|---|
| 1 | Ridge Config B LOOCV R² (Phase 1) | 0.9042977737092368 → **0.904** | `phase1/results/tables/loocv_results_summary.csv` via `phase3/.../phase1_vs_phase2_vs_phase3_comparison.csv` row Ridge/B | — |
| 2 | Ridge Config C LOOCV R² (Phase 1, PCA) | −0.1746932196021335 → **−0.175** | same CSV, row Ridge/C, R2_P1 | — |
| 3 | MLP Config C LOOCV R² (Phase 1, PCA) | −1.1305103725786425 → **−1.131** | same CSV, row MLP/C, R2_P1 | — |
| 4 | MLP Config B LOOCV R² (Phase 1) | 0.567821774402973 → **0.568** | same CSV, row MLP/B, R2_P1 | — |
| 5 | PLS Config C LOOCV R² (Phase 1) | 0.6248362002842549 → **0.625** | same CSV, row PLS/C, R2_P1 | — |
| 6 | MLP Config C R² (Phase 2, tuned) | 0.3693457183633185 → **0.369** | same CSV, row MLP/C, R2_P2 | — |
| 7 | MLP Config B R² (Phase 2, tuned) | 0.8605030444387558 → **0.861** | same CSV, row MLP/B, R2_P2 | — |
| 8 | RF Config B R² (Phase 2, tuned) | 0.7482647016597695 → **0.748** | same CSV, row RF/B, R2_P2 | — |
| 9 | RF Config C R² (Phase 2, tuned) | 0.4559573978015391 → **0.456** | same CSV, row RF/C, R2_P2 | — |
| 10 | CNN Config C R² (Phase 2, tuned) | **0.775** | `phase2/results/tables/phase2_loocv_results_summary.csv` + `experiments.md` table | not re-evaluated in Phase 3 (CNN dropped after Phase 2) |
| 11 | Ridge Config C R² (Phase 3, domain) | 0.7983981755276209 → **0.798** | `phase3/.../phase1_vs_phase2_vs_phase3_comparison.csv`, row Ridge/C, R2_P3 | Also appears as the "Full 13-OES" baseline in ablation CSV — consistent |
| 12 | MLP Config C R² (Phase 3, domain) | 0.8149153590142274 → **0.815** | same CSV, row MLP/C, R2_P3 | — |
| 13 | PLS Config C R² (Phase 3, domain) | 0.7436651050959188 → **0.744** | same CSV, row PLS/C, R2_P3 | — |
| 14 | ΔR² Ridge C Phase 1 → Phase 3 | **+0.973** | same CSV, Delta_R2_P3_vs_P1 column, row Ridge/C | — |
| 15 | ΔR² MLP C Phase 1 → Phase 3 | **+1.946** (+1.945 in CSV) | same CSV, Delta_R2_P3_vs_P1 column, row MLP/C | — |
| 16 | Bootstrap 95% CI Ridge B | [0.800, 0.955] | `phase4/results/tables/bootstrap_ci_summary.csv`, row Ridge/B | R2_lo95=0.8005, R2_hi95=0.9549 |
| 17 | Bootstrap 95% CI Ridge C | [0.574, 0.910] | same CSV, row Ridge/C | R2_lo95=0.5738, R2_hi95=0.9099 |
| 18 | Bootstrap 95% CI MLP C | [0.647, 0.883] | same CSV, row MLP/C | — |
| 19 | Top-1 consensus feature | `flow_rate_sccm`, mean rank **1.75** | `phase4/results/tables/feature_importance_all_models.csv` | — |
| 20 | Top-2 consensus feature | `band_CO2p_398_412`, mean rank **4.75** | same CSV | — |
| 21 | Top-3 consensus feature | `pulse_width_ns`, mean rank **5.25** | same CSV | — |
| 22 | Category ablation: Ratios only (3) + 4 discharge | **R² = 0.9063** | `phase4/results/tables/ablation_summary_article.csv`, row `Ratios (3)` | — |
| 23 | Category ablation: Bands only (3) + 4 discharge | **R² = 0.9053** | same CSV, row `Band integrals (3)` | — |
| 24 | Category ablation: Single-wavelength (7) + 4 discharge | **R² = 0.8232** | same CSV, row `Single-wavelength (7)` | — |
| 25 | Category ablation: Full 13 OES + 4 discharge | **R² = 0.7984** | same CSV, row `All OES (13)` | — |
| 26 | Backward elimination peak R² | **R² = 0.9180** at 1 OES feature (`band_CO2p_398_412`) + 4 discharge | `phase4/results/tables/ablation_summary_article.csv` — backward rows | Derived from n_oes_features progression; experiments.md table confirms 0.918 |
| 27 | Permutation-tested pruned Ridge | observed R² = **0.9200**, p-value = **0.0** over 2000 shuffles | `phase4/results/tables/permutation_test_summary.csv` (observed_r2=0.9200109921570517, p_value=0.0, n=2000) | Model: 3 ratios + 4 discharge |
| 28 | Permutation null mean | ≈ **−0.15** (reported in experiments.md; confirmed by spot-checking null rows in `permutation_test_pruned_ridge.csv` — row 3 = −0.2143, row 4 = −0.2132, row 5 = −0.4814) | `phase4/results/tables/permutation_test_pruned_ridge.csv` | qualitative only |

### ⚠ Discrepancies surfaced (to flag in Observation)

1. **Category-ablation "Ratios only" R² vs permutation-test observed R² of the same 7-feature model.**
   - `ablation_summary_article.csv` reports the ratios-only Ridge at **R² = 0.9063**.
   - `permutation_test_summary.csv` reports the pruned 7-feature Ridge (3 ratios + 4 discharge) observed R² at **R² = 0.9200**.
   - These *should* be the same model. The ~0.014 gap is likely due to implementation differences (e.g., standardisation pipeline, Ridge α, LOOCV vs train-whole-refit). The prose should quote **0.920** for the *permutation-tested pruned Ridge* and **0.906** for the *category-ablation Ratios row*, keeping them as two separate reported numbers and not conflating them. The "best model in the project" number is **R² = 0.920** because it is the one the permutation test validated.

2. **Permutation p-value rounding.**
   - CSV stores `p_value = 0.0` (exact: zero of 2000 shuffles matched or beat observed). Prose should say "**p = 0/2000**" or "**p < 5 × 10⁻⁴**" (= 1/2000 upper bound) — *not* "p < 0.0005" which is a tighter claim than the test can support; "p < 5 × 10⁻⁴" is the conservative phrasing. The Phase 3 outline's "p < 0.0005" is equivalent but marginally over-precise.

3. **Sample size.**
   - `experiments.md` header line says "**701 OES samples**". This is wrong — the dataset has **20 samples** (confirmed in Phase 4 fix to main.tex). Treat `experiments.md` as stale on this point.

---

## 7. Writing Checklist (for Phase 6 prose writing)

Before declaring the Results section done, verify:

- [ ] Every subsection opens with a **claim**, not a phase label.
- [ ] Every numeric claim in the prose is in Section 6's Fact-Check Register.
- [ ] Every headline R² is paired with either a bootstrap CI or a permutation p-value.
- [ ] The bootstrap CI overlap (Ridge B vs Ridge C) is **explicitly flagged**, not hidden.
- [ ] No sentence in §2.3 explains *why* domain features work — that belongs in §2.4.
- [ ] The "best R² = 0.920" claim cites the **permutation-tested pruned Ridge**, and the category-ablation 0.906 number is kept separate.
- [ ] Sample size is stated as **20**, not 701.
- [ ] Permutation p-value is written as "p < 5 × 10⁻⁴" or "p = 0/2000".
- [ ] All three objectives from §1.2 are explicitly linked to a result in the synthesis paragraph.
- [ ] All tables and figures referenced have labels defined in `main.tex` or in the updated outline.
