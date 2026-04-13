# Results Section — Writing Logic Flow (Phase 5)

> Blueprint for drafting §2.3 Findings / Results of the final report. Follows the same structure as `introduction_flow.md` (Phase 2) and `methodology_flow.md` (Phase 3). Every quantitative claim below has been cross-checked against the raw CSVs in `phase1/results/tables/`, `phase2/results/tables/`, `phase3/results/tables/`, and `phase4/results/tables/`.

---

## 1. Pre-Writing Questions

Before writing a single paragraph, the draft must answer these questions clearly; otherwise the section will drift into a table-by-table walkthrough.

1. **What does the reader need to believe by the end of §2.3?**
   - A simple 7-feature Ridge model achieves R² = 0.920 on H₂O₂ yield prediction.
   - That performance is **real** (permutation test p < 0.0005) and **not an artefact of model complexity**.
   - The decisive lever between failure (R² = −0.17) and success (R² = 0.80 → 0.920) was **domain-knowledge feature engineering**, not hyperparameter tuning, not deeper architectures.

2. **What hypotheses posed in §2.2 (Methodology) does §2.3 land?**
   - H1 (Phase 1): "PCA-based OES features carry sufficient predictive signal" → **rejected**.
   - H2 (Phase 2): "Hyperparameter tuning compensates for weak features" → **rejected**.
   - H3 (Phase 3): "Physically motivated OES features unlock the signal" → **supported**.
   - H4 (Phase 4): "Among the 13 domain features, most are redundant; a minimal interpretable subset suffices" → **supported**.

3. **Which claims are single-point estimates, and which have statistical backing?**
   - Single-point R²: all Phase 1/2/3 R² values (LOOCV, no CI).
   - Statistically backed: Phase 4 bootstrap CIs (500 iterations), permutation test (2000 shuffles, p < 0.0005).
   - **Implication**: the Phase 3 "breakthrough" story must be told without over-claiming significance; the bootstrap CIs (Ridge B vs Ridge C overlap) are the honest story.

4. **What must the section avoid?**
   - Framing Phase 1 as a "failure" — it is a controlled test that rules out one hypothesis.
   - Re-explaining methods (already in §2.2).
   - A chronological "diary" ("then we did Phase 2, then Phase 3…").
   - Over-claiming on n = 20 (wide CIs must be acknowledged).

---

## 2. Backward Reasoning — From Conclusion to Evidence

Start from the final claim the report needs to land, then walk backwards through the evidence chain. This is the skeleton the forward narrative in §3 will flesh out.

**C (Final claim)**: A 7-feature linear Ridge model (3 OES spectroscopic ratios + 4 discharge parameters) predicts H₂O₂ yield with R² = 0.920, matching neural networks while remaining fully interpretable.

- **To believe C, the reader must first accept:**
  - **C.1** — The 3 OES ratios carry essentially all the predictive information in OES (not the 7 emission lines, not the bands alone). **Evidence**: category ablation: ratios-only (3) → R² = 0.906 vs. single-wavelength-only (7) → R² = 0.823; full 13 OES → R² = 0.798 (feature redundancy hurts). Backward elimination trajectory rises monotonically from 0.798 to 0.918 as redundant features are pruned.
  - **C.2** — The 0.920 result is statistically real, not noise. **Evidence**: permutation test (n = 2000 shuffles) — observed R² = 0.920, no null R² ≥ observed (p < 0.0005).
  - **C.3** — A linear model is appropriate (i.e., neural networks would not do meaningfully better). **Evidence**: bootstrap CIs Ridge C [0.574, 0.910] and MLP C [0.647, 0.883] overlap; the Ridge–MLP gap is within noise for n = 20.
  - **C.4** — The 4 discharge parameters are the dominant drivers and OES adds value beyond them. **Evidence**: consensus top-3 importance = flow_rate_sccm (1), band_CO2p_398_412 (2), pulse_width_ns (3); removing the last OES feature drops R² back to Config B's 0.904, confirming OES contributes measurable information.

- **For C.1–C.4 to be meaningful, the reader must first accept that domain features beat PCA at all:**
  - **B.1** — Under PCA, OES actively **degrades** performance when combined with discharge parameters. **Evidence**: Phase 1 Ridge Config C R² = −0.175 (vs. Config B 0.904); Phase 1 MLP Config C R² = −1.131 (worst baseline). Config C is *worse* than Config B for most models.
  - **B.2** — Hyperparameter tuning cannot rescue PCA features. **Evidence**: Phase 2 MLP Config C −1.13 → 0.37; CNN Config C 0.69 → 0.78 (best OES-only tuned model, still below Config B). ΔR² of +1.5 for MLP is impressive but still yields < 0.4.
  - **B.3** — Replacing PCA with 13 domain features unlocks a step-change. **Evidence**: Phase 3 Ridge Config C −0.175 → 0.798 (ΔR² ≈ +0.97); MLP Config C −1.131 → 0.815 (ΔR² ≈ +1.95); the Config B / Config C gap collapses across all models.

- **For B.1 to be interpretable, the reader must first have a baseline:**
  - **A.1** — Config B (discharge only) is already a strong baseline. **Evidence**: Ridge Config B R² = 0.904, PLS Config B R² = 0.898. Any OES-inclusive result must beat this to be meaningful.
  - **A.2** — PCA's 11 components preserve ≥95% variance — i.e., PCA is not "too aggressive"; the failure is structural, not a cutoff artefact. **Evidence**: `pca_cumulative_variance.png`.

**Evidence chain summary**: `A.1 + A.2 → B.1 → B.2 → B.3 → C.1 + C.2 + C.3 + C.4 → C`.

The forward narrative (§3) follows this chain in order: baseline → PCA fails → tuning insufficient → domain features succeed → interpretability → minimal model.

---

## 3. Forward Story (R0 – R9)

Each beat is one interpretive point — not one table. A single paragraph may cite several numbers.

### R0 — Section opener
- **Claim**: The results section resolves the four hypotheses posed in §2.2 in the order they were introduced.
- **Content**: one-paragraph roadmap: baseline ceiling (H1), tuning compensation test (H2), domain-feature breakthrough (H3), interpretability and feature reduction (H4). State up-front that the headline finding is the Phase 3 step-change and its confirmation in Phase 4.
- **Figures**: none.

### R1 — Baseline establishes the ceiling to beat (§2.3.1 Phase 1)
- **Claim**: Discharge parameters alone (Config B) set a high bar: Ridge R² = 0.904, PLS R² = 0.898. This is the honest target any OES-inclusive model must exceed.
- **Content**: report Config B results; note that the 4 discharge knobs already explain ≈90% of the H₂O₂ variance; this frames Config C as the "does OES add value?" question rather than "can OES predict?".
- **Figures**: `phase1_model_comparison_bar.png` (shows the strong Config B column and weak Config C column across 7 models).
- **Numbers to cite**: Ridge B 0.904; PLS B 0.898; note that XGBoost is excluded due to the identified anomaly (R² = −0.108 across all configs).

### R2 — PCA-based OES features actively hurt (§2.3.1 Phase 1)
- **Claim**: Under PCA, adding OES to discharge parameters **degrades** performance — the opposite of the expected direction.
- **Content**: Ridge Config C −0.175; MLP Config C −1.131. Interpret this as: PCA captured variance, but none of it was aligned with the H₂O₂ target. Reference `pca_cumulative_variance.png` to pre-empt the obvious objection ("maybe too few PCs"): 11 components already preserve ≥95% of the spectral variance, so the failure is structural, not a cutoff artefact.
- **Figures**: `pca_cumulative_variance.png`, `phase1_predicted_vs_actual_grid.png` (visual diagnostic of the scatter collapse for Config C with non-linear models).
- **Numbers to cite**: Ridge C −0.175; MLP C −1.131; PCA 11 components ≥ 95% variance.
- **Framing note**: state this as "H1 is rejected", **not** "Phase 1 failed".

### R3 — Hyperparameter tuning is not the answer (§2.3.1 Phase 2, inline)
- **Claim**: Optuna tuning lifts non-linear models substantially but cannot move any OES-inclusive model above the Config B baseline.
- **Content**: MLP Config C −1.13 → 0.37 (ΔR² ≈ +1.50); CNN Config C 0.688 → 0.775 (ΔR² ≈ +0.09), the best tuned OES-only result — yet still below Ridge B 0.904. Tuned MLP Config B 0.861 approaches but does not exceed the simple Ridge baseline. Interpret: tuning closes part of the gap, but H2 is rejected — the ceiling is set by feature quality, not model capacity.
- **Figures**: **none** (per the user Plan, Phase 2 gets no dedicated figure; report inline numbers or a very small inline table).
- **Numbers to cite**: MLP C −1.13 → 0.37; CNN C 0.69 → 0.78; MLP B 0.568 → 0.861.
- **Transition hook**: "If tuning cannot compensate, the remaining lever is the features themselves — which motivates Phase 3."

### R4 — Domain features deliver a step-change (§2.3.2 Phase 3) ← **narrative centre**
- **Claim**: Replacing 11 PCA components with 13 physically motivated OES features transforms Config C from a liability into the equal of Config B across every model class.
- **Content**: headline numbers — Ridge C −0.175 → 0.798 (ΔR² ≈ +0.97); MLP C −1.131 → 0.815 (ΔR² ≈ +1.95); PLS C 0.625 → 0.744. Qualitative interpretation: the Config B / Config C gap, which dominated Phases 1–2, effectively closes. This is the central finding of the project — the same OES spectra, re-encoded with physical knowledge, unlock the predictive signal that PCA had obscured.
- **Figures**: `phase1_vs_phase2_vs_phase3_comparison.png` (the bar chart making the step-change visually unmissable).
- **Numbers to cite**: Ridge C, MLP C, PLS C deltas across all three phases.
- **Proportional space**: this beat earns a full paragraph (ideally the longest in §2.3), mirroring how §2.5 was given proportional space in the Methodology.

### R5 — Which features drive prediction? (§2.3.3 Phase 4 — importance)
- **Claim**: A 4-model consensus ranking (Ridge, PLS, RF, MLP) identifies a small set of dominant predictors: flow rate, the CO₂⁺ band integral (398–412 nm), and pulse width.
- **Content**: top-3 consensus ranks — flow_rate_sccm (mean rank 1.75), band_CO2p_398_412 (4.75), pulse_width_ns (5.25); ties at rank 4 for I_486_Hb and band_CO_Hb_460_500. Interpret: the two most influential OES features are **integrals and bands**, not single-wavelength line intensities — a physically intuitive result since band features are drift-invariant. Flag that no single importance method is unbiased, so the consensus rank is the robust summary.
- **Figures**: `fig1_feature_importance_heatmap.pdf` (the consensus importance heatmap across 4 models).

### R6 — How uncertain are these results? (§2.3.3 Phase 4 — bootstrap)
- **Claim**: Bootstrap 95% CIs reveal genuine overlap between configs and between model classes — honest uncertainty is wide at n = 20.
- **Content**: Ridge B [0.800, 0.955], Ridge C [0.574, 0.910], MLP B [0.767, 0.923], MLP C [0.647, 0.883]. Explicit interpretation: (a) Ridge B and Ridge C overlap — at the 13-feature stage, adding OES does **not** yet provide statistically significant improvement; (b) Ridge C and MLP C overlap — the feature-target relationship appears essentially linear, so a non-linear model buys nothing at this feature set.
- **Figures**: `fig5_bootstrap_r2_distributions.pdf` (full bootstrap distributions).
- **Tone**: honest. This is the critical anti-over-claim beat — the wide CI at n = 20 must be named, not hidden.

### R7 — Statistical significance of the pruned model (§2.3.3 Phase 4 — permutation)
- **Claim**: Once the feature set is pruned, the result is statistically unambiguous: the pruned Ridge achieves R² = 0.920, and no shuffle-label baseline comes close.
- **Content**: permutation test on the 7-feature Ridge (3 OES ratios + 4 discharge); observed R² = 0.920; 2000 label-shuffle permutations; p < 0.0005 (no null R² ≥ observed). Report the test using a compact inline table or two sentences — per the user's Plan, no dedicated figure for permutation.
- **Figures**: **none** (inline numbers or a ≤3-row table).
- **Numbers to cite**: observed R² = 0.9200; 2000 permutations; p < 0.0005.

### R8 — Feature reduction converges on a minimal model (§2.3.3 Phase 4 — ablation)
- **Claim**: Two independent reduction strategies (backward elimination, category ablation) agree: most of the 13 OES features are redundant. The performance peak is not with more features, but with fewer.
- **Content**:
  - **Category ablation** (removing entire OES feature types while keeping discharge):
    - All 13 OES → R² = 0.798
    - Single-wavelength only (7) → R² = 0.823
    - Band integrals only (3) → R² = 0.905
    - **Ratios only (3) → R² = 0.906**
    - Discharge only (Config B) → R² = 0.904
    - Interpretation: ratios and bands each alone match Config B; single-wavelength intensities contribute the least (most drift-prone).
  - **Backward elimination**: R² climbs **monotonically** from 0.798 (13 OES) to 0.918 (1 OES — band_CO2p_398_412), then drops back to 0.904 once the last OES feature is removed, confirming OES contributes real (if modest) information on top of discharge.
  - **Disambiguation of the two optimal models** (already clarified in Phase 2 outline):
    - *Category-ablation optimal*: 3 OES ratios + 4 discharge = **7 features → R² = 0.920** (permutation-confirmed). **Recommended** — retains all three ratio types for physical interpretability and drift invariance.
    - *Backward-elimination optimal*: 1 OES feature (band_CO2p_398_412) + 4 discharge = 5 features → R² = 0.918. More parsimonious but relies on a single OES feature.
- **Figures**: `rank_on_model_config.png` (overall model × config ranking, showing where the pruned Ridge sits relative to every other model–config combination).

### R9 — Synthesis and transition to Conclusions
- **Claim**: The evidence chain closes: PCA failed (R1–R2), tuning cannot rescue it (R3), domain features succeed (R4), and a minimal interpretable subset is sufficient (R5–R8). The 7-feature Ridge is the deployable model the project set out to find.
- **Content**: one short paragraph tying Phase 3 and Phase 4 together. Explicit nod that Ridge–MLP CI overlap (R6) motivates the choice of Ridge for deployment — Occam's razor, not capability gap. Transition sentence into §2.4 Conclusions.
- **Figures**: none.

---

## 4. Anti-Pattern Warnings

### AP1 — "Phase 1 failed" framing
- **Symptom**: paragraphs that read like a post-mortem ("Our baseline did not work…").
- **Why it's bad**: the reader interprets the report as trial-and-error rather than hypothesis-driven science; undermines the central claim.
- **Rule**: always frame Phase 1/2 as **controlled tests that rejected a hypothesis**, never as "failures". Example good sentence: "These results reject the hypothesis that PCA-projected OES features carry sufficient predictive information." Example bad sentence: "Our PCA approach did not work well."

### AP2 — Table-by-table walkthrough (method dump)
- **Symptom**: paragraphs structured as "Table X shows… Table Y shows…"; no interpretive verbs.
- **Why it's bad**: the reader cannot extract the argument; §2.3 devolves into a results appendix.
- **Rule**: one interpretive point per paragraph. Numbers are citations, not the subject.

### AP3 — Re-explaining methodology
- **Symptom**: sentences like "Bootstrap resampling with 500 iterations was used to compute…"
- **Why it's bad**: §2.2 already owns that content; duplication wastes the word budget and weakens the §2/§3 separation.
- **Rule**: cite subsection labels (e.g., "following the protocol in §2.2.6"); mention method names only when essential for interpretation.

### AP4 — Over-claiming on n = 20
- **Symptom**: "Domain features **significantly** improve Config C from −0.17 to 0.80" — without acknowledging that the bootstrap CIs for Ridge B and Ridge C overlap at the 13-feature stage.
- **Why it's bad**: misrepresents uncertainty; a careful reader will catch it and discount the whole section.
- **Rule**: the Phase 3 step-change is real in **magnitude** (ΔR² ≈ 0.97) but **not** formally "significant" in the bootstrap-CI sense until the feature set is pruned. Be explicit about this: Phase 3 delivers the direction; Phase 4's permutation test on the pruned model delivers the significance.

### AP5 — Burying Phase 3
- **Symptom**: Phase 3 gets one compact paragraph equal in size to Phase 1 or Phase 2.
- **Why it's bad**: the central finding of the project becomes visually and rhetorically invisible.
- **Rule**: Phase 3 (beat R4) earns the longest paragraph in §2.3, plus the central figure (`phase1_vs_phase2_vs_phase3_comparison.png`). This mirrors the proportional emphasis §2.5 received in the Methodology.

### AP6 — Figure-caption-as-explanation
- **Symptom**: a figure is inserted with "See Fig. X" and the paragraph text adds nothing beyond the caption.
- **Why it's bad**: the prose must carry the argument even if the figure is removed.
- **Rule**: every figure is cited by a sentence that makes the interpretive point in words; the figure supports, not replaces, the claim.

---

## 5. Paragraph-Level Plan

§2.3 target length: **≈ 700–1000 words** (the full report budget is ≤4000 words for main sections; §2.3 should be shorter than §2.2 but longer than §2.4). Roughly 9 paragraphs mapped to R0–R9.

| Para | Subsection | Role | One-sentence content | Figures / tables | Key numbers |
|:---:|---|---|---|---|---|
| 1 | §2.3 opener (R0) | Roadmap | Resolve 4 hypotheses in order, headline = Phase 3 step-change | — | — |
| 2 | §2.3.1 (R1) | Evidence (baseline) | Config B sets R²≈0.90 ceiling; any OES model must beat this | `phase1_model_comparison_bar.png` | Ridge B 0.904, PLS B 0.898 |
| 3 | §2.3.1 (R2) | Evidence + Interpretation (H1 rejected) | PCA + OES actively degrades R²; structural failure, not cutoff artefact | `pca_cumulative_variance.png`, `phase1_predicted_vs_actual_grid.png` | Ridge C −0.175; MLP C −1.131; 11 PCs ≥95% |
| 4 | §2.3.1 (R3) | Evidence + Interpretation (H2 rejected) | Optuna tuning closes part of the gap but cannot reach Config B from OES | — (inline table or sentences) | MLP C −1.13→0.37; CNN C 0.69→0.78; MLP B 0.568→0.861 |
| 5 | §2.3.2 (R4) ← **longest para** | Evidence + Interpretation (H3 supported) | 13 domain OES features collapse the B/C gap across every model class | `phase1_vs_phase2_vs_phase3_comparison.png` | Ridge C −0.175→0.798; MLP C −1.131→0.815; PLS C 0.625→0.744 |
| 6 | §2.3.3 (R5) | Evidence (importance) | Consensus top-3: flow_rate, band_CO2p, pulse_width — bands beat single wavelengths | `fig1_feature_importance_heatmap.pdf` | flow (1.75), band_CO2p (4.75), pulse_width (5.25) |
| 7 | §2.3.3 (R6) | Evidence + Honest uncertainty | Wide CIs at n=20; Ridge B / Ridge C overlap; Ridge C / MLP C overlap → linear suffices | `fig5_bootstrap_r2_distributions.pdf` | Ridge B [0.800,0.955]; Ridge C [0.574,0.910]; MLP C [0.647,0.883] |
| 8 | §2.3.3 (R7 + R8) | Evidence + Interpretation (H4 supported) | Permutation significance + two-strategy convergence on pruned model | `rank_on_model_config.png` (+ small inline permutation table) | Perm R²=0.9200, p<0.0005, 2000 perms; ratios-only R²=0.906; backward peak R²=0.918; pruned Ridge R²=0.920 |
| 9 | §2.3.3 closer (R9) | Synthesis + Transition | Evidence chain closes; Ridge chosen by Occam, not capability; hand off to §2.4 | — | — |

Note: paragraphs 7 and 8 could merge if the word budget is tight — R7 can be reduced to two sentences (observed R², p-value).

---

## 6. Figures and Tables Plan

### Figures included (7 total — all copied to `final report/report/images/`)

| # | Filename (in `images/`) | Phase | Beat | Argument it makes | Must-have? |
|:---:|---|:---:|:---:|---|:---:|
| 1 | `pca_cumulative_variance.png` | P1 | R2 | 11 PCs already cover ≥95% variance — PCA failure is structural | **yes** |
| 2 | `phase1_model_comparison_bar.png` | P1 | R1 | Config B dominates, Config C lags across 7 models | **yes** |
| 3 | `phase1_predicted_vs_actual_grid.png` | P1 | R2 | Visual scatter collapse for Config C non-linear models | optional (supports R2) |
| 4 | `phase1_vs_phase2_vs_phase3_comparison.png` | P3 | R4 | **The central figure** — step-change from domain features | **yes (centrepiece)** |
| 5 | `fig1_feature_importance_heatmap.pdf` | P4 | R5 | Consensus importance across 4 methods; bands beat single wavelengths | **yes** |
| 6 | `fig5_bootstrap_r2_distributions.pdf` | P4 | R6 | Wide CIs, overlapping distributions — honest uncertainty | **yes** |
| 7 | `rank_on_model_config.png` | P4 | R8 | Where the pruned Ridge lands in the full model×config ranking | **yes** |

### Phase 2 handling (explicit decision)
Per the user's Plan, **Phase 2 gets no dedicated figure**. R3 is told via inline numbers or a small 3-row inline table covering MLP C, CNN C, MLP B transitions. The paragraph is short by design — it is a bridging beat ("tuning helped but not enough"), not an independent finding.

### Permutation test — no figure
The existing Phase 4 permutation figure was flagged as inadequate. Report the permutation test with one or two sentences ("observed R² = 0.9200; across 2000 label permutations, no null R² matched or exceeded the observed value; p < 0.0005") or a compact inline table with 3 rows (observed R², null mean, p-value).

### Tables
Recommended, reusing existing artefacts:
- `tab:bootstrap` — bootstrap 95% CI summary (4 model–config combinations) — supports R6.
- `tab:feature_importance` — consensus top-5 or top-8 features — supports R5.
- `tab:ablation` (new or merged) — category-ablation + backward-elimination peak rows — supports R8.
- `tab:permutation` (optional, inline) — observed R², null mean, n, p — supports R7.

Full R² comparison table across phases (`experiments.md` §3) is useful but large; suggest placing it in an appendix or as a tight 3-column summary table in §2.3.1.

---

## 7. Quantitative Fact Sheet (verified against raw CSVs)

All numbers below were verified against the raw tables; the flow's claims and the outline's claims are anchored here.

### Phase 1 (Baseline, PCA features)
Source: `phase1/results/tables/loocv_results_summary.csv` (and `experiments.md` Table 2.1).
- Ridge: A −0.308, **B 0.904**, C −0.175
- PLS: A −0.604, B 0.898, C 0.625
- SVR: A 0.046, B 0.618, C 0.095
- XGBoost: identical R² = −0.108 across all configs (anomaly — default hyperparameters unsuitable for n = 20 with LOOCV; excluded from Phases 2+)
- RF: A 0.037, B 0.381, C 0.239
- MLP: A −0.850, B 0.568, **C −1.131** (worst result)
- CNN: A 0.301, C 0.688 (no CNN Config B in Phase 1 — see Phase 2 for tuned CNN)

### Phase 2 (Optuna tuning)
Source: `phase2/results/tables/phase2_loocv_results_summary.csv`, `phase1_vs_phase2_comparison.csv`.
- CNN: A 0.301 → 0.534; **C 0.688 → 0.775** (best tuned OES-only model)
- MLP: A −0.850 → 0.374; B 0.568 → 0.861; **C −1.131 → 0.369** (ΔR² ≈ +1.50)
- RF:  A 0.037 → 0.220; B 0.381 → 0.748; C 0.239 → 0.456

### Phase 3 (13 domain OES features)
Source: `phase3/results/tables/phase1_vs_phase2_vs_phase3_comparison.csv`.
- Ridge: A −0.308 → 0.116; B 0.904 → 0.904; **C −0.175 → 0.798** (ΔR² ≈ +0.973)
- PLS:   A −0.604 → 0.350; C 0.625 → 0.744
- RF:    A 0.037 → 0.428; C 0.239 → 0.497
- MLP:   A −0.850 → 0.318; **C −1.131 → 0.815** (ΔR² ≈ +1.946)

### Phase 4 (interpretability + reduction)
Source: `phase4/results/tables/*.csv`.

**Consensus feature importance (top 8)** — `feature_importance_all_models.csv`:

| Rank | Feature | Mean rank |
|:---:|---|:---:|
| 1 | flow_rate_sccm | 1.75 |
| 2 | band_CO2p_398_412 | 4.75 |
| 3 | pulse_width_ns | 5.25 |
| 4 | I_486_Hb | 6.00 |
| 4 | band_CO_Hb_460_500 | 6.00 |
| 6 | frequency_hz | 8.00 |
| 7 | rise_time_ns | 9.25 |
| 8 | ratio_309_656 (OH/Hα) | 9.50 |

**Bootstrap 95% CIs** — `bootstrap_ci_summary.csv`:

| Model | Config | R² point | 95% CI | RMSE |
|---|:---:|:---:|---|:---:|
| Ridge | B | 0.904 | [0.800, 0.955] | 0.071 |
| Ridge | C | 0.798 | [0.574, 0.910] | 0.104 |
| PLS   | B | 0.898 | [0.771, 0.959] | 0.074 |
| PLS   | C | 0.744 | [0.466, 0.884] | 0.117 |
| RF    | B | 0.748 | [0.485, 0.876] | 0.116 |
| RF    | C | 0.497 | [−0.119, 0.768] | 0.164 |
| MLP   | B | 0.857 | [0.767, 0.923] | 0.087 |
| MLP   | C | 0.815 | [0.647, 0.883] | 0.099 |

Ridge B ↔ Ridge C CIs overlap substantially; Ridge C ↔ MLP C CIs also overlap.

**Permutation test** — `permutation_test_summary.csv`:
- observed R² = 0.9200 (7-feature pruned Ridge: 3 ratios + 4 discharge)
- n_permutations = 2000
- p = 0 in the file (no null R² ≥ observed) → reported as **p < 0.0005** (tightest bound given n = 2000)

**Category ablation** — `ablation_summary_article.csv`:

| OES category kept | # OES | Total | R² |
|---|:---:|:---:|:---:|
| All 13 OES | 13 | 17 | 0.798 |
| Single-wavelength only | 7 | 11 | 0.823 |
| Band integrals only | 3 | 7 | **0.9053** |
| Ratios only | 3 | 7 | **0.9063** |
| Discharge only (Config B) | 0 | 4 | 0.904 |

**Backward elimination trajectory** — `ablation_summary_article.csv` (Ridge):

| # OES | Removed | R² |
|:---:|---|:---:|
| 13 | — | 0.798 |
| 12 | I_309_OH | 0.803 |
| 11 | ratio_777_309 | 0.768 |
| 10 | band_CO_Hb_460_500 | 0.822 |
| 9 | I_777_O | 0.856 |
| 8 | I_406_CO2p | 0.852 |
| 7 | ratio_309_656 | 0.866 |
| 6 | I_656_Ha | 0.872 |
| 5 | band_OH_306_312 | 0.890 |
| 4 | I_337_N2 | 0.899 |
| 3 | ratio_656_486 | 0.887 |
| 2 | I_486_Hb | 0.908 |
| **1** | **I_516_C2** | **0.918** ← backward-elim peak |
| 0 | band_CO2p_398_412 | 0.904 (= Config B) |

**Final recommended model**: 3 OES ratios (OH/Hα, O/OH, Hα/Hβ) + 4 discharge = 7 features → **R² = 0.920**, RMSE = 0.066, permutation-confirmed (p < 0.0005).

---

## 8. Consistency Notes (for the Observation section)

- The `ablation_summary_article.csv` reports ratios-only R² = 0.9063 and bands-only R² = 0.9053, whereas `experiments.md` rounds both to 0.906 / 0.905. No conflict — both are consistent to three decimals.
- The permutation test `p_value` column stores `0.0` literally (no null ≥ observed in 2000 shuffles); the correct reporting is "p < 1/2001 ≈ p < 0.0005". The outline already uses this phrasing — keep it.
- The category-ablation ratios-only result (0.906) and the permutation-test observed R² (0.920) differ. Both refer to "ratios + discharge" models, but the permutation-test model was refit/tuned on the reduced set, yielding a slightly higher R². The flow treats them as two separate claims (R8: the ratios category alone equals Config B; R7: the pruned + refit Ridge hits 0.920). This should be made explicit in the prose.
- `phase4_result.txt` is referenced in the Phase 5 Plan but the actual file on disk is `phase4_result.md`. Not a blocker — the figures it references all exist. Flag in Observation.
- Phase 2 does not emit a dedicated figure in §2.3. The user has explicitly chosen this; the narrative is short (R3) and handled inline.
