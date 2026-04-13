# Conclusion Section — Writing Logic Flow (Phase 7)

> Blueprint for drafting §2.4 Conclusions of the final report. Follows the same 8-part structure as `introduction_flow.md` (Phase 2), `methodology_flow.md` (Phase 3), and `results_flow.md` (Phase 5). Every quantitative claim below is sourced from `results_flow.md` §7; if the Conclusion contradicts a number there, the Conclusion is wrong.

---

## 1. Pre-Writing Questions

The Conclusion cannot begin until these four questions have a one-sentence answer.

1. **What single sentence does the reader take away if they read nothing else in §2.4?**
   - *Answer*: On a small, physically structured plasma dataset, encoding traditional spectroscopic knowledge (ratios and band integrals) as ML features — not model complexity or hyperparameter tuning — is the decisive lever that turns H₂O₂ yield prediction from unworkable (R² = −0.175) into deployable (R² = 0.920 on a 7-feature Ridge, permutation p < 0.0005).

2. **Which objectives from §1.2 are met, partially met, or unmet?**
   - *Objective 1* (Build ML models predicting H₂O₂ yield from OES + discharge): **fully met** — pruned Ridge R² = 0.920, RMSE = 0.066.
   - *Objective 2* (Test whether domain features outperform PCA): **fully met** — Ridge Config C Δ = +0.973 (−0.175 → 0.798) at the 13-feature stage, formally confirmed at R² = 0.920 after pruning.
   - *Objective 3* (Identify a minimal, interpretable feature set): **fully met** — two independent reduction strategies converge on ≤ 7 features; the recommended model uses 3 drift-invariant OES ratios plus 4 discharge parameters.

3. **What is the honest limitation statement — what did `n = 20` and single-reactor data *prevent* the project from concluding?**
   - At the 13-feature stage, the bootstrap CIs of Ridge Config B [0.800, 0.955] and Ridge Config C [0.574, 0.910] overlap — so the project cannot claim a statistically significant OES benefit at that stage. The permutation significance (p < 0.0005) attaches *only* to the pruned 7-feature Ridge, not to the full 13-feature set.
   - All results come from one reactor, one operator, one wavelength calibration; cross-reactor generalisation is untested.
   - The specific feature list was derived from the CO₂-bubble plasma chemistry; other plasma systems will require a different list.

4. **What broader implication generalises beyond this dataset (link back to §1.3)?**
   - §1.3 framed the literature gap as "modern OES + ML pipelines rely on PCA/PLS and overlook the diagnostic domain knowledge that traditional spectroscopy has encoded for decades". §2.4 closes this loop: for small, physically structured scientific datasets, domain-knowledge feature engineering beats both automated dimensionality reduction and model capacity as the dominant predictive lever.

---

## 2. Backward Reasoning — From Takeaway to Evidence

The Conclusion's job is *interpretation*, not recapitulation. The chain below starts at the takeaway sentence and walks back to the §2.3 beats that license it.

- **Takeaway (T)**: Domain-knowledge features are the decisive predictive lever on this small, physically structured dataset; a 7-feature Ridge is the deployable artefact.
- ← **Objective resolution (O1, O2, O3)** licensed by:
  - O1 by the pruned Ridge R² = 0.920 (R8 / fact sheet §7, pruned-refit row)
  - O2 by the Phase 1 → Phase 3 step-change (R4 / fact sheet §7, Ridge C −0.175 → 0.798; MLP C −1.131 → 0.815)
  - O3 by the two-strategy convergence at ≤ 7 features (R8 / fact sheet §7)
- ← **Central contribution (CC)**: "domain features, not model complexity, closed the gap" licensed by R4 (step-change magnitudes) and R3 (tuning insufficient: MLP C −1.13 → 0.37, CNN C 0.69 → 0.78, still below Ridge B 0.904).
- ← **Minimal-model claim (MM)**: the 7-feature Ridge is permutation-significant (R7: observed R² = 0.920, p < 0.0005 on 2000 shuffles) and lives inside the same CI envelope as the non-linear models (R6: Ridge C [0.574, 0.910] overlaps MLP C [0.647, 0.883]).
- ← **Honest limitation (HL)**: at the 13-feature stage the bootstrap CIs of Ridge B and Ridge C overlap (R6), so the significance claim is *only* defensible after pruning.
- ← **Broader implication (BI)**: the §1.3 framing — traditional spectroscopy knows ratios and bands matter; modern ML overlooks them — is confirmed operationally by this evidence chain; this generalises to other small, physically structured scientific datasets.

Every link above points at an R-beat or a row in `results_flow.md` §7. The Conclusion therefore introduces no new numbers; it interprets numbers the reader has already seen in §2.3.

---

## 3. Forward Story (C1 – C5)

Five paragraphs, one interpretive point each. Total target: ≈ 420 words.

### C1 — Objective resolution (≈ 80 words)
- **Role**: opening; explicitly name each of the three objectives from §1.2 and state the verdict with the headline number.
- **Content**: "The first objective… was met by the pruned 7-feature Ridge at R² = 0.920. The second objective… was met by the Phase 1 → Phase 3 step-change: Ridge Config C moved from −0.175 to 0.798 and MLP Config C from −1.131 to 0.815. The third objective… was met by the convergence of backward elimination and category ablation on a model of at most seven features."
- **Numbers cited**: 0.920 (pruned Ridge), −0.175 → 0.798 (Ridge C), −1.131 → 0.815 (MLP C), "≤ 7 features".
- **Citation**: none.

### C2 — Central contribution (≈ 100 words)
- **Role**: interpret *why* the project succeeded; name domain-knowledge feature engineering as the decisive lever and contrast it with the two levers that did not work (PCA and hyperparameter tuning).
- **Content**: Phase 1 ruled out the hypothesis that PCA-projected OES features carry the signal; Phase 2 ruled out the hypothesis that hyperparameter tuning compensates (best tuned OES-only result CNN C = 0.775, still below Ridge B = 0.904); Phase 3 showed that re-encoding the same OES spectra with 13 physically motivated features was sufficient to close the Config B / Config C gap across every model class.
- **Numbers cited**: Ridge B 0.904, CNN C 0.688 → 0.775, MLP C −1.131 → 0.815 (the two deltas stay in C1; here reinforce the "tuning cannot rescue" point).
- **Constraint**: the wording must be about a **step-change in point estimates**, not about a formal PCA-vs-domain significance test — that claim is reserved for the pruned model in C3.

### C3 — Minimal deployable model (≈ 90 words)
- **Role**: name the 7-feature Ridge as the deployable artefact; attach its permutation significance; justify the choice of Ridge over MLP via Occam's razor rather than capability gap.
- **Content**: the recommended model is a Ridge regression on three drift-invariant spectroscopic ratios (OH/Hα, O/OH, Hα/Hβ) plus the four discharge parameters, yielding R² = 0.920 and RMSE = 0.066, with permutation significance p < 0.0005 across 2000 label shuffles. The overlap of Ridge Config C and MLP Config C bootstrap CIs at the 13-feature stage (R6) motivates the Occam's-razor choice of the linear model.
- **Numbers cited**: 0.920, RMSE 0.066, 2000 perms, p < 0.0005.

### C4 — Honest limitations (≈ 80 words)
- **Role**: state what the evidence does *not* license.
- **Content**: (i) at the 13-feature stage, Ridge Config B [0.800, 0.955] and Ridge Config C [0.574, 0.910] have overlapping bootstrap CIs, so a "domain features significantly better than PCA" claim is *not* defensible before pruning — only the pruned 7-feature Ridge carries the permutation significance; (ii) n = 20 is the root cause of the wide CIs; (iii) the data originate from a single reactor and a single operator, so cross-reactor generalisation is untested; (iv) the feature list itself is specific to this plasma chemistry and other systems will need their own domain features.
- **Numbers cited**: CIs from fact sheet §7.

### C5 — Broader implication + transition (≈ 70 words)
- **Role**: generalise beyond this dataset; link back to §1.3; hand off to §2.5.
- **Content**: the evidence closes the literature gap identified in §1.3 — for small, physically structured scientific datasets, domain-knowledge feature engineering outperforms both automated dimensionality reduction and model complexity as the dominant predictive lever. Close with a one-sentence transition that signals §2.5 Future Work will take up cross-reactor generalisation and live deployment.
- **Citations**: none in §2.4 (reserve `Liu2022UDA` and `Zhao2024TransferSSL` for §2.5).

### C6 — Optional one-sentence Wang2025 comparison (≈ 30 words, budget-permitting)
- **Role**: external validation.
- **Content**: the pruned Ridge R² ≈ 0.92 falls inside the R² ≈ 0.90–0.97 band reported by Wang et al. for LIR-based Te/ne prediction on a different plasma system~\cite{Wang2025MLOESCascaded}, reinforcing the generality of domain-feature ML on OES.
- **Decision**: include only if paragraph budget permits; otherwise drop and keep §2.4 citation-free.

---

## 4. Anti-Pattern Warnings

### AP1 — Summary-only ending
- **Symptom**: Conclusion reads "Phase 1 showed…, Phase 2 showed…, Phase 3 showed…, Phase 4 showed…" — a mini-§2.3.
- **Why it's bad**: the reader already has §2.3; rewriting it in smaller text wastes the word budget and hides the interpretive payload.
- **Rule**: §2.4 interprets and generalises; it does not recapitulate. Each paragraph must contain at least one verb that is not "showed" / "found" / "obtained".

### AP2 — Over-claim on n = 20
- **Symptom**: "Domain features are statistically significantly better than PCA."
- **Why it's bad**: at the 13-feature stage the bootstrap CIs of Ridge B and Ridge C overlap, so this claim is not supported by the data.
- **Rule**: frame the Phase 3 finding as a "step-change in point estimates" — a large, directionally consistent improvement. Attach the word "significant" only to the pruned 7-feature Ridge via the permutation test.

### AP3 — Introducing new numbers
- **Symptom**: a quantity appears in §2.4 that is not in §2.3 or `results_flow.md` §7.
- **Why it's bad**: the Conclusion is supposed to close the evidence chain, not extend it; a new number forces the reader to re-verify the whole section.
- **Rule**: every R² / Δ / CI / p-value in §2.4 must be traceable to an R-beat in §2.3 and a row in `results_flow.md` §7.

### AP4 — Under-claim by hedging everything
- **Symptom**: compound "perhaps", "may", "might" hedges that dilute a valid finding into nothing.
- **Why it's bad**: once the qualifier (n = 20, single reactor) has been stated, further hedging on the core finding reads as a lack of confidence.
- **Rule**: state the central claim plainly *after* C4 has admitted the limitation. The pruned-model permutation significance is real — do not hedge it.

### AP5 — Future-Work creep
- **Symptom**: §2.4 turns into a list of "next steps: generalisation, deployment, data collection, transfer learning, …".
- **Why it's bad**: §2.5 owns that content; duplicating it here wastes the word budget and blurs the §2.4 / §2.5 boundary.
- **Rule**: §2.4 transitions *to* §2.5 in a single sentence at the end of C5. No enumerated future-work list in §2.4.

---

## 5. Paragraph-Level Plan

| Para | Role | One-sentence content | Numbers | Citation |
|:---:|---|---|---|---|
| C1 | Objective resolution | Name each §1.2 objective and mark it met with the headline number. | 0.920, −0.175 → 0.798, −1.131 → 0.815, ≤ 7 feats | — |
| C2 | Central contribution | Domain-knowledge features, not PCA or tuning, were the decisive lever. | Ridge B 0.904, CNN C 0.688 → 0.775 | — |
| C3 | Minimal deployable model | 7-feature Ridge (3 ratios + 4 discharge), permutation-confirmed; Ridge over MLP by Occam. | 0.920, RMSE 0.066, 2000 perms, p < 0.0005 | — |
| C4 | Honest limitations | CI overlap at 13-feature stage; n = 20; single reactor; feature list is chemistry-specific. | Ridge B [0.800, 0.955], Ridge C [0.574, 0.910] | — |
| C5 | Broader implication + transition | Generalises the §1.3 gap; hand off to §2.5. | — | — |
| C6 (opt.) | External comparison | Ridge R² ≈ 0.92 matches Wang2025 LIR band (R² ≈ 0.90–0.97). | 0.90–0.97 | `Wang2025MLOESCascaded` |

---

## 6. Numbers and Claims Checklist

Every number below must already appear in `results_flow.md` §7 and/or §2.3. The Conclusion cites, does not introduce.

| # | Claim | Number | Fact-sheet row | Paragraph |
|:---:|---|:---:|---|:---:|
| 1 | Pruned Ridge point estimate | R² = 0.920 | §7 Phase 4 "Final recommended model" | C1, C3 |
| 2 | Pruned Ridge RMSE | 0.066 | §7 Phase 4 "Final recommended model" | C3 |
| 3 | Permutation significance | p < 0.0005 (2000 perms) | §7 Phase 4 "Permutation test" | C3 |
| 4 | Ridge C Phase 1 → Phase 3 | −0.175 → 0.798 (Δ ≈ +0.97) | §7 Phase 3 row 1 | C1, C2 |
| 5 | MLP C Phase 1 → Phase 3 | −1.131 → 0.815 (Δ ≈ +1.95) | §7 Phase 3 row 4 | C1 |
| 6 | Ridge B ceiling | R² = 0.904 | §7 Phase 1 row 1 | C2 |
| 7 | CNN C tuned (best OES-only tuned) | 0.688 → 0.775 | §7 Phase 2 row 1 | C2 |
| 8 | Ridge B bootstrap CI | [0.800, 0.955] | §7 Phase 4 bootstrap | C4 |
| 9 | Ridge C bootstrap CI (13-feat) | [0.574, 0.910] | §7 Phase 4 bootstrap | C4 |
| 10 | Feature count recommended | 7 (3 ratios + 4 discharge) | §7 Phase 4 ablation | C1, C3 |
| 11 (opt.) | Wang2025 LIR band | R² ≈ 0.90–0.97 | §1.3 | C6 |

Anything not on this list is not in the Conclusion.

---

## 7. Transition Plan

- **§2.3 → §2.4**: §2.3 closes (R9) with the sentence "…at n = 20, the simplest interpretable model that achieves the best point estimate is also the easiest to trust and the cheapest to deploy." This is the exact rhetorical hook §2.4 picks up — Conclusion C1 opens by mapping that deployable model onto the three §1.2 objectives.
- **§2.4 → §2.5**: Conclusion C5 ends with a single transition sentence: "The generalisation, deployment, and data-expansion questions raised by these limitations are addressed in Section §2.5." That sentence is the entire hand-off; §2.5 owns the future-work content itself.

---

## 8. Word Budget

| Beat | Target words | Cumulative |
|---|:---:|:---:|
| C1 Objective resolution | ≈ 80 | 80 |
| C2 Central contribution | ≈ 100 | 180 |
| C3 Minimal deployable model | ≈ 90 | 270 |
| C4 Honest limitations | ≈ 80 | 350 |
| C5 Broader implication + transition | ≈ 70 | 420 |
| C6 Optional Wang2025 sentence | ≈ 30 | 450 |

**Hard bounds**: 350 words minimum, 500 words maximum. This leaves ≈ 750–900 words for §2.5 Future Work + §2.6 Reflection out of the ≈ 1250-word budget remaining after §1 + §2 + §2.3.
