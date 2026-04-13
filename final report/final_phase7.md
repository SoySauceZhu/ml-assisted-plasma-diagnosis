# Plan

## Background
Previous phases has covered sections including intro, method, result.

In this phase, we are gonna generate conclusion at once. So you should articulating the narrative arc of conclusion and then generate the conclusion to main.tex

## Task
- Acquire enough context from my research. Including the report up to now in previous three sections.
- Generate the narrative arc of conclusion section and outline
- Write the conclusion and update main tex.


## Deliverables
- conclusion_flow.md
- outline.md
- updated main latex file

## Tips
- You can use skill .claude/skills/research-paper-writing as writing suggestion

---

# Action

## Prerequisites

Before executing, the agent must read and internalise the following context (roughly in order):

1. **Phase 5 results writing flow** → `final report/final_phase5/results_flow.md`
   - Contains: the verified quantitative fact sheet (§7) and the forward story R0–R9 that §2.3 lands
   - Purpose: authoritative source of every number the Conclusion will cite. If the Conclusion contradicts a number here, the Conclusion is wrong

2. **Phase 5 outline** → `final report/final_phase5/outline.md`
   - Contains: the most recent skeleton of §2.4 Conclusions (objective-keyed bullets and broader implication)
   - Purpose: the structural template — §2.4 must still answer the three objectives stated in §1.2 and close with the broader implication

3. **Current `main.tex`** → `final report/report/main.tex`
   - Key sections to re-read **before** writing:
     - §1.2 Objectives (three objectives — the Conclusion must resolve each one by name)
     - §2.3 Findings / Results (the prose just drafted in Phase 6) — specifically the R6 paragraph on bootstrap CI overlap and the R7/R8 paragraph disambiguating 0.906 vs 0.920
     - The existing §2.4 Conclusions skeleton (lines around 458+) — this will be replaced
     - §1.3 State of the Art — the Conclusion should echo the "domain knowledge is decisive" framing set up in §1.3 and resolved in §2.3.2
   - Purpose: the Conclusion must land the claims set up in §1 and delivered in §2.3, without contradicting either

4. **Phase 2 and Phase 3 writing flows** → `final report/final_phase2/introduction_flow.md`, `final report/final_phase3/methodology_flow.md`
   - Purpose: match tone, citation density, and paragraph style used in the already-drafted §1 and §2

5. **Phase 6 Observation** → `final report/final_phase6.md` (Observation section)
   - Key items the Conclusion must respect:
     - **§2.3 R6 honesty rule**: the Conclusion must not claim "domain features are statistically significantly better than PCA" — the correct formulation is "domain features produce a step-change in point estimates and, after pruning, deliver a permutation-confirmed result" (CI overlap at the 13-feature stage)
     - **0.906 vs 0.920 distinction**: the Conclusion cites R² = 0.920 only for the pruned-refit 7-feature Ridge, and must attach the permutation significance (`p < 0.0005`) only to that model
     - **Word budget remaining ≈ 1250 words** for §2.4 + §2.5 + §2.6 combined — the Conclusion should budget ≈ 350–500 words, leaving room for Future Work and Reflection
     - The skeleton §1 objectives use "R² = 0.80" and the skeleton §2.4 uses "R² = 0.920" — the Conclusion must match the verified numbers in `results_flow.md` §7, not the old skeleton

6. **References** → `final report/report/references.bib`
   - Purpose: available IEEE citation keys. §2.4 should be citation-light — cite only `Wang2025MLOESCascaded` (one comparison sentence if used) and `Liu2022UDA` / `Zhao2024TransferSSL` are reserved for §2.5 Future Work, not §2.4

7. **Report guidance & marking descriptors** (PDFs, optional skim):
   - `final report/Final report Guidance 2026.pdf`
   - `final report/marking descriptors Final Report_2026.pdf`
   - Purpose: what the marker expects from a Conclusions section (clear objective resolution, explicit limitations, no new results, no repetition for its own sake)

8. **Skill available:**
   - `research-paper-writing` — consult for Conclusion-section templates, anti-patterns (summary-only endings, over-claim, under-claim)

---

## Step 1: Write `conclusion_flow.md`

Create `final report/final_phase7/conclusion_flow.md` following the same 8-part structure used for `introduction_flow.md`, `methodology_flow.md`, and `results_flow.md`. Put the file under a new `final_phase7/` subfolder, creating the folder if it does not yet exist.

Required sections:

1. **Pre-writing questions** — the four questions the Conclusion must answer before the first sentence is written:
   1. What single sentence does the reader take away if they read nothing else in §2.4?
   2. Which objective(s) from §1.2 are fully met, partially met, or unmet?
   3. What is the honest limitation statement — what did `n = 20` and single-reactor data *prevent* the project from concluding?
   4. What broader implication generalises beyond this dataset (link back to §1.3 literature framing)?

2. **Backward reasoning** — start from the single takeaway sentence and work backwards through the evidence chain:
   - Takeaway → objective resolution → headline numbers → honest limitation → broader implication
   - Each link must point at a specific §2.3 beat (R1–R9) or fact-sheet row so the Conclusion remains grounded, not rhetorical

3. **Forward story (C1 – C5 or C1 – C6)** — numbered beats of the Conclusion narrative:
   - C1: objective resolution paragraph — name each objective from §1.2 and state whether it is met, citing the headline number
   - C2: central contribution paragraph — domain-knowledge features as the decisive factor (Phase 3 step-change) with the R4 numbers
   - C3: minimal-model paragraph — the 7-feature Ridge with R² = 0.920 (permutation-confirmed) as the deployable artefact
   - C4: limitation paragraph — `n = 20`, bootstrap CI overlap, single-reactor scope, no cross-dataset validation
   - C5: broader-implication paragraph — short, cites back to the §1.3 literature gap (domain knowledge overlooked by modern ML), explicit transition to §2.5 Future Work
   - (Optional C6: one-sentence literature comparison citing `Wang2025MLOESCascaded`'s R² ≈ 0.90–0.97 on Te/ne as a parallel validation of the domain-feature approach)

4. **Anti-pattern warnings** (≥ 4):
   - **AP1 — Summary-only ending**: do not restate §2.3 verbatim. The Conclusion's job is to *interpret* and *generalise*, not to recapitulate numbers
   - **AP2 — Over-claim on n = 20**: do not say "domain features are significantly better than PCA" — the 13-feature bootstrap CIs overlap. The correct claim is about the pruned 7-feature model
   - **AP3 — New results in the Conclusion**: every R²/p-value must already appear in §2.3. The Conclusion cites, does not introduce
   - **AP4 — Under-claim by hedging everything**: once the claim is qualified (n = 20, single reactor), state the finding plainly. Do not undermine a valid result with compound "perhaps", "may", "might" hedges
   - **AP5 — Future-Work creep**: the Conclusion must *transition* to §2.5 but not *become* §2.5. Keep forward-looking content to one sentence at the very end

5. **Paragraph-level plan** — one entry per paragraph with: paragraph role (Objective resolution / Central contribution / Minimal model / Limitation / Broader implication), one-sentence content summary, numbers to include, any citation key

6. **Numbers and claims checklist** — explicit list of every quantitative claim the Conclusion will make, each tied to a row in `results_flow.md` §7. Anything not on this list is not in the Conclusion

7. **Transition plan** — two sentences: how §2.3 hands off to §2.4, and how §2.4 hands off to §2.5 (Future Work)

8. **Word budget** — target 350–500 words total, distributed roughly as: C1 ≈ 80w / C2 ≈ 100w / C3 ≈ 90w / C4 ≈ 80w / C5 ≈ 70w

---

## Step 2: Update `final_phase5/outline.md` §2.4 → Phase 7

Refine the existing §2.4 Conclusions block in `final_phase5/outline.md` (do **not** create a new outline file — append to the existing one). The current §2.4 is three objective-keyed bullets and one broader-implication bullet; replace with a paragraph-structured outline that mirrors the C1–C5 beats from `conclusion_flow.md`:

- C1 Objective resolution → one bullet per objective with the verified number
- C2 Central contribution → explicit pointer to §2.3.2 R4 and the step-change numbers
- C3 Minimal-model recommendation → explicit pointer to §2.3.3 R8 and the permutation-confirmed R²
- C4 Limitations → list at minimum: `n = 20` / CI overlap at 13-feature stage / single reactor / no cross-dataset generalisation / XGBoost anomaly noted but not a true limitation
- C5 Broader implication → one bullet linking back to the §1.3 literature gap

Add a `> **Phase 7 note:**` line at the top of §2.4 explaining that this outline is the blueprint for the rewritten §2.4 in `main.tex`. Keep §2.5 Future Work and §2.6 Reflection unchanged — they are out of Phase 7's scope.

---

## Step 3: Rewrite `\section{Conclusions}` in `main.tex`

Replace the existing `\section{Conclusions}` block (currently around line 458+) with polished prose following the C1–C5 beats from `conclusion_flow.md`. The section should compile cleanly and produce 5–6 paragraphs.

### Writing rules for the Conclusion

1. **Follow `conclusion_flow.md` strictly** — do not introduce beats not planned
2. **Answer each objective by name**: §1.2 poses three objectives; §2.4's opening paragraph (C1) must resolve each one explicitly ("The first objective was met by…"). Use the exact objective wording from §1.2
3. **Cite headline numbers from `results_flow.md` §7**:
   - Config B Ridge R² = 0.904 (baseline)
   - Phase 1 Ridge C R² = −0.175 → Phase 3 Ridge C R² = 0.798 (Δ ≈ +0.97) — the step-change
   - MLP C −1.131 → 0.815 (Δ ≈ +1.95) — second-largest swing
   - Pruned 7-feature Ridge R² = 0.920, permutation `p < 0.0005`
   - Bootstrap CI overlap (Ridge B [0.800, 0.955] vs Ridge C [0.574, 0.910]) — mentioned in C4, not C2
4. **No new numbers**: every quantitative claim must already appear in §2.3. The Conclusion interprets, it does not introduce
5. **Honest limitation statement** (C4):
   - `n = 20` as the root cause of the wide bootstrap CIs
   - The 13-feature Ridge C is *not* statistically distinguishable from Config B; the pruned 7-feature Ridge *is* (via permutation test)
   - Single-reactor dataset; no cross-reactor generalisation tested
   - One sentence acknowledging that the central findings depend on the specific feature list derived in §2.2.5 and that different plasmas may need different feature sets
6. **Broader implication** (C5): one short paragraph linking back to the §1.3 gap — for physically structured scientific datasets (especially small ones), domain-knowledge feature engineering is more decisive than automated dimensionality reduction or model complexity. Transition sentence to §2.5
7. **Optional `Wang2025MLOESCascaded` comparison**: if space allows, one sentence noting that the pruned Ridge R² ≈ 0.92 lands in the same range as Wang et al.'s LIR-based R² ≈ 0.90–0.97 for a different prediction task (Te/ne), reinforcing the generality of the domain-feature approach
8. **Tone and style**: passive voice, no first-person, IEEE citation style (though §2.4 should be citation-light or citation-free), consistent with §1 and §2
9. **No figures or tables in §2.4** — the Conclusion is prose-only
10. **Approximate length**: 350–500 words for the entire §2.4

### Table updates required

None. §2.4 does not contain tables or figures.

### Skeleton content to remove

- Any `\todo{}` markers in the existing §2.4
- Any bullet-list formatting — the Conclusion should be flowing prose, not a list
- Any numbers that contradict `results_flow.md` §7 (e.g., if the skeleton says R² = 0.80 without distinguishing Ridge C 13-feature vs pruned 7-feature, rewrite)

---

## Step 4: Quality checks

After writing, verify:

### Content checks
- [ ] §2.4 opens by resolving the three objectives from §1.2 by name, in order
- [ ] Every quantitative claim matches `results_flow.md` §7 exactly
- [ ] The step-change numbers (Ridge C −0.175 → 0.798 and/or MLP C −1.131 → 0.815) appear in the central-contribution paragraph
- [ ] The 7-feature Ridge is cited as R² = 0.920 with `p < 0.0005`, attached only to the pruned-refit model
- [ ] Bootstrap CI overlap appears in the limitation paragraph (C4), not the contribution paragraph (C2)
- [ ] No R² numbers appear that are not already in §2.3
- [ ] The Conclusion does not claim "statistically significantly better than PCA" at the 13-feature stage
- [ ] A broader-implication sentence links back to the §1.3 literature gap and transitions to §2.5
- [ ] All 5 anti-patterns from `conclusion_flow.md` are avoided
- [ ] Word count is in the 350–500 range

### LaTeX checks
- [ ] The rewritten `\section{Conclusions}` compiles cleanly (pdflatex + bibtex + pdflatex + pdflatex)
- [ ] The `\label{sec:conclusions}` is preserved for cross-references from §1 and §2
- [ ] No new undefined citation keys; `\cite{Wang2025MLOESCascaded}` resolves if used
- [ ] No `\todo{}` placeholders remain in §2.4

### Consistency with §1, §2, and §2.3
- [ ] Objective resolution in §2.4 matches the three objectives verbatim from §1.2
- [ ] Numbers in §2.4 match §2.3 and `results_flow.md` §7
- [ ] Framing ("domain knowledge is decisive") is preserved from §1 and §2.2.5
- [ ] Tone matches §1 / §2 / §2.3 (passive voice, no first-person)
- [ ] The R6 honesty rule is respected — no over-claim relative to `n = 20`

---

## Deliverables

| # | Deliverable | Location | Description |
|---|---|---|---|
| 1 | Conclusion writing flow | `final report/final_phase7/conclusion_flow.md` | 8-part writing logic flow following the same structure as `introduction_flow.md` / `methodology_flow.md` / `results_flow.md` |
| 2 | Updated outline §2.4 | `final report/final_phase5/outline.md` §2.4 | Refined from skeleton bullets to the C1–C5 paragraph blueprint with numbers and pointers |
| 3 | Rewritten §2.4 Conclusions | `final report/report/main.tex` §2.4 | 5–6 paragraphs, 350–500 words, C1–C5 beats, honest limitation statement, compilable |

No changes to §1, §2, §2.3, §2.5, §2.6, `references.bib`, or the `images/` directory in this phase.

---

## Quality Criteria

1. **Objective resolution**: all three §1.2 objectives are answered explicitly by number and by name
2. **Narrative arc**: the Conclusion interprets and generalises rather than recapitulates — C1 (what was done) → C2 (central finding) → C3 (deployable artefact) → C4 (what we cannot claim) → C5 (why it matters beyond this project)
3. **Honest uncertainty**: the `n = 20` ceiling and the CI overlap at the 13-feature stage are named explicitly; the permutation significance is attached only to the pruned model
4. **No new results**: every number is traceable to §2.3 and `results_flow.md` §7
5. **Anti-patterns avoided**: AP1 (summary-only), AP2 (over-claim), AP3 (new results), AP4 (hedge-everything under-claim), AP5 (Future-Work creep) all clear
6. **Compilable LaTeX**: the file compiles without errors; §2.4 is 350–500 words and sits cleanly between §2.3 and §2.5

---

# Observation

## What worked

- **8-part flow held**: `conclusion_flow.md` cleanly inherited the structure from `introduction_flow.md` / `methodology_flow.md` / `results_flow.md`. The pre-writing questions + backward reasoning pair forced an explicit takeaway sentence and evidence chain before any prose was written, which kept C1–C5 from drifting into a §2.3 recap.
- **Numbers checklist did its job**: the 11-row claims checklist in §6 of `conclusion_flow.md` was used directly as a lookup table while drafting; every R² / Δ / CI / p-value in the final §2.4 is traceable to a single row in `results_flow.md` §7. Zero new numbers introduced — AP3 cleared by construction, not by retrospective audit.
- **Honest-uncertainty discipline survived**: the R6 rule from Phase 6 (CI overlap at the 13-feature stage → no "PCA-vs-domain significance" claim) carried through to C2 and C4. C2 describes Phase 3 as a "step-change" (point estimates); "significance" appears only once, attached to the pruned 7-feature Ridge via the permutation test. AP2 cleared.
- **Occam framing landed**: the Ridge-over-MLP choice is justified in C3 by bootstrap-CI overlap, not by a capability claim — reinforces the Phase 6 synthesis paragraph's framing and keeps §2.4 internally consistent with §2.3 R9.
- **Outline edit stayed in place**: §2.4 of `final_phase5/outline.md` was rewritten in situ (no new outline file), as the Plan required, with a `Phase 7 note` at the top and the C1–C5 paragraph blueprint below. §2.5 and §2.6 untouched.
- **Compilation clean**: pdflatex + bibtex + pdflatex×2, 27 pages, zero `!` errors. Only residual warnings are the pre-existing `acro:*` hyperref placeholders from the List of Abbreviations — outside Phase 7 scope.

## Self-critique of the §2.4 draft

- **Word count: 514 (target 350–500)**. The draft is ≈ 3 % over the stated ceiling after one round of trimming. Paragraphs C1 and C2 carry most of the slack — C1 names all three objectives by quoting each one (necessary for marker visibility) and C2 carries both Phase 1 and Phase 2 rejection numbers. Further trimming would force dropping either an objective restatement or the Phase 2 rebuttal; neither is desirable at the cost of a ~14-word overrun. **Verdict: accept as "approximately 500".**
- **C1 reads slightly list-like**. "The first, …; the second, …; the third, …" is rhetorically close to a three-bullet enumeration. Acceptable given that §1.2 is itself an enumerated list and the marker explicitly wants each objective named, but on a second editing pass this could be loosened into prose that threads the three objectives into a single argumentative sentence.
- **C6 Wang2025 comparison folded into C5**: the optional one-sentence external-validation comparison was merged into C5 rather than kept as a standalone C6. This preserved the word budget but means §2.4 carries exactly one citation (`Wang2025MLOESCascaded`) instead of being strictly citation-free. Still consistent with the Plan ("citation-light or citation-free").
- **C4 limitation paragraph is dense**. Four independent limitations (CI overlap, n=20, single reactor, feature-list specificity) are compressed into two sentences. A reader who misses the semicolon structure could miss one of the four. Could be split into two paragraphs if word budget were relaxed, but not at the current 500-word target.
- **No XGBoost anomaly mention in §2.4**. The Plan explicitly permitted leaving it out of C4 ("documented and excluded, not a ceiling"). The decision is defensible — the anomaly belongs to §2.3.1 where it is already noted — but a careful marker might expect it flagged as a caveat. Acceptable risk.

## Anti-pattern audit

| AP | Description | Status |
|:---:|---|:---:|
| AP1 | Summary-only ending (restating §2.3) | **clear** — each paragraph leads with an interpretive verb ("decisive lever", "dictated not by a capability gap", "overlooked knowledge proved to be the dominant lever") |
| AP2 | Over-claim on n=20 ("significantly better than PCA") | **clear** — Phase 3 framed as "step-change", significance attached only to pruned Ridge |
| AP3 | New numbers not in §2.3 | **clear** — all 11 numbers match fact sheet §7 rows |
| AP4 | Under-claim by hedging everything | **clear** — C3 states R²=0.920 with p<0.0005 plainly, no compound "perhaps" / "may" hedges on the core finding |
| AP5 | Future-Work creep | **clear** — single transition sentence at the end of C5; no enumerated future-work list in §2.4 |

## Consistency flags (for Phase 8+)

- **§2.3 R9 closing sentence and §2.4 C3 both use the Occam's-razor framing with near-identical phrasing**. This is by design (§2.4 picks up the §2.3 rhetorical hook), but on a final editing pass the exact word-level overlap could be softened to avoid reading as a repeat.
- **`\cite{Wang2025MLOESCascaded}` now appears three times in the report** (§1.3, §2.3 R4 paragraph, §2.4 C5). This is legitimate — the same precedent supports three different claims — but is worth noting in case the reference density feels uneven to a marker.
- **Word budget accounting after Phase 7**: §1 ≈ 550 + §2 methodology ≈ 1000 + §2.3 ≈ 1200 + §2.4 ≈ 514 ≈ 3264. If the total budget ceiling for main sections is 4000, then §2.5 Future Work + §2.6 Reflection + §2.7 Summary together have ≈ 736 words remaining. §2.6 Reflection is currently a `\todo{}` placeholder plus ≈ 250 drafted words; §2.5 is an enumerated list of ≈ 200 words; §2.7 Executive Summary is also drafted. A tight audit in a later phase should confirm these fit the remaining envelope.

## Risks and follow-ups (out of Phase 7 scope)

- `\todo{Write personal reflection — approximately 200 words.}` still sits in §2.6 Reflection at line ≈ 494. Phase 8 or later should resolve.
- Appendices A, C, D, E retain `\todo{}` placeholders from earlier phases — unchanged in Phase 7.
- `figures/uol_logo.png` orphan reference (line 108) — still present, still harmless given `\IfFileExists` guard, still out of scope.
- Stefas2025 preprint-status monitoring and feature-justification citation additions — still pending user action.

## Summary verdict

Phase 7 delivers the Conclusion section as specified. All three objectives are resolved by name and by headline number; the R6 honesty rule is preserved; the permutation significance is attached only to the pruned 7-feature Ridge; the broader-implication paragraph closes the §1.3 loop. The 514-word draft is ≈ 3 % over the 500-word ceiling, a tolerable overrun for the number of claims the section must carry. §2.4 compiles cleanly and sits between §2.3 and §2.5 without disturbing the surrounding sections or the bibliography. Phase 7 is complete; the next step is a whole-report editing pass that would sit naturally in a Phase 8.