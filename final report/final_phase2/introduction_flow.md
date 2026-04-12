# Introduction Writing Logic Flow (行文思路)

## 1. Literature Map

| Paper Key | Title (short) | Category | Role in Introduction | Key finding to cite |
|---|---|:---:|---|---|
| Gao2024NSCO2Discharge | Efficient synthesis of CO and H₂ via ns pulsed CO₂ bubble discharge | B | Dataset source; background on the plasma system | Nanosecond pulsed CO₂ bubble discharge for green chemistry |
| Shao2018PulsedDischarges | Atmospheric-pressure pulsed discharges and plasmas | B | Background on nanosecond pulsed discharge mechanisms | Review of pulsed discharge mechanisms and applications |
| Laux2003OESAir | Optical diagnostics of atmospheric pressure air plasmas | B | OES fundamentals; traditional spectroscopic methods | OES techniques, non-LTE challenges, Boltzmann/Saha methods |
| Gidon2019MLCAP | ML for real-time diagnostics of cold atmospheric plasma sources | A | Foundational OES+ML work; motivates ML approach | ML enables real-time plasma diagnostics from OES |
| Srikar2024Accelerated | Accelerated real-time plasma diagnostics: CR model + ML | A | OES+ML for electron temperature; uses RF/DNN | R² > 0.98 for Te prediction with ML |
| Srikar2025MLOES | ML-enabled prediction of OES in Ar plasma multi jet | A | PCA+RF/DNN approach — our "gap" target | Uses PCA for dimensionality reduction before ML |
| Wang2025MLOESCascaded | ML assisted OES for ne and Te in cascaded arc plasma | A | Uses line intensity ratios (similar to our approach!) | SVM + LIRs achieves R² ≈ 0.90–0.97; domain features work |
| Stefas2025MLSDBD | ML assisted optical diagnostics on cylindrical surface DBD | A | PCA+MLP for DBD plasma — our "gap" target | PCA used for feature extraction from OES |
| Wang2019MLOESAqueous | Data acquisition platform for ML of OES in aqueous solution | A | PCA fails, deep ANN needed — supports our narrative | PCA alone insufficient; deep ANN reduces MSE by 3 orders |
| Park2021MLOESNitrogen | ML prediction of ne and Te from OES in nitrogen plasma | A | ML virtual metrology from OES | 97% accuracy for ne, 90% for Te |
| Paris2005N2Ratio | N₂ spectral band intensity ratio as E-field measure | B | Traditional use of line ratios in spectroscopy | Line ratios have long been used for plasma diagnostics |
| Cai2024MLDRM | ML-driven optimization of plasma-catalytic DRM | C | ML in plasma catalysis (same supervisor group) | Domain-specific operating params for process prediction |
| Wang2021PlasmaTarML | Hybrid ML for plasma arc reforming of naphthalene | C | ML for plasma process with small data | GA-optimized hybrid ML on limited experimental data |
| Dong2023MLBiomass | ML prediction of pyrolytic products from biomass | C | Parallel finding: domain features > generic inputs | Physicochemical properties more important than conditions |
| Zhao2024TransferSSL | Transfer learning vs self-supervised learning review | D | Future work only (cross-reactor generalisation) | — |
| Liu2022UDA | Deep unsupervised domain adaptation review | D | Future work only (cross-reactor generalisation) | — |
| Yang2021MLAlN | ML classification of AlN film stress via OES | C | PCA+ANN for OES in manufacturing | PCA used for OES dimensionality reduction |

---

## 2. Introduction Logic Flow (行文思路)

### Backward Reasoning (先反向思考)

| Question | Answer |
|---|---|
| **What technical problem do we solve?** | PCA/PLS-based OES feature extraction discards physical meaning of emission lines, leading to poor ML prediction on small datasets. No established solution exists for domain-knowledge-based OES feature engineering in ML pipelines for chemical yield prediction. |
| **What are our contributions?** | (1) Systematic comparison of PCA vs. domain-knowledge features; (2) 13 physically motivated OES features from plasma chemistry; (3) Minimal 7-feature interpretable model matching neural networks; (4) Rigorous statistical validation framework |
| **Why does our method work?** | Spectroscopic ratios and band integrals encode physical relationships (species ratios, reaction pathways) that PCA cannot discover from small data. These features are also drift-invariant and calibration-robust. Traditional spectroscopy has long validated these features — we simply encode them for ML. |
| **How do we use prior methods to lead to our challenge?** | Prior OES+ML works (Srikar, Stefas, Wang2019) all use PCA → show it works for large datasets but fails on small ones. Wang2025 uses line ratios (like us) but only for Te/ne, not chemical yield. The gap: no one has systematically tested domain-knowledge features for yield prediction. |

### Forward Story (再正向写作)

| Step | Logic | Content | Key Citations |
|:---:|---|---|---|
| **L1** | Task | Real-time H₂O₂ yield prediction from OES in nanosecond pulsed CO₂ bubble plasma discharge | [Gao2024] |
| **L2** | Why it matters | Green chemistry demand; offline titration is slow/intrusive; real-time monitoring enables closed-loop control | [Shao2018] |
| **L3** | OES as diagnostic tool | OES provides non-intrusive window; traditional methods (Boltzmann, line-ratio) require expertise; ML can automate | [Laux2003], [Gidon2019] |
| **L4** | SOTA approaches | Recent OES+ML works use PCA on raw spectra; works for large datasets but discards physical meaning | [Srikar2025], [Stefas2025], [Wang2019] |
| **L5** | Root technical issue | PCA discards emission line identity → loses domain knowledge; on small datasets, PCA cannot discover task-relevant variance | [Wang2019] (PCA insufficient) |
| **L6** | Our insight (backed by tradition) | Traditional spectroscopy uses line ratios and band integrals for robust diagnostics; Wang2025 shows line ratios work for ML-based Te prediction; but this approach is not applied to chemical yield prediction | [Wang2025], [Paris2005], [Laux2003] |
| **L7** | Our solution | Domain-knowledge feature engineering: 13 features (7 lines + 3 bands + 3 ratios) from plasma chemistry literature; replace PCA entirely | — |
| **L8** | Why it works | Ratios are drift-invariant; bands integrate over physically meaningful ranges; features encode domain knowledge PCA cannot discover from small data | [Wang2025] (ratios work) |
| **L9** | Results preview | Ridge Config C R² from −0.17 → 0.80; minimal 7-feature model R² = 0.920 matching neural networks; permutation test p < 0.0005 | — |
| **L10** | Additional contributions | Statistical validation framework (bootstrap CI, permutation test); feature redundancy analysis; consensus feature importance | — |

---

## 3. Introduction Template Selection

### Part A — Introduce Task and Application

**Use Version 3: General → Specific Setting**

> "Plasma diagnostics and real-time process monitoring are critical for advancing green chemistry applications [Shao2018]. Among various diagnostic techniques, Optical Emission Spectroscopy (OES) provides a non-intrusive, in-situ window into plasma composition and energy [Laux2003]. This project focuses on a specific application: predicting hydrogen peroxide (H₂O₂) yield from OES measurements during nanosecond pulsed CO₂ bubble discharge [Gao2024]."

**Rationale:** The task (OES-based chemical yield prediction) is relatively niche. Start from the general field (plasma diagnostics) then narrow to the specific setting (H₂O₂ yield from CO₂ bubble discharge OES).

### Part B — Introduce Technical Challenge

**Use Technical-Challenge Version 2: Existing Task + Our Insight Backed by Traditional Methods**

Paragraph structure:
1. Recent OES+ML methods use PCA/PLS on raw spectra [Srikar2025, Stefas2025, Wang2019] → these work when data is abundant, but discard physical meaning
2. **Traditional spectroscopy has long used line ratios and band integrals** for robust diagnostics [Laux2003, Paris2005, Wang2025] — this is not new knowledge
3. However, modern ML pipelines **systematically overlook** this domain knowledge, defaulting to automated dimensionality reduction
4. The gap: no systematic comparison of PCA-based vs. domain-knowledge features for OES-based yield prediction exists

**Rationale:** This framing positions our insight as rediscovering and encoding established spectroscopic knowledge for ML — not as a novel feature engineering trick invented after PCA failed. This is more intellectually honest and avoids the "incremental patching" anti-pattern.

### Part C — Introduce Our Pipeline

**Use Pipeline Version 4: Observation-Driven**

Paragraph structure:
1. **Key innovation:** Replace PCA with 13 domain-knowledge OES features (emission lines, band integrals, spectroscopic ratios)
2. **Key observation:** This single change transforms Ridge regression R² on Config C from −0.17 to 0.80 — a swing of nearly 1.0
3. **Implementation:** 7 emission line intensities + 3 band integrals + 3 spectroscopic ratios, derived from plasma chemistry literature
4. **Technical advantage:** Features are physically interpretable, drift-robust (ratios normalize out absolute intensity variations), and dramatically reduce dimensionality (13 vs. 701)
5. **Further finding:** Feature reduction reveals a minimal 7-feature Ridge model (R² = 0.920) matching neural networks

**Rationale:** The contribution is driven by a single powerful observation (domain features transform performance), not by a complex pipeline. Version 4 (observation-driven) is the most natural fit.

---

## 4. Anti-Pattern Warning

> **DO NOT** write the Introduction as:
> "We first tried PCA (Phase 1), it failed; we tuned hyperparameters (Phase 2), still bad; we then used domain features (Phase 3), it worked. Finally we reduced features (Phase 4)."
>
> This makes the work look like incremental trial-and-error patching. The Introduction Writing Guide explicitly warns: *"Do not first present a naive solution and then describe our improvement over it. That writing makes the work look like a low-score incremental patch."*

> **DO** write:
> "Domain knowledge is the decisive factor in OES-based ML for plasma diagnostics. Traditional spectroscopy has long recognized the importance of line intensity ratios and band integrals [Laux2003, Paris2005, Wang2025], but modern ML approaches overlook this, relying instead on automated dimensionality reduction (PCA) [Srikar2025, Stefas2025]. We demonstrate that encoding this domain knowledge as ML features transforms prediction performance — from R² = −0.17 (PCA) to R² = 0.80 (domain features) for Ridge regression. A minimal 7-feature model achieves R² = 0.920, matching neural networks."

> **The four-phase structure (Phase 1–4) belongs in the Methodology section**, not in the Introduction. The Introduction should present the insight and contribution, not the chronological research process.

---

## 5. Paragraph-Level Plan for Introduction

| Para # | Role | Message (first sentence) | Content | Citations |
|:---:|---|---|---|---|
| 1 | Opening | Plasma-based green chemistry is an active field... | Introduce CO₂ bubble discharge, H₂O₂ production, real-time monitoring need | [Gao2024], [Shao2018] |
| 2 | Context | OES provides a non-intrusive diagnostic window... | OES principles, traditional methods, 701-wavelength challenge | [Laux2003] |
| 3 | Prior work | Recent studies have applied ML to automate OES analysis... | Survey PCA+ML approaches, their strengths on large datasets | [Gidon2019], [Srikar2025], [Stefas2025], [Park2021] |
| 4 | Challenge | However, PCA-based feature extraction discards physical meaning... | PCA limitations on small datasets; Wang2019 shows PCA insufficient | [Wang2019], [Srikar2024] |
| 5 | Insight | Traditional spectroscopy has long used line ratios... | Line ratios and band integrals as established diagnostic tools; Wang2025 validates for ML | [Laux2003], [Paris2005], [Wang2025] |
| 6 | Our method | In this project, we replace PCA with domain-knowledge features... | 13 features (7 lines + 3 bands + 3 ratios); R² from −0.17 → 0.80 | — |
| 7 | Results + contributions | Our experiments demonstrate three key findings... | (1) Domain > PCA; (2) 7-feature Ridge = 0.920; (3) permutation test p < 0.0005 | — |
| 8 | Objectives | This report addresses three objectives... | List 3 objectives | — |
