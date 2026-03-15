# Week 9: February 7 - 13, 2026

## Summary
Completed all Phase 2 tuning runs. Results showed universal improvement across all models — most dramatically MLP Config C improved from R² = -1.13 to 0.37 (eliminating the severe overfitting). However, the fundamental pattern of B >> C > A persisted, meaning that tuning alone cannot overcome the PCA bottleneck. This led to a key insight: the problem is not in the models, but in the feature representation. Began conceptualising a domain-knowledge feature engineering approach.

## Tasks Completed
- Completed MLP and CNN Optuna tuning (100 trials each, ~30 minutes total runtime)
- Phase 2 results summary:
  - MLP Config C: -1.13 -> 0.37 (most dramatic improvement, eliminated catastrophic overfitting)
  - CNN Config C: 0.69 -> 0.77 (marginal but consistent improvement)
  - RF Config B: 0.38 -> 0.75 (significant improvement with tuned tree depth)
  - All models improved universally — no regressions
- Generated cross-phase comparison tables and Optuna optimisation history plots
- **Key conclusion**: Config B still dominates; tuning refines within the existing feature space but cannot create better features
- Brainstormed Phase 3 approach: replace blind PCA with physically meaningful emission line features
  - The idea: instead of statistically decomposing 701 wavelengths into 11 abstract components, select specific wavelengths corresponding to known reactive species
  - Identified candidate diagnostic wavelengths from literature: OH 309 nm, O 777 nm, Halpha 656 nm, Hbeta 486 nm, N2 337 nm, CO2+ 406 nm, C2 516 nm
- Re-read sections of Shao 2018 and Mrozek 2021 to verify emission line assignments
- Consulted NIST Atomic Spectra Database (online) for wavelength verification of candidate features

## Papers Read
- Re-read Shao (2018). — Focused on emission line identification in atmospheric pulsed discharges; verified OH, O, H, N2 line assignments
- Re-read Mrozek (2021). — Cross-referenced plasma species identifications; confirmed CO2+ emission in the 398-412 nm range

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 4 |
| Coding / experiments | 5 |
| Data analysis | 5 |
| Writing / documentation | 3 |
| Meetings / discussion | 1 |
| **Total** | **18** |

## Next Week Plan
- Design the complete set of domain-knowledge OES features with physical justifications
- Define three feature categories: single-wavelength intensities, spectral band integrals, and intensity ratios
- Validate each candidate feature against the actual dataset for dynamic range and physical relevance
- Begin implementing the feature extraction pipeline
