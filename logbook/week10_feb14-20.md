# Week 10: February 14 - 20, 2026

## Summary
Designed and implemented the 13 hand-crafted OES features for Phase 3. Each feature is grounded in plasma chemistry: 7 single-wavelength intensities tracking specific reactive species, 3 spectral band integrals for noise-robust emission measurement, and 3 intensity ratios as self-normalising diagnostics. Also read new papers on ML for plasma-chemical process monitoring that validated this domain-knowledge approach.

## Tasks Completed
- Defined 13 OES features across 3 categories:
  - **Single-wavelength intensities (7)**: OH 309 nm (H2O2 precursor), O 777 nm (CO2 dissociation), Halpha 656 nm (H2O dissociation), Hbeta 486 nm (electron diagnostics), N2 337 nm (high-energy electron marker, ~11 eV threshold), CO2+ 406 nm (CO2 ionisation), C2 516 nm (deep decomposition indicator)
  - **Band integrals (3)**: OH 306-312 nm (rotational branches), CO2+ 398-412 nm (FDB band), CO+Hbeta 460-500 nm (composite)
  - **Intensity ratios (3)**: OH/Halpha (OH availability), O/OH (radical pool balance), Halpha/Hbeta (Balmer decrement, electron temperature/density diagnostic)
- Validated each feature against the actual dataset:
  - Verified sufficient dynamic range (CV and high/low intensity ratio)
  - Checked correlation with H2O2 rate
  - Excluded 6 candidate features that failed validation: I_297 (not independent peak), I_844 (low dynamic range), I_549 (low correlation), I_875 (spurious noise correlation), UV integral (near noise floor), total visible integral (too generic)
- Wrote detailed documentation in `phase3_feature_engineering.md` with physical justifications and NIST references
- Implemented `feature_engineer.py`: extracts all 13 features from raw OES spectra using `np.trapezoid` for band integrals
- Set up Phase 3 config with initial model parameters informed by Phase 2 insights
- Excluded CNN from Phase 3 — discrete tabular features negate the convolution advantage designed for sequential spectral data

## Papers Read
- Wang et al. (2021). "Machine learning prediction of tar from biomass pyrolysis." *J. Hazardous Materials*. — ML for plasma-chemical process monitoring; demonstrated that physically meaningful features outperform raw spectral input for small datasets
- (2024). *Journal of Energy Chemistry*. — Recent work on plasma-driven chemical synthesis with spectroscopic monitoring; validated the feature engineering approach for OES-based prediction

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 3 |
| Coding / experiments | 7 |
| Data analysis | 4 |
| Writing / documentation | 3 |
| Meetings / discussion | 1 |
| **Total** | **18** |

## Next Week Plan
- Run Phase 3 LOOCV evaluation with the engineered features
- Tune RF and MLP hyperparameters on the new feature set using Optuna
- Compare Phase 3 results against Phase 1 and Phase 2 baselines
- If results improve substantially, plan Phase 4 interpretability analysis
