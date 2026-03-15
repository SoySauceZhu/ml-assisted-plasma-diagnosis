# Week 3: December 8 - 14, 2025

## Summary
Intensive literature review week — read 12 papers spanning plasma diagnostics, OES analysis, chemometrics, and ML for spectroscopy. Solidified the understanding of PCA as the initial dimensionality reduction strategy and LOOCV as the only viable cross-validation approach for n=20. Drafted the overall ML pipeline design.

## Tasks Completed
- Read a large batch of papers on plasma diagnostics, pulsed discharges, OES-based process monitoring, and ML for spectral data
- Studied PCA theory in depth: cumulative explained variance, scree plot interpretation, loading vectors, and the trade-off between variance retention and overfitting
- Identified PLS regression as the standard chemometric method for spectral prediction — it simultaneously reduces dimensions and regresses on the target
- Decided on LOOCV (Leave-One-Out Cross-Validation) as the evaluation strategy: with n=20, k-fold CV would have too few samples per fold; LOOCV maximises training data per fold
- Drafted the Phase 1 pipeline: Raw OES -> StandardScaler -> PCA (95% variance threshold) -> ML models -> H2O2 prediction
- Began thinking about model selection — Ridge and PLS for linear baselines, SVR/RF/XGBoost for non-linear, MLP/CNN for deep learning comparison

## Papers Read
- Balazinski (2025). *J. Phys. D: Appl. Phys.* 58, 295202. — ML approaches for spectroscopic data analysis; useful for understanding state-of-the-art methods
- Shao (2018). "Atmospheric-pressure pulsed discharges and plasmas." *High Voltage*. — Detailed mechanisms of pulsed discharge similar to our experimental setup; OES diagnostic techniques for species identification
- Mrozek (2021). *Plasma Sources Sci. Technol.* 30, 125007. — Plasma diagnostics methodology with OES; validated emission line assignments for key species
- Wang (2019). *Plasma Sources Sci. Technol.* 28, 105013. — Spectroscopy-based plasma process monitoring; demonstrated real-time OES applications
- 1-s2.0-S0957417423033092 (2023). Expert systems for spectral analysis. — ML architectures for spectral classification and regression
- 1-s2.0-S2352179125001346 (2025). Recent advances in plasma ML. — State-of-the-art review of ML in plasma science
- Advances in Plasma Diagnostics and Applications (textbook chapter). — Survey of optical diagnostic techniques including OES, actinometry, and Boltzmann plot methods
- 2208.07422v1 (2022). — Relevant methodology for spectral data preprocessing and feature extraction
- s41597-020-0396-8 (2020). — Open spectral dataset practices; informed our approach to data handling
- s41597-025-05203-5 (2025). — Similar small-sample spectral dataset; validated our LOOCV approach
- materials-16-03846 (2023). — Materials characterisation via spectroscopy and ML
- processes-10-00654 (2022). — Process monitoring with OES in industrial plasma applications

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 9 |
| Coding / experiments | 2 |
| Data analysis | 0 |
| Writing / documentation | 2 |
| Meetings / discussion | 1 |
| **Total** | **14** |

## Next Week Plan
- Finalise Phase 1 experimental design: define 3 input configurations (OES only, discharge params only, combined)
- Begin implementing the data loading and PCA pipeline in Python
- Select and implement the 7 candidate ML models
