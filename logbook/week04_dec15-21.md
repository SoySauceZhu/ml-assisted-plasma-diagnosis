# Week 4: December 15 - 21, 2025

## Summary
Designed the complete Phase 1 experimental framework with 3 input configurations and 7 ML models. Started implementing the core pipeline modules: data loading, PCA analysis, and the first few model wrappers. Also read additional papers on plasma chemistry pathways relevant to feature interpretation.

## Tasks Completed
- Designed the 3-configuration comparison framework:
  - Config A: OES features only (PCA-reduced)
  - Config B: 4 discharge parameters only (frequency, pulse width, rise time, flow rate)
  - Config C: OES (PCA-reduced) + discharge parameters combined
- Selected 7 candidate models: Ridge Regression, PLS Regression, SVR (RBF kernel), XGBoost, Random Forest, MLP (PyTorch), 1D-CNN (PyTorch)
- Implemented `data_loader.py`: CSV loading, OES column extraction (I_200 to I_900), discharge parameter separation, target variable isolation
- Implemented `pca_analysis.py`: PCA fit with cumulative variance threshold at 95%, scree plot generation, component selection
- Implemented `config.py`: centralised hyperparameters, file paths, PCA threshold, random seed (42)
- Started implementing model wrappers: `ridge.py` (sklearn RidgeCV), `pls.py` (sklearn PLSRegression)
- Defined evaluation metrics: R-squared, RMSE, MAE

## Papers Read
- d5ja00260e (2025). *Journal of Analytical Atomic Spectrometry*. — OES feature extraction methods for quantitative plasma analysis; informed our approach to spectral feature selection
- 114103_1_online (2020). — Plasma chemistry reaction pathways and kinetics; helped map species relationships (CO2 -> CO + O, H2O -> OH + H, OH + OH -> H2O2)
- s44205-024-00098-7 (2024). — Plasma chemistry modelling approaches; provided theoretical framework for understanding why certain species correlate with H2O2 yield

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 5 |
| Coding / experiments | 5 |
| Data analysis | 2 |
| Writing / documentation | 2 |
| Meetings / discussion | 1 |
| **Total** | **15** |

## Next Week Plan
- Complete implementation of all 7 model wrappers (SVR, XGBoost, RF, MLP, CNN)
- Implement the LOOCV evaluation framework in `evaluation.py`
- Run PCA analysis and determine the number of components needed
- Aim to have a functional pipeline before the holiday break
