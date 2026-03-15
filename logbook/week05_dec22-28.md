# Week 5: December 22 - 28, 2025

## Summary
Pre-vacation push to get the Phase 1 pipeline fully functional. Completed all 7 model implementations and the LOOCV evaluation framework. Ran PCA analysis on the dataset — found that 11 components are needed for 95% variance, which is higher than expected. Initial test runs revealed training stability issues with MLP and CNN on the small dataset.

## Tasks Completed
- Implemented remaining model wrappers:
  - `svr.py`: sklearn SVR with RBF kernel
  - `xgboost_model.py`: XGBoost regressor wrapper
  - `rf.py`: Random Forest regressor
  - `mlp.py`: PyTorch MLP with architecture [32, 16] hidden layers, dropout 0.4, ReLU activations
  - `cnn.py`: PyTorch 1D-CNN with [16, 32] conv channels, kernel size 7, for raw 701-dim spectral input
- Implemented `evaluation.py`: LOOCV loop iterating over all model-config combinations, with StandardScaler fitted inside each fold to prevent data leakage
- Ran PCA on the full OES dataset: 11 components needed for 95% cumulative variance — this is worryingly high for 20 samples (11/20 = 0.55 feature-to-sample ratio after PCA)
- Generated scree plot and cumulative variance plot
- Debug session: CNN input shape handling for raw 701-dim spectrum required reshaping for 1D convolution layers
- MLP showed signs of overfitting in initial tests — training loss continued decreasing but validation error diverged
- Implemented `plotting.py` for predicted-vs-actual scatter plots

## Papers Read
- Validity of three-fluid plasma modelling for AC-DBD plasma actuators (2023). — Fluid modelling context for discharge physics; complemented the OES-based approach with physical modelling perspective
- dad5569supp1 (supplementary materials). — Additional methodological details on spectral data preprocessing and normalisation techniques

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 2 |
| Coding / experiments | 7 |
| Data analysis | 3 |
| Writing / documentation | 2 |
| Meetings / discussion | 1 |
| **Total** | **15** |

## Next Week Plan
- **VACATION**: December 29, 2025 - January 23, 2026
- Upon return: run the full Phase 1 LOOCV evaluation across all 7 models x 3 configurations
- Analyse and interpret Phase 1 results
