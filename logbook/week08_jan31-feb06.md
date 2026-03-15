# Week 8: January 31 - February 6, 2026

## Summary
Designed and implemented the Phase 2 Bayesian hyperparameter tuning pipeline using Optuna. Focused on tuning RF, MLP, and CNN — the three models showing the most potential for improvement. Also read new papers on ML optimisation methods and plasma process control, which reinforced the importance of proper hyperparameter selection for small datasets.

## Tasks Completed
- Installed and configured Optuna for Bayesian optimisation with TPE (Tree-structured Parzen Estimator) sampler
- Designed hyperparameter search spaces:
  - RF: n_estimators [50-500], max_depth [2-10], min_samples_split [2-5], min_samples_leaf [1-3], max_features
  - MLP: hidden layer sizes, number of layers [1-3], dropout rate [0.1-0.5], weight decay, learning rate [1e-4 to 1e-2], batch size
  - CNN: conv channels, kernel sizes, pooling type, FC layer sizes, dropout
- Chose a single inner LOOCV approach: run Optuna on all 20 samples to find best hyperparameters, then evaluate with outer LOOCV using those fixed parameters
- Implemented `tuner_rf.py` using RandomizedSearchCV with LOOCV scoring
- Implemented `tuner_mlp.py` and `tuner_cnn.py` using Optuna with 100 trials each
- Ran RF tuning first (completed in ~2 minutes) — significant improvement in Config B
- Started MLP and CNN tuning runs (longer due to neural network training overhead)

## Papers Read
- 2404.06817v2 (2024). — Advanced ML methodology paper; discussed Bayesian optimisation strategies for scientific applications with limited data, reinforcing our choice of Optuna over grid search
- IEEE Plasma 2019. — IEEE paper on plasma process optimisation with ML; demonstrated successful application of tuned ML models for plasma parameter prediction, motivating Phase 2 approach

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 3 |
| Coding / experiments | 9 |
| Data analysis | 3 |
| Writing / documentation | 2 |
| Meetings / discussion | 1 |
| **Total** | **18** |

## Next Week Plan
- Complete all Phase 2 tuning runs and analyse results
- Compare Phase 2 (tuned) vs Phase 1 (default) performance across all model-config pairs
- Assess whether tuning alone can overcome the PCA dimensionality bottleneck
- Begin thinking about alternative dimensionality reduction strategies if the B >> C pattern persists
