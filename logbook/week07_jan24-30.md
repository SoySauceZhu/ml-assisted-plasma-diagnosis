# Week 7: January 24 - 30, 2026

## Summary
Returned from the holiday break. Ran the full Phase 1 LOOCV evaluation across all 7 models and 3 configurations (21 total combinations). The results were surprising: Config B (discharge parameters only) dominated all others, with Ridge and PLS both achieving R-squared of approximately 0.90, while OES-based configurations performed poorly or even negatively. This was not what I expected going in.

## Tasks Completed
- Ran complete Phase 1 LOOCV evaluation for all 21 model-config combinations
- Generated results summary tables and predicted-vs-actual scatter plots for each combination
- Key Phase 1 findings:
  - **Config B (discharge params only)**: Ridge R² = 0.90, PLS R² = 0.90 — best results by far
  - **Config A (OES only, PCA)**: Ridge R² = -0.31, PLS R² = -0.60 — worse than predicting the mean
  - **Config C (OES + params, PCA)**: Ridge R² = -0.17, PLS R² = 0.63, CNN R² = 0.69
  - MLP Config C: R² = -1.13 — severe overfitting, predictions wildly off
  - XGBoost collapsed to near-trivial predictions across all configs
  - CNN Config C (R² = 0.69) was the only model where adding OES improved over Config B baseline
- Analysed PCA loading plots to understand which spectral regions the 11 components captured
- Concluded that the 11 PCA components from 20 samples still carry too much noise — the curse of dimensionality is the fundamental bottleneck
- Re-read sections of Gao 2024 to interpret results in the context of plasma chemistry

## Papers Read
- Re-read Gao et al. (2024) — focused on the plasma chemistry pathway sections to understand why discharge parameters (which directly control the plasma energy input) predict H2O2 yield so well, while OES (which measures secondary emission) fails under PCA

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 3 |
| Coding / experiments | 5 |
| Data analysis | 6 |
| Writing / documentation | 2 |
| Meetings / discussion | 1 |
| **Total** | **17** |

## Next Week Plan
- Design Phase 2: hyperparameter tuning strategy using Bayesian optimisation (Optuna) for RF, MLP, and CNN — the models with the most room for improvement
- Research Optuna and TPE (Tree-structured Parzen Estimator) sampling for efficient hyperparameter search
- Consider whether the PCA strategy itself needs to be rethought
