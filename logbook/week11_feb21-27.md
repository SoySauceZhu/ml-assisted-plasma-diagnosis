# Week 11: February 21 - 27, 2026

## Summary
Phase 3 produced the most significant breakthrough of the entire project. Domain-knowledge feature engineering dramatically improved all models: Ridge Config C jumped from R² = -0.17 (Phase 1) to 0.80, and MLP Config C from -1.13 to 0.81 — both now surpassing Phase 2's best CNN result (0.77). This confirms the central hypothesis: on small datasets, domain expertise matters more than model complexity. Began planning the Phase 4 interpretability analysis.

## Tasks Completed
- Ran full Phase 3 LOOCV evaluation for 4 models (Ridge, PLS, RF, MLP) x 3 configs with engineered features
- Ran Optuna hyperparameter tuning for RF and MLP on the new 13-feature set (100 trials each)
- **Phase 3 results — the breakthrough**:
  - Ridge Config C: R² = 0.80 (up from -0.17 in Phase 1, a +0.97 improvement)
  - MLP Config C: R² = 0.81 (up from -1.13 in Phase 1)
  - PLS Config C: R² = 0.74
  - RF Config C: R² = 0.50 (improved but still underperforming)
  - Ridge Config B: R² = 0.90 (unchanged — discharge params remain strong)
- Config C no longer hurts performance relative to Config B — engineered OES features add genuine predictive information instead of PCA noise
- Generated cross-phase comparison table showing Phase 1 -> Phase 2 -> Phase 3 progression
- **Core insight**: simple linear models (Ridge) with well-engineered features match or beat neural networks (MLP) with raw/PCA features — domain knowledge is the decisive factor
- Planned Phase 4 analyses: SHAP values, feature importance stability, VIF for multicollinearity, backward elimination ablation, bootstrap confidence intervals, permutation significance testing

## Papers Read
- Biomass Pyrolysis Data documents. — Studied as a methodological reference for ML interpretability in chemical process prediction; the feature importance and ablation study approaches informed the Phase 4 design
- Machine learning prediction of pyrolytic products of lignocellulosic biomass (2022). — Demonstrated SHAP analysis and feature ablation for interpretable ML in chemical engineering; adapted their methodology for our Phase 4 plan

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 2 |
| Coding / experiments | 5 |
| Data analysis | 6 |
| Writing / documentation | 3 |
| Meetings / discussion | 1 |
| **Total** | **17** |

## Next Week Plan
- Implement Phase 4 interpretability analyses:
  - Multi-model feature importance (Ridge coefficients, PLS VIP, RF permutation importance, SHAP)
  - Bootstrap confidence intervals (500 resamples)
  - VIF analysis for multicollinearity among the 13 OES features
  - Backward elimination ablation study
  - Permutation significance test
  - Residual analysis with outlier identification
