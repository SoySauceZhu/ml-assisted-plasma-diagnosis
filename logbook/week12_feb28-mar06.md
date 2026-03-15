# Week 12: February 28 - March 6, 2026

## Summary
Implemented the comprehensive Phase 4 interpretability and statistical validation suite. Key discoveries: (1) the 13 OES features contain severe multicollinearity — 9 of 13 have VIF > 10; (2) backward elimination monotonically improves Ridge Config C from R² = 0.798 to 0.918 by removing redundant features; (3) a minimal model with just 3 intensity ratios + 4 discharge parameters achieves the best overall result (R² = 0.920, p < 0.0005); (4) MLP feature attributions are unreliable at n=20 (82% instability rate).

## Tasks Completed
- **Multi-model feature importance** (`interpretability.py`):
  - Extracted Ridge absolute coefficients, PLS VIP scores, RF permutation importance, MLP SHAP values
  - Consensus ranking: flow_rate_sccm (#1), band_CO2p_398_412 (#2), pulse_width_ns (#3), I_486_Hb (#4)
  - Counterintuitive: I_309_OH (direct OH emission) ranks only #16 — its signal is captured more reliably by band_OH and ratio_309_656
- **Bootstrap confidence intervals** (500 resamples):
  - Ridge Config C: R² = 0.798, 95% CI [0.574, 0.910]
  - MLP Config C: R² = 0.815, 95% CI [0.647, 0.883]
  - CIs overlap extensively — the R² difference is not statistically significant
- **Feature importance stability** (CV across LOOCV folds):
  - Ridge: only 1/17 features unstable (6% instability rate) — most trustworthy
  - MLP: 14/17 features unstable (82% instability rate) — unreliable for interpretation
- **VIF analysis** (`feature_redundancy.py`):
  - 9/13 OES features have VIF > 10: I_309_OH (381.7), band_OH_306_312 (318.6), band_CO2p_398_412 (104.9), I_656_Ha (78.5), etc.
  - Only 4 features with VIF < 10: I_516_C2, I_337_N2, ratio_309_656, ratio_777_309
- **Backward elimination ablation** (Ridge Config C):
  - Starting from all 17 features (R² = 0.798), iteratively removing the least important feature
  - R² monotonically improves: 0.798 -> 0.856 -> 0.890 -> 0.918 (peak at 5 total features)
  - Peak model: ratio_656_486 + 4 discharge parameters -> R² = 0.918
  - Category ablation: 3 intensity ratios + 4 discharge = R² = 0.906 (best fixed category)
- **Permutation significance test** (pruned 7-feature model):
  - Observed R² = 0.920; p-value < 0.0005 (2000 permutations) — statistically significant
- **Residual analysis** (`residual_analysis.py`):
  - Sample 9 (pulse_width = 2000 ns, H2O2 rate = 0.83) identified as consistent outlier
  - Ridge prediction: 0.579 (residual = -0.251); MLP prediction: 0.624 (residual = -0.206)
  - Residual magnitude correlates with pulse_width_ns (r = 0.46, p = 0.04) — model underperforms at extreme discharge conditions, likely non-linear saturation behaviour

## Papers Read
- 1-s2.0-S0584854724000533 (2024). *Spectrochimica Acta Part B*. — Spectral feature selection methods for quantitative OES analysis; validated our backward elimination approach
- Paris (2005). *J. Phys. D: Appl. Phys.* 38, 3894. — Classical plasma spectroscopy reference; Boltzmann plot methods and actinometry for species quantification
- Srikar (2025). *J. Phys. D: Appl. Phys.* 58, 415204. — Recent ML interpretability methods in scientific applications; SHAP analysis best practices
- Urabe (2016). *Plasma Sources Sci. Technol.* 25, 045004. — OES quantification in plasma-liquid systems; relevant to our H2O2 prediction context
- coatings-11-01221 (2021). — Plasma process monitoring with OES diagnostics; feature importance analysis for coating process control

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 2 |
| Coding / experiments | 4 |
| Data analysis | 5 |
| Writing / documentation | 4 |
| Meetings / discussion | 1 |
| **Total** | **16** |

## Next Week Plan
- Compile all results into the final report
- Prepare presentation slides and poster for bench inspection
- Write the project conclusion synthesising findings across all four phases
- Final code cleanup and documentation
