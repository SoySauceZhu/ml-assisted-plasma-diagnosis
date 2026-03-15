# Week 2: December 1 - 7, 2025

## Summary
Expanded the literature review to ML applications in plasma diagnostics and spectroscopy. Performed initial exploratory data analysis on the OES dataset and quickly identified the central technical challenge: extreme dimensionality (701 features from only 20 samples).

## Tasks Completed
- Surveyed ML methods commonly applied to spectral data: PCA for dimensionality reduction, PLS regression (the gold standard in chemometrics), SVR, Random Forest, and neural networks (MLP, 1D-CNN)
- Loaded and explored `oes_ml_dataset_1nm.csv` in Python — plotted raw spectra for all 20 samples
- Identified prominent spectral peaks visually: OH emission near 309 nm, atomic O at 777 nm, Halpha at 656 nm
- Noted the "curse of dimensionality" as the central methodological challenge: p/n ratio = 701/20 = 35, meaning most ML models will overfit severely without dimensionality reduction
- Began reading about PCA theory and its application to spectroscopic data
- Reviewed a materials science paper on ML applications for spectral classification

## Papers Read
- JPhysD-140473 (group's paper, proof copy). — Detailed the OES measurement methodology and spectral acquisition setup; helped understand the data generation process
- materials-14-04445 (2021). "Machine Learning in Materials Science." — Reviewed general ML pipelines for materials characterisation; noted the importance of feature engineering for small datasets

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 7 |
| Coding / experiments | 3 |
| Data analysis | 1 |
| Writing / documentation | 1 |
| Meetings / discussion | 1 |
| **Total** | **13** |

## Next Week Plan
- Deep dive into PCA and PLS regression for spectroscopy
- Review more papers on plasma diagnostics and OES analysis methods
- Begin drafting the ML pipeline architecture: preprocessing -> dimensionality reduction -> model training -> evaluation
