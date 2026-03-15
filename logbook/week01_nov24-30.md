# Week 1: November 24 - 30, 2025

## Summary
Project officially kicked off this week. Met with supervisor to discuss the research scope: using machine learning to predict H2O2 yield rate from nanosecond pulsed CO2 bubble discharge plasma, based on Optical Emission Spectroscopy (OES) data. Began reading the group's foundational publications to understand the experimental setup and plasma chemistry.

## Tasks Completed
- Attended project kickoff meeting; received the OES dataset (`oes_ml_dataset_1nm.csv`) and background materials
- Read and annotated the group's published paper (Gao et al., 2024) thoroughly — understood the full experimental setup: nanosecond pulsed discharge in CO2 bubbles submerged in water
- Reviewed the dataset structure: 20 experimental samples, 701 OES wavelength columns (200-900 nm at 1 nm resolution), 4 discharge parameters (frequency, pulse width, rise time, flow rate), and H2O2 yield rate as the target variable
- Noted the factorial experimental design: 4 parameter groups x 5 levels each
- Reviewed Alliati's end-of-year report and PhD plan for additional context on the research group's prior work in plasma-liquid interactions
- Set up the project Git repository and folder structure

## Papers Read
- Gao et al. (2024). "Plasma-liquid interactions: a review and roadmap." *J. Phys. D: Appl. Phys.* 57, 375204. — Comprehensive review of CO2 bubble discharge mechanisms; identified key reactive species (OH, O, H, CO2+, N2, C2) and the H2O2 formation pathway: OH + OH -> H2O2
- Alliati (2018). End of year report. — Background on plasma diagnostics techniques used in the group's earlier experiments
- Alliati (2017). PhD Plan. — Overview of research group's long-term goals in plasma-driven chemical conversion

## Hours Spent
| Activity | Hours |
|----------|-------|
| Literature review | 9 |
| Coding / experiments | 1 |
| Data analysis | 0 |
| Writing / documentation | 2 |
| Meetings / discussion | 1 |
| **Total** | **13** |

## Next Week Plan
- Begin surveying ML methods used in plasma diagnostics and spectroscopy applications
- Perform initial exploratory data analysis on the OES dataset
- Identify the key challenge of high-dimensional spectral data with very few samples
