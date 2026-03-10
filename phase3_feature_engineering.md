# Phase 2: Domain-Knowledge-Driven OES Feature Engineering

## Overview

Based on the plasma chemistry of nanosecond pulsed CO₂ bubble discharge in water (Gao et al., 2024) and data-driven verification on the 20-sample OES dataset, we propose **13 handcrafted features** (p/n = 0.65) to replace the 11-component blind PCA used in Phase 1. Every feature is grounded in an identified reactive species or a recognized spectroscopic diagnostic technique, and has been validated against the actual spectral data for sufficient signal dynamic range and physical relevance.

## Feature Definitions

### Category 1: Single-Wavelength Intensities (7 features)

**F1 — I(OH 309 nm): OH radical (A²Σ⁺→X²Π, 0-0 band head)**
The most critical feature. OH is the direct precursor of H₂O₂ via the recombination pathway ·OH + ·OH → H₂O₂ in both the gas phase and at the gas–liquid interface. In the dataset, I_309 has a coefficient of variation (CV) of 0.54 and a high/low sample intensity ratio of 5.0, confirming strong dynamic response to changing discharge conditions.

**F2 — I(O 777 nm): Atomic oxygen (⁵S°→⁵P triplet at 777.4 nm)**
Atomic O is the primary product of electron-impact CO₂ dissociation (CO₂ + e⁻ → CO + O + e⁻) and also contributes to OH generation through O + H₂O → 2·OH. This line exhibits the largest dynamic range in the entire spectrum (intensity ratio = 10.86, CV = 0.53, r = 0.58 with H₂O₂ rate).

**F3 — I(Hα 656 nm): Hydrogen Balmer-alpha (n=3→2 at 656.3 nm)**
Hα originates from electron-impact dissociation of H₂O (H₂O + e⁻ → ·OH + H + e⁻). Its intensity reflects the overall degree of water molecule decomposition, which is the upstream process generating OH radicals. Highest absolute intensity in the spectrum (mean = 5405), CV = 0.61.

**F4 — I(Hβ 486 nm): Hydrogen Balmer-beta (n=4→2 at 486.1 nm)**
Paired with Hα for the Balmer decrement diagnostic (see F13). Independently, Hβ tracks H₂O dissociation at higher excitation energies. CV = 0.35, r = 0.52 with H₂O₂ rate.

**F5 — I(N₂ 337 nm): N₂ second positive system (C³Πᵤ→B³Πg, 0-0 band at 337.1 nm)**
The N₂(C) state has an excitation threshold of ~11 eV, making this emission a sensitive marker for high-energy electron density. Higher electron energies drive more dissociation of both CO₂ and H₂O. Despite moderate absolute intensity (mean = 981), it has the highest single-wavelength correlation with H₂O₂ rate among all identified emission lines (r = 0.57).

**F6 — I(CO₂⁺ 406 nm): CO₂⁺ Fox–Duffendack–Barker band (A²Πᵤ→X²Πg)**
CO₂⁺ is produced by electron-impact ionization of CO₂ (CO₂ + e⁻ → CO₂⁺ + 2e⁻). The 398–412 nm region forms a coherent high-correlation cluster (r = 0.63–0.72), indicating that the degree of CO₂ ionization is strongly linked to H₂O₂ production. I_406 is chosen as the representative peak of this band (r = 0.63, CV = 0.32).

**F7 — I(C₂ 516 nm): C₂ Swan band (d³Πg→a³Πᵤ, Δv = 0 band head at ~516 nm)**
C₂ radicals require deep decomposition of CO (CO → C + O, then C + C → C₂), serving as an indicator of extreme dissociation conditions. CV = 0.45, r = 0.41.

### Category 2: Spectral Band Integrals (3 features)

**F8 — ∫I(306–312 nm): OH (A-X) 0-0 band integral**
Integrating over the full P, Q, R rotational branches of the OH 0-0 band provides a more noise-robust measure of total OH emission than the single-point I_309, and reduces sensitivity to minor wavelength calibration shifts.

**F9 — ∫I(398–412 nm): CO₂⁺ FDB band integral**
Captures the entire CO₂⁺ emission feature whose individual wavelengths all show r > 0.65 with H₂O₂ rate. Serves as an aggregate indicator of CO₂ ionization intensity.

**F10 — ∫I(460–500 nm): CO Angstrom + Hβ composite band integral**
This spectral region contains overlapping CO Angstrom system (B¹Σ⁺→A¹Π) and the Hβ line, jointly reflecting CO₂ dissociation product (CO) abundance and H₂O dissociation degree.

### Category 3: Intensity Ratios (3 features)

**F11 — I(309) / I(656): OH-to-Hα ratio**
Since H₂O dissociation simultaneously produces OH and H (H₂O + e⁻ → OH + H + e⁻), a higher OH/H ratio indicates that more OH radicals are available for recombination into H₂O₂ rather than being consumed in competing pathways. This ratio is also self-normalizing, cancelling out fluctuations in total plasma emission intensity.

**F12 — I(777) / I(309): Atomic O-to-OH ratio**
Characterizes the relative balance between atomic oxygen and OH in the radical pool. The reaction O + H₂O → 2·OH converts atomic O into OH, so this ratio reflects the state of that conversion process. A high O/OH ratio may indicate that the indirect OH production pathway (via atomic O) still has capacity to contribute additional OH for H₂O₂ formation.

**F13 — I(656) / I(486): Hα-to-Hβ Balmer decrement**
A classical plasma diagnostic quantity. In an optically thin plasma under Case B recombination, the theoretical Hα/Hβ ratio is ~2.87. Deviations indicate changes in electron density, electron temperature, or optical thickness of the plasma, all of which affect the reactive species production rates.

## Features Excluded After Data Verification

| Previously proposed | Reason for exclusion |
|---|---|
| I_297 (NO γ) | Not an independent peak; the 296 nm signal is the tail of the OH (A-X) 0-1 vibrational band, collinear with I_309 |
| I_844 (O atom) | Dynamic range too low (high/low ratio = 1.19); signal indistinguishable from continuum background |
| I_549 (C₂ Swan Δv=-1) | Very low correlation with H₂O₂ (r = 0.20); the Δv=0 band at 516 nm already captures C₂ information |
| I_875 region | No identifiable emission line; CV = 0.044 (flat continuum); the high r = 0.75 is a spurious noise correlation |
| ∫(200–250 nm) UV integral | Signal near detector noise floor (mean ~590); no diagnostic value |
| ∫(400–900 nm) total integral | Too generic; loses species-specific information provided by targeted band integrals |

## Summary Table

| ID | Feature | Type | Species / Diagnostic | Role in H₂O₂ chemistry |
|---|---|---|---|---|
| F1 | I(309 nm) | Intensity | OH (A-X) 0-0 | Direct H₂O₂ precursor |
| F2 | I(777 nm) | Intensity | O (⁵S°→⁵P) | CO₂ dissociation product; indirect OH source via O+H₂O→2OH |
| F3 | I(656 nm) | Intensity | Hα | H₂O dissociation degree |
| F4 | I(486 nm) | Intensity | Hβ | H₂O dissociation; electron diagnostics |
| F5 | I(337 nm) | Intensity | N₂ SPS 0-0 | High-energy electron density marker |
| F6 | I(406 nm) | Intensity | CO₂⁺ FDB | CO₂ ionization degree |
| F7 | I(516 nm) | Intensity | C₂ Swan Δv=0 | Deep CO₂ decomposition indicator |
| F8 | ∫(306–312 nm) | Band integral | OH 0-0 band | Robust total OH emission |
| F9 | ∫(398–412 nm) | Band integral | CO₂⁺ FDB band | Aggregate CO₂ ionization |
| F10 | ∫(460–500 nm) | Band integral | CO Angstrom + Hβ | CO abundance + H₂O dissociation |
| F11 | I(309)/I(656) | Ratio | OH / Hα | OH availability for recombination |
| F12 | I(777)/I(309) | Ratio | O / OH | Radical pool balance |
| F13 | I(656)/I(486) | Ratio | Hα / Hβ | Electron temperature / density diagnostic |


# Why choosing these frequencies
> NIST Atomic Spectr Database

## F1 OH-309mm
> Plasma–liquid interactions: a review and roadmap
> Gao et al.

In Gao's article, the process pathway (liquid/gas phase) is OH(aqueous/gas) + OH = H2O2(aq/gas) 

## F2 O-777nm
CO2 + e = CO + O

O + H = OH

## F3 F4 H alpha/beta Balmer

H2O + e- = OH + H


## F5 N2
> Intensity ratio of spectral bands of nitrogen as a measure of electric field strength in plasmas

等离子体中电子能量诊断量的方法论基础: the dependence of nitrogen spectral band intensity ratios (R₃₉₁/₃₃₇ and R₃₉₁/₃₉₄) on reduced electric field strength E/N across a wide range for the first time, deriving empirical formulae (equations 2–4) that can be used for E/N estimation in low-temperature air plasmas.

## F6 CO2+

Indicate the dissociation of CO2

CO2 + e- = ...

## Integral Band:

Robustness ensurance. Due to measurement error/ noises, the spectra line might drift.

## F11: OH/H alpha

Since the inital process that generate OH and H is 1:1. "H2O -> H + OH"
After that OH and H goes into different subsequent reactions. The ratio thus indicates the amount of two substances that goes into different pathway. If more OH are associated with each other, then apparently more H2O2 is generated. 

And also here we use **relative abundances between species**. "Relative" means it eliminates the flucutation of overall plasma intensity

## F12 O/OH
Same logic as OH/H 

## F13 H alpha/beta
> Griem, Principles of Plasma Spectroscopy, Cambridge University Press (1997), Chapter 7
> Optical diagnostics of atmospheric pressure air plasmas

It indicates electron concentration. 