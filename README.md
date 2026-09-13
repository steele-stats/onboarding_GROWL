# GROWL Onboarding — Population Synthesis Analysis

Onboarding exercise for the GROWL Task Force. Analyzes a COMPAS population
synthesis catalog of double compact objects to identify the PISN gap and
separate the common-envelope vs. stable-mass-transfer formation channels.

## Contents

- `onboarding_GROWL.qmd` — Quarto source (prose + R code)
- `onboarding_GROWL.html` — rendered output

## Data

Requires `COMPAS_Output_wWeights.h5` from the GROWL dataset. The HDF5 file
is not included in this repository.

## Requirements

- R (≥ 4.4)
- R packages: `rhdf5`, `dplyr`, `ggplot2`
- Quarto

## What it does

1. Loads the `BSE_Double_Compact_Objects` table from the COMPAS output.
2. Compares raw row counts vs. statistical weights by compact-object type.
3. Plots the primary-mass distribution, showing the PISN gap near 46 M⊙.
4. Reconstructs orbital separation from Peters (1964) and splits by
   formation channel (CE vs. SMT) in the (a, q) plane.

## Author

Mike Steele