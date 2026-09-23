# Tutorial 3 — Cosmic integration: from a simulation to a merger rate

GROWL Task Force onboarding, exercise 3 (of 3).

This folder contains my implementation of
[Tutorial 3](https://github.com/FloorBroekgaarden/GROWL-catalog-public/blob/main/onboarding_growl/tutorial_3_cosmic_integration_predicting_merger_rates.ipynb)
— turning the van Son et al. (2022) COMPAS population of merging binary black
holes into a merger rate density R(z) and a primary-mass distribution dR/dm1 at
z = 0.2, and comparing both to GWTC-5.0.

## Contents

- `scripts/` — four Python scripts, run in order:
  - `part1_formation_efficiency.py` — Kroupa IMF, star-forming mass per simulated
    binary, formation efficiency eta_BBH(Z)
  - `part3_cosmic_integration.py` — builds the metallicity-specific star formation
    history S(Z, z) and runs the SSPC cosmic integration
  - `part4_results.py` — the two headline plots and the check against the authors'
    own answer key stored in the COMPAS file
  - `part5_lvk_comparison.py` — comparison to the LVK GWTC-5.0 population results
- `figures/` — PNGs produced by the scripts
- `output/R_BBH_vs_z.csv` — the merger rate table in Gpc^-3 yr^-1

## Environment

Python 3.10, plus: `h5py numpy matplotlib pandas scipy astropy tables pyarrow`,
and SSPC 0.4 from
[github.com/FloorBroekgaarden/growl-syntheticstellarpopconvolve](https://github.com/FloorBroekgaarden/growl-syntheticstellarpopconvolve).

## Data (not included — too large)

- `COMPAS_Output_wWeights.h5` (5.6 GB) — van Son et al. (2022) population,
  DOI [10.5281/zenodo.7612755](https://doi.org/10.5281/zenodo.7612755)
- `gwtc5_bbh_rate_and_mass.h5` (7 MB) — GWTC-5.0 extract from
  [GROWL-catalog-public](https://github.com/FloorBroekgaarden/GROWL-catalog-public)
  (original: DOI [10.5281/zenodo.20292639](https://doi.org/10.5281/zenodo.20292639))

## Results verified against the tutorial notebook

| Quantity | Value |
|---|---|
| Star-forming mass per simulated binary | 101.7 Msun |
| eta_BBH (mean) | 1.47e-5 per Msun (one per ~68,000 Msun) |
| R_BBH(z=0.2) | 67.5 Gpc^-3 yr^-1 |
| R_BBH peak | 284.5 Gpc^-3 yr^-1 at z = 2.6 |
| Ratio ours / authors' answer key at z=0.2 | 0.997 |
| CE / SMT channel split at z=0.2 | 70% / 30% |
| Median m1, raw vs cosmic-integrated | 16.6 vs 12.7 Msun |
| LVK R(z=0.2) | 30.4 (90%: 25.0-36.8) Gpc^-3 yr^-1 |
| Simulation / observation at z=0.2 | 2.22 |
| Redshift slope kappa (sim vs LVK) | 1.53 vs 2.54 |

## Tasks 1-3 (write-ups)

Pending. Will be added as `TASKS.md` once written.