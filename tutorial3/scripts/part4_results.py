"""
GROWL Tutorial 3 - Part 4: results and answer-key check.

Reads:
    C:\\GROWL_data\\COMPAS_Output_wWeights.h5     (for the answer key)
    C:\\GROWL_tutorial3\\output\\bbh_with_eta.parquet
    C:\\GROWL_tutorial3\\output\\sspc\\vanSon22_BBH_cosmic_integration.h5
    C:\\GROWL_tutorial3\\output\\R_BBH_vs_z.csv

Writes (to C:\\GROWL_tutorial3\\output\\figures\\):
    part4_Rz_vs_authors.png        - our R(z) vs the authors' answer key
    part4_Rz_channels.png          - CE vs SMT channel split
    part4_mass_distribution.png    - dR/dm1 at z=0.2, raw vs cosmic-integrated
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = r"C:\GROWL_data\COMPAS_Output_wWeights.h5"
CACHE_PATH  = r"C:\GROWL_tutorial3\output\bbh_with_eta.parquet"
SSPC_FILE   = r"C:\GROWL_tutorial3\output\sspc\vanSon22_BBH_cosmic_integration.h5"
OUT_FIG_DIR = r"C:\GROWL_tutorial3\output\figures"

# notebook constants
Z_SUN = 0.0142
N_Z_BINS = 50
CSYS, C_CE, C_SMT, C_WRONG = "#222222", "#E07B39", "#2A9D8F", "#999999"

CONV_Z_EDGES = np.arange(0.0, 6.41, 0.4)
CONV_Z       = 0.5 * (CONV_Z_EDGES[1:] + CONV_Z_EDGES[:-1])

RATES_GROUP = ("Rates_mu00.025_muz-0.05_alpha-1.77_"
               "sigma01.125_sigmaz0.05_a0.02_b1.48_c4.45_d5.9_zBinned")
Z_INDEX_02  = 4   # the authors' 0.20 < z < 0.25 bin

print("=" * 78)
print("Part 4 - results vs. the authors' answer key")
print("=" * 78)

# ---------------------------------------------------------------------------
# Load cached bbh table (has eta) and confirm SSPC output is intact
# ---------------------------------------------------------------------------
bbh = pd.read_parquet(CACHE_PATH)
print(f"loaded cached bbh table: {len(bbh):,} rows")

def read_sspc_yield(z_centre):
    with h5py.File(SSPC_FILE, "r") as f:
        return f[f"output_data/bbh/rates/convolution_results/{np.round(z_centre, 4)}/yield"][()]

R_sspc = np.array([read_sspc_yield(z).sum() for z in CONV_Z])
print(f"read SSPC yields at {len(CONV_Z)} redshifts")

# ---------------------------------------------------------------------------
# Read the authors' answer key (chunked - the array is ~3 GB in memory if whole)
# ---------------------------------------------------------------------------
print("\nReading authors' answer key from COMPAS file (this takes ~20-30 s)...")

with h5py.File(DATA_PATH, "r") as f:
    g = f[RATES_GROUP]
    z_edges_authors = g["redshifts"][()]
    z_authors       = 0.5 * (z_edges_authors[1:] + z_edges_authors[:-1])
    dco_mask        = g["DCOmask"][()]

    # We need the same selection the notebook uses: BBH & merges, then pull
    # out the rows within dco_mask's selection.
    DCO_COLUMNS = ["Stellar_Type(1)", "Stellar_Type(2)", "Merges_Hubble_Time"]
    with h5py.File(DATA_PATH, "r") as ff:
        type1   = ff["BSE_Double_Compact_Objects"]["Stellar_Type(1)"][()]
        type2   = ff["BSE_Double_Compact_Objects"]["Stellar_Type(2)"][()]
        merges  = ff["BSE_Double_Compact_Objects"]["Merges_Hubble_Time"][()]
    is_bbh_merge = (type1 == 14) & (type2 == 14) & (merges == 1)
    is_our_bbh   = is_bbh_merge[dco_mask]
    assert is_our_bbh.sum() == len(bbh), \
        f"row count mismatch: {is_our_bbh.sum()} vs {len(bbh)}"

    R_authors    = np.zeros(len(z_authors))
    R_authors_z0 = g["merger_rate_z0"][()][is_our_bbh].sum()
    rate_authors_z02 = np.empty(len(bbh), dtype=np.float64)

    n_rows = g["merger_rate"].shape[0]
    chunk = 100_000
    k = 0
    for i0 in range(0, n_rows, chunk):
        block = g["merger_rate"][i0:i0 + chunk, :]
        sel   = is_our_bbh[i0:i0 + chunk]
        if sel.any():
            R_authors += block[sel].sum(axis=0)
            rate_authors_z02[k:k + sel.sum()] = block[sel, Z_INDEX_02]
            k += int(sel.sum())

print(f"authors' bins: {len(z_authors)} bins of width {z_edges_authors[1]-z_edges_authors[0]:.2f} "
      f"from z = {z_edges_authors[0]:g} to z = {z_edges_authors[-1]:g}")
print(f"authors' BBH rate: z=0 exact {R_authors_z0:.1f}   "
      f"z~0.2 bin {R_authors[Z_INDEX_02]:.1f}   "
      f"peak {R_authors.max():.1f} at z={z_authors[np.argmax(R_authors)]:.3f}   [Gpc^-3 yr^-1]")
R_authors_at_02 = np.interp(0.2, z_authors, R_authors)
print(f"ours (SSPC)     : z=0.2 {R_sspc[0]:.1f}   peak {R_sspc.max():.1f} at z={CONV_Z[np.argmax(R_sspc)]:.1f}")
print(f"authors interp to z=0.2: {R_authors_at_02:.1f}   -> ratio ours/authors = {R_sspc[0]/R_authors_at_02:.3f}")

# ---------------------------------------------------------------------------
# Figure 1: our R(z) over the authors' answer key
# ---------------------------------------------------------------------------
print("\nBuilding figures...")

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(z_authors, R_authors, color=C_WRONG, lw=5, alpha=0.5,
        label="van Son et al. - authors' own integration (answer key)")
ax.plot([0], [R_authors_z0], "D", color=C_WRONG, ms=7, label="authors, exact z = 0")
ax.plot(CONV_Z, R_sspc, "o-", color=CSYS, lw=2, ms=6,
        label="this notebook (SSPC), all merging BBHs")
ax.set_yscale("log"); ax.set_ylim(5, 600); ax.set_xlim(0, 8)
ax.set_xlabel("merger redshift $z$")
ax.set_ylabel(r"$\mathcal{R}_{\rm BBH}(z)$  [Gpc$^{-3}$ yr$^{-1}$]")
ax.set_title("Binary black hole merger rate density - agreement with the answer key", loc="left")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, "part4_Rz_vs_authors.png"), dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: channel split (SMT = no CE events, CE = at least one)
# ---------------------------------------------------------------------------
is_smt = (bbh.n_CE == 0).values
R_sspc_ce  = np.array([read_sspc_yield(z)[~is_smt].sum() for z in CONV_Z])
R_sspc_smt = np.array([read_sspc_yield(z)[is_smt].sum()  for z in CONV_Z])

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(CONV_Z, R_sspc,     "o-", color=CSYS,  lw=2, ms=6, label="all merging BBHs")
ax.plot(CONV_Z, R_sspc_ce,  "s--", color=C_CE,  lw=1.6, ms=4, label="common-envelope channel")
ax.plot(CONV_Z, R_sspc_smt, "^--", color=C_SMT, lw=1.6, ms=4, label="stable-mass-transfer channel")
ax.set_yscale("log"); ax.set_ylim(5, 600); ax.set_xlim(0, 6.5)
ax.set_xlabel("merger redshift $z$")
ax.set_ylabel(r"$\mathcal{R}_{\rm BBH}(z)$  [Gpc$^{-3}$ yr$^{-1}$]")
ax.set_title("Channel split: common-envelope vs stable-mass-transfer", loc="left")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, "part4_Rz_channels.png"), dpi=130)
plt.close()

i_pk = int(np.argmax(R_sspc))
print(f"\npeak at z = {CONV_Z[i_pk]:.1f}: {R_sspc[i_pk]:.0f} Gpc^-3 yr^-1 "
      f"= {R_sspc[i_pk]/R_sspc[0]:.1f}x the z = 0.2 rate")
print(f"channel share at z = 0.2: CE {100*R_sspc_ce[0]/R_sspc[0]:.0f}%, "
      f"SMT {100*R_sspc_smt[0]/R_sspc[0]:.0f}%")
print(f"channel share at peak z = {CONV_Z[i_pk]:.1f}: "
      f"CE {100*R_sspc_ce[i_pk]/R_sspc[i_pk]:.0f}%, "
      f"SMT {100*R_sspc_smt[i_pk]/R_sspc[i_pk]:.0f}%")

# ---------------------------------------------------------------------------
# Figure 3: dR/dm1 at z = 0.2, cosmic-integrated vs raw population
# ---------------------------------------------------------------------------
yield_z02 = read_sspc_yield(0.2)
m1_bins   = np.arange(0, 70.5, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

ax = axes[0]
ax.hist(bbh.m1, bins=m1_bins, weights=rate_authors_z02, histtype="stepfilled",
        color=C_WRONG, alpha=0.35, label="authors' answer key (0.20 < z < 0.25 bin)")
ax.hist(bbh.m1, bins=m1_bins, weights=yield_z02, histtype="step", lw=2.2,
        color=CSYS, label="this notebook (SSPC)")
ax.hist(bbh.m1[~is_smt], bins=m1_bins, weights=yield_z02[~is_smt],
        histtype="step", lw=1.6, color=C_CE, label="common-envelope channel")
ax.hist(bbh.m1[is_smt],  bins=m1_bins, weights=yield_z02[is_smt],
        histtype="step", lw=1.6, color=C_SMT, label="stable-mass-transfer channel")
ax.set_yscale("log"); ax.set_ylim(1e-3, 30)
ax.set_ylabel(r"$\mathrm{d}\mathcal{R}/\mathrm{d}m_1$ at $z=0.2$  [Gpc$^{-3}$ yr$^{-1}$ M$_\odot^{-1}$]")
ax.set_title("The merger rate per unit primary mass, today", loc="left")
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(bbh.m1, bins=m1_bins, weights=bbh.w,     density=True,
        histtype="step", lw=2, color=C_WRONG, ls="--",
        label="Notebook 2: raw simulation (flat in log Z)")
ax.hist(bbh.m1, bins=m1_bins, weights=yield_z02, density=True,
        histtype="step", lw=2.2, color=CSYS,
        label="after cosmic integration, mergers at z = 0.2")
ax.set_yscale("log"); ax.set_ylim(1e-4, 0.3)
ax.set_ylabel("probability density [M$_\\odot^{-1}$]")
ax.set_title("Same binaries, different weights: what the Universe selects", loc="left")
ax.legend(fontsize=9)

for ax in axes:
    ax.set_xlabel(r"$m_1$, mass of the more massive black hole [M$_\odot$]")
    ax.set_xlim(0, 70)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, "part4_mass_distribution.png"), dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# Summary numbers: mass distribution, raw vs cosmic-integrated
# ---------------------------------------------------------------------------
def weighted_quantile(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, q * cw[-1])]

print("\n--- mass distribution summary ---")
for label, wts in [("raw simulation (Notebook 2)", bbh.w.values),
                   ("cosmic-integrated, z = 0.2",  yield_z02)]:
    med = weighted_quantile(bbh.m1.values, wts, 0.5)
    hi  = 100 * wts[bbh.m1.values > 30].sum() / wts.sum()
    print(f"  {label:30s}: median m1 = {med:5.1f} Msun,  "
          f"fraction with m1 > 30 Msun = {hi:4.1f}%")

print("\n" + "=" * 78)
print("Part 4 done.")
print("=" * 78)