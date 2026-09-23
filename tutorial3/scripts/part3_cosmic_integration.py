"""
GROWL Tutorial 3 - Part 3: cosmic integration with SSPC.

Ports Part 3 of tutorial_3_cosmic_integration_predicting_merger_rates.ipynb
to a plain Python script. Run from PowerShell:

    cd C:\GROWL_tutorial3\scripts
    python part3_cosmic_integration.py

Reads:
    C:\\GROWL_tutorial3\\output\\bbh_with_eta.parquet    (cached from Part 1)

Writes:
    C:\\GROWL_tutorial3\\output\\sspc\\vanSon22_BBH_cosmic_integration.h5
    C:\\GROWL_tutorial3\\output\\figures\\part3_SFR_and_Z.png
    C:\\GROWL_tutorial3\\output\\figures\\part3_Rz_initial.png

Prints the R(z) table. Should match the notebook to a few percent.
"""

import os, copy, logging, time
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from scipy.stats import norm as NormDist

import syntheticstellarpopconvolve
from syntheticstellarpopconvolve import (
    convolve, default_convolution_config, default_convolution_instruction,
)
from syntheticstellarpopconvolve.general_functions import generate_boilerplate_outputfile
from syntheticstellarpopconvolve.starformation_rate_distributions import (
    starformation_rate_distribution_vanSon2023,
)
from syntheticstellarpopconvolve.metallicity_distributions import (
    metallicity_distribution_vanSon2022,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_PATH  = r"C:\GROWL_tutorial3\output\bbh_with_eta.parquet"
OUT_FIG_DIR = r"C:\GROWL_tutorial3\output\figures"
SSPC_DIR    = r"C:\GROWL_tutorial3\output\sspc"
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(SSPC_DIR, "tmp"), exist_ok=True)
SSPC_FILE = os.path.join(SSPC_DIR, "vanSon22_BBH_cosmic_integration.h5")

Z_SUN = 0.0142
N_Z_BINS = 50   # MUST match Part 1

# ---------------------------------------------------------------------------
# Reload the bbh table + eta cached by Part 1
# ---------------------------------------------------------------------------
print("=" * 78)
print("Part 3 - cosmic integration with SSPC", syntheticstellarpopconvolve.__version__)
print("=" * 78)

bbh = pd.read_parquet(CACHE_PATH)
print(f"loaded cached bbh table: {len(bbh):,} rows, columns {list(bbh.columns)}")

# also need LOG_Z_EDGES - rebuild identically to Part 1
Z_all_min, Z_all_max = 1.0e-4, 3.0e-2      # simulated range printed by Part 1
LOG_Z_MIN, LOG_Z_MAX = np.log10(Z_all_min), np.log10(Z_all_max)
LOG_Z_EDGES  = np.linspace(LOG_Z_MIN, LOG_Z_MAX, N_Z_BINS + 1)
Z_EDGES_LIN  = 10 ** LOG_Z_EDGES

# ---------------------------------------------------------------------------
# Part 2 physics, redefined here: SFR density psi(z) and metallicity
# distribution dP/dlnZ(z).  Same functions the notebook uses.
# ---------------------------------------------------------------------------
def sfr_density(z, a=0.02, b=1.48, c=4.45, d=5.9):
    """Madau & Dickinson functional form; van Son+23 TNG100 params.
    Returns Msun/yr/Mpc^3 (comoving)."""
    return a * (1 + z) ** b / (1 + ((1 + z) / c) ** d)


def dP_dlnZ(lnZ, z, mu0=0.025, muz=-0.05, sigma0=1.125, sigmaz=0.05, alpha=-1.77):
    """Skewed log-normal metallicity distribution (van Son+23 §2).
    Returns dP/dlnZ at natural-log metallicities lnZ (1D) for redshifts z (1D)."""
    z = np.atleast_1d(z)[:, None]; lnZ = np.atleast_1d(lnZ)[None, :]
    sigma  = sigma0 * 10 ** (sigmaz * z)
    mean_Z = mu0 * 10 ** (muz * z)
    beta   = alpha / np.sqrt(1 + alpha**2)
    mu = np.log(mean_Z / 2 / (np.exp(0.5 * sigma**2) * NormDist.cdf(beta * sigma)))
    x  = (lnZ - mu) / sigma
    return 2 / sigma * NormDist.pdf(x) * NormDist.cdf(alpha * x)


# --- SFR grid for SSPC: 200 redshift bins from z=0 to z=10 ------------------
Z_SFR_EDGES  = np.linspace(0, 10, 201)
z_sfr_centres = 0.5 * (Z_SFR_EDGES[1:] + Z_SFR_EDGES[:-1])
psi_grid = sfr_density(z_sfr_centres) * 1e9 * u.Msun / u.yr / u.Gpc**3   # Mpc^-3 -> Gpc^-3

# --- metallicity distribution per (z, Z) bin, integrated on a fine subgrid --
N_SUB = 40
lnZ_fine = np.linspace(np.log(Z_EDGES_LIN[0]), np.log(Z_EDGES_LIN[-1]),
                       N_Z_BINS * N_SUB + 1)
P_fine = dP_dlnZ(0.5 * (lnZ_fine[1:] + lnZ_fine[:-1]), z_sfr_centres) * np.diff(lnZ_fine)
dP_per_bin = P_fine.reshape(len(z_sfr_centres), N_Z_BINS, N_SUB).sum(axis=2)
dPdZ_grid = dP_per_bin / np.diff(Z_EDGES_LIN)[None, :]

sfr_dict = {
    "redshift_bin_edges":             Z_SFR_EDGES,
    "starformation_rate_array":       psi_grid,
    "metallicity_bin_edges":          Z_EDGES_LIN,
    "metallicity_distribution_array": dPdZ_grid,
}

# cross-check against SSPC's built-in functions
psi_sspc    = starformation_rate_distribution_vanSon2023(z_sfr_centres).to(u.Msun/u.yr/u.Gpc**3)
lnZ_centres = 0.5 * (np.log(Z_EDGES_LIN[1:]) + np.log(Z_EDGES_LIN[:-1]))
dPdlnZ_sspc = metallicity_distribution_vanSon2022(log_metallicity_centers=lnZ_centres,
                                                  redshifts=z_sfr_centres)
dPdlnZ_ours = dP_dlnZ(lnZ_centres, z_sfr_centres)

print("\n--- S(Z,z) cross-check against SSPC's own implementations ---")
print(f"  psi(z):    max |ours/SSPC - 1| = {np.abs(psi_grid / psi_sspc - 1).max():.1e}")
print(f"  dP/dlnZ:   max |ours/SSPC - 1| = {np.abs(dPdlnZ_ours / dPdlnZ_sspc - 1).max():.1e}")

# --- plot the SFR and metallicity distribution, save as PNG ------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

z_grid = np.linspace(0, 10, 400)
axes[0].plot(z_grid, sfr_density(z_grid), color="#222222", lw=2.2, label="van Son+23 fit (TNG100)")
axes[0].plot(z_grid, sfr_density(z_grid, 0.015, 2.7, 2.9, 5.6), color="#999999", lw=1.6,
             ls="--", label="Madau & Dickinson 2014")
axes[0].set_yscale("log")
axes[0].set_xlabel("redshift $z$")
axes[0].set_ylabel(r"$\psi(z)$  [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]")
axes[0].set_title("Cosmic star formation history", loc="left")
axes[0].legend(loc="lower left")

lnZ_grid = np.linspace(np.log(1e-6), np.log(0.1), 600)
z_show = np.array([0, 0.5, 1, 2, 4, 6])
P = dP_dlnZ(lnZ_grid, z_show)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(z_show)))
for k, zz in enumerate(z_show):
    axes[1].plot(np.log10(np.exp(lnZ_grid) / Z_SUN), P[k], color=colors[k], lw=2, label=f"z = {zz:g}")
axes[1].axvspan(LOG_Z_MIN - np.log10(Z_SUN), LOG_Z_MAX - np.log10(Z_SUN),
                color="#2A9D8F", alpha=0.12, label="simulated Z range")
axes[1].set_xlabel(r"$\log_{10}(Z/Z_\odot)$")
axes[1].set_ylabel(r"$\mathrm{d}P/\mathrm{d}\ln Z$")
axes[1].set_title("Metallicity distribution of newly formed stars", loc="left")
axes[1].legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, "part3_SFR_and_Z.png"), dpi=130)
plt.close()

# what fraction of SF is inside the simulated Z range at key redshifts?
print("\n--- fraction of star formation inside the simulated Z range ---")
for k, zz in enumerate([0, 2, 6]):
    idx = np.argmin(np.abs(z_sfr_centres - zz))
    print(f"  z = {zz}: {dP_per_bin[idx].sum():.3f}")

# ---------------------------------------------------------------------------
# Write SSPC's input file and run the convolution
# ---------------------------------------------------------------------------
if os.path.exists(SSPC_FILE):
    os.remove(SSPC_FILE)
generate_boilerplate_outputfile(SSPC_FILE)

sspc_input = pd.DataFrame({
    "delay_time":  bbh.t_delay_Myr.values,
    "metallicity": bbh.Z.values,
    "eta":         bbh.eta.values,
})
sspc_input.to_hdf(SSPC_FILE, key="input_data/bbh")
print(f"\nwrote {len(sspc_input):,} rows to {SSPC_FILE}  ({os.path.getsize(SSPC_FILE)/1e6:.0f} MB)")

CONV_Z_EDGES = np.arange(0.0, 6.41, 0.4)   # centres: 0.2, 0.6, ..., 6.2

config = copy.copy(default_convolution_config)
config["logger"].setLevel(logging.ERROR)
config["output_filename"] = SSPC_FILE
config["tmp_dir"] = os.path.join(SSPC_DIR, "tmp")
config["redshift_interpolator_data_output_filename"] = os.path.join(SSPC_DIR, "tmp", "z_interpolator.p")
config["time_type"] = "redshift"
config["cosmology"] = cosmo
config["multiprocessing"] = False
config["convolution_redshift_bin_edges"] = CONV_Z_EDGES
config["SFR_info"] = sfr_dict
config["convolution_instructions"] = [{
    **default_convolution_instruction,
    "convolution_type": "integrate",
    "input_data_name": "bbh",
    "output_data_name": "rates",
    "data_column_dict": {
        "normalized_yield": "eta",
        "delay_time": {"column_name": "delay_time", "unit": u.Myr},
        "metallicity": "metallicity",
    },
}]

print("\nRunning SSPC convolution (this may take a minute the first time)...")
t0 = time.time()
convolve(config=config)
print(f"done in {time.time() - t0:.0f} s; output file is now {os.path.getsize(SSPC_FILE)/1e6:.0f} MB")

# ---------------------------------------------------------------------------
# Read the output and print the rate table
# ---------------------------------------------------------------------------
def read_sspc_yield(z_centre):
    with h5py.File(SSPC_FILE, "r") as f:
        return f[f"output_data/bbh/rates/convolution_results/{np.round(z_centre, 4)}/yield"][()]

CONV_Z  = 0.5 * (CONV_Z_EDGES[1:] + CONV_Z_EDGES[:-1])
R_sspc  = np.array([read_sspc_yield(z).sum() for z in CONV_Z])

print("\n--- R_BBH(z) from SSPC ---")
print(f"{'z':>5s}  {'R [Gpc^-3 yr^-1]':>20s}  {'notebook':>10s}")
# notebook values, for a line-by-line check
_nb = {0.2: 67.5, 0.6: 104.6, 1.0: 147.8, 1.4: 194.1, 1.8: 237.5,
       2.2: 269.8, 2.6: 284.5, 3.0: 279.5, 3.4: 258.3, 3.8: 227.6,
       4.2: 194.1, 4.6: 162.0, 5.0: 133.7, 5.4: 109.5, 5.8: 89.4, 6.2: 72.7}
for z, r in zip(CONV_Z, R_sspc):
    nb = _nb.get(round(z, 1), float("nan"))
    pct = 100 * (r / nb - 1) if nb == nb else float("nan")
    print(f"{z:5.1f}  {r:20.1f}  {nb:10.1f}   ({pct:+.1f}%)")

# quick figure: R(z) alone
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(CONV_Z, R_sspc, "o-", color="#222222", lw=2, ms=6)
ax.set_yscale("log")
ax.set_xlabel("merger redshift $z$")
ax.set_ylabel(r"$\mathcal{R}_{\rm BBH}(z)$  [Gpc$^{-3}$ yr$^{-1}$]")
ax.set_title("Binary black hole merger rate density - first SSPC run", loc="left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, "part3_Rz_initial.png"), dpi=130)
plt.close()

# save the rate table for Part 4
out_csv = r"C:\GROWL_tutorial3\output\R_BBH_vs_z.csv"
np.savetxt(out_csv, np.column_stack([CONV_Z, R_sspc]), delimiter=",",
           header="z,R_BBH_Gpc3_yr", comments="")
print(f"\nsaved R(z) table to {out_csv}")

print("\n" + "=" * 78)
print("Part 3 done.")
print("=" * 78)