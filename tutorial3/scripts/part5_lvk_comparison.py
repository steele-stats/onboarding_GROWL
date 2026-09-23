"""
GROWL Tutorial 3 - Part 5: comparison with LVK observations (GWTC-5.0).

Reads:
    C:\\GROWL_data\\gwtc5_bbh_rate_and_mass.h5
    C:\\GROWL_tutorial3\\output\\bbh_with_eta.parquet
    C:\\GROWL_tutorial3\\output\\sspc\\vanSon22_BBH_cosmic_integration.h5
    C:\\GROWL_data\\COMPAS_Output_wWeights.h5       (answer key only, for overlay)

Writes:
    C:\\GROWL_tutorial3\\output\\figures\\part5_Rz_sim_vs_LVK.png
    C:\\GROWL_tutorial3\\output\\figures\\part5_dRdm1_sim_vs_LVK.png
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LVK_FILE    = r"C:\GROWL_data\gwtc5_bbh_rate_and_mass.h5"
CACHE_PATH  = r"C:\GROWL_tutorial3\output\bbh_with_eta.parquet"
SSPC_FILE   = r"C:\GROWL_tutorial3\output\sspc\vanSon22_BBH_cosmic_integration.h5"
DATA_PATH   = r"C:\GROWL_data\COMPAS_Output_wWeights.h5"
OUT_FIG_DIR = r"C:\GROWL_tutorial3\output\figures"

# colours matching the notebook
CSYS, C_CE, C_SMT, C_OBS, C_WRONG = "#222222", "#E07B39", "#2A9D8F", "#7B2CBF", "#999999"
C_LVK, C_PP = C_OBS, "#2C5AA0"

CONV_Z_EDGES = np.arange(0.0, 6.41, 0.4)
CONV_Z       = 0.5 * (CONV_Z_EDGES[1:] + CONV_Z_EDGES[:-1])
RATES_GROUP  = ("Rates_mu00.025_muz-0.05_alpha-1.77_"
                "sigma01.125_sigmaz0.05_a0.02_b1.48_c4.45_d5.9_zBinned")

print("=" * 78)
print("Part 5 - comparison with LVK GWTC-5.0")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1. Reload our simulation data
# ---------------------------------------------------------------------------
bbh = pd.read_parquet(CACHE_PATH)

def read_sspc_yield(z_centre):
    with h5py.File(SSPC_FILE, "r") as f:
        return f[f"output_data/bbh/rates/convolution_results/{np.round(z_centre, 4)}/yield"][()]

R_sspc     = np.array([read_sspc_yield(z).sum() for z in CONV_Z])
yield_z02  = read_sspc_yield(0.2)
is_smt     = (bbh.n_CE == 0).values
R_sspc_ce  = np.array([read_sspc_yield(z)[~is_smt].sum() for z in CONV_Z])
R_sspc_smt = np.array([read_sspc_yield(z)[is_smt].sum()  for z in CONV_Z])

# also pull the authors' answer key (only needed for the faint grey overlay)
with h5py.File(DATA_PATH, "r") as f:
    g = f[RATES_GROUP]
    z_edges_authors = g["redshifts"][()]
    z_authors       = 0.5 * (z_edges_authors[1:] + z_edges_authors[:-1])
    dco_mask        = g["DCOmask"][()]
    with h5py.File(DATA_PATH, "r") as ff:
        type1  = ff["BSE_Double_Compact_Objects"]["Stellar_Type(1)"][()]
        type2  = ff["BSE_Double_Compact_Objects"]["Stellar_Type(2)"][()]
        merges = ff["BSE_Double_Compact_Objects"]["Merges_Hubble_Time"][()]
    is_our_bbh = ((type1 == 14) & (type2 == 14) & (merges == 1))[dco_mask]
    R_authors = np.zeros(len(z_authors))
    chunk = 100_000
    for i0 in range(0, g["merger_rate"].shape[0], chunk):
        block = g["merger_rate"][i0:i0 + chunk, :]
        sel   = is_our_bbh[i0:i0 + chunk]
        R_authors += block[sel].sum(axis=0)

# ---------------------------------------------------------------------------
# 2. Load the LVK file
# ---------------------------------------------------------------------------
lvk = {}
with h5py.File(LVK_FILE, "r") as f:
    print("LVK source:", f.attrs["source"])
    Z_PE_1ST, Z_PE_99TH = f.attrs["pe_redshift_1st_99th_percentile"]
    M1_PE_1ST, M1_PE_99TH = f.attrs["pe_mass_1_source_1st_99th_percentile"]
    for model in ["default_bbh", "pixelpop"]:
        for key in ["rate_vs_redshift", "dR_dm1_z0.2"]:
            g = f[f"{model}/{key}"]
            lvk[(model, key)] = {
                "x":    g["positions"][()],
                "q05":  g["quantiles"][0],
                "q50":  g["quantiles"][1],
                "q95":  g["quantiles"][2],
                "unit": g.attrs["unit"],
                "n":    g.attrs["n_draws_total"],
            }
    lvk_lamb = f["default_bbh/lamb"][()]

print(f"LVK data: events constrain z in [{Z_PE_1ST:.2f}, {Z_PE_99TH:.2f}] "
      f"and m1 in [{M1_PE_1ST:.1f}, {M1_PE_99TH:.0f}] Msun")
print(f"LVK kappa_z: median {np.median(lvk_lamb):.2f}, "
      f"90% [{np.quantile(lvk_lamb, 0.05):.2f}, {np.quantile(lvk_lamb, 0.95):.2f}]")

i02 = int(np.argmin(np.abs(lvk[("default_bbh", "rate_vs_redshift")]["x"] - 0.2)))
d   = lvk[("default_bbh", "rate_vs_redshift")]
lvk_R02       = d["q50"][i02]
lvk_R02_low   = d["q05"][i02]
lvk_R02_high  = d["q95"][i02]
print(f"\nLVK R(z=0.2) = {lvk_R02:.1f} (90% {lvk_R02_low:.1f} - {lvk_R02_high:.1f}) Gpc^-3 yr^-1")
print(f"our R(z=0.2) = {R_sspc[0]:.1f}   ->  simulation / observation = {R_sspc[0]/lvk_R02:.2f}")

# ---------------------------------------------------------------------------
# 3. Figure: R(z) simulation vs LVK
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.4))

# LVK bands
d = lvk[("pixelpop", "rate_vs_redshift")]
ax.fill_between(d["x"], d["q05"], d["q95"], step="pre", color=C_PP, alpha=0.18, lw=0)
ax.step(d["x"], d["q50"], color=C_PP, lw=1.6, label="LVK GWTC-5.0, PixelPop (median, 90% band)")
d = lvk[("default_bbh", "rate_vs_redshift")]
ax.fill_between(d["x"], d["q05"], d["q95"], color=C_LVK, alpha=0.3, lw=0)
ax.plot(d["x"], d["q50"], color=C_LVK, lw=2, label="LVK GWTC-5.0, Default BBH (median, 90% band)")

# our simulation
ax.plot(z_authors, R_authors, color=C_WRONG, lw=6, alpha=0.5,
        label="van Son et al. fiducial model (authors)")
ax.plot(CONV_Z, R_sspc, "o-", color=CSYS, lw=3, ms=6,
        label="van Son et al. fiducial model (this notebook, SSPC)")
ax.plot(CONV_Z, R_sspc_ce,  "s--", color=C_CE,  lw=2, ms=4, label="  common-envelope channel")
ax.plot(CONV_Z, R_sspc_smt, "^--", color=C_SMT, lw=2, ms=4, label="  stable-mass-transfer channel")

# shaded "no events beyond here"
ax.axvspan(Z_PE_99TH, 2, facecolor="none", edgecolor="grey", hatch="////", alpha=0.4, lw=0)
ax.axvline(Z_PE_99TH, color="grey", ls="-.", lw=1)
ax.text(Z_PE_99TH + 0.02, 3.5, "no events beyond here", rotation=90,
        va="bottom", fontsize=9, color="grey")

ax.set_yscale("log"); ax.set_xlim(0, 1.5); ax.set_ylim(3, 400)
ax.set_xlabel("merger redshift $z$")
ax.set_ylabel(r"$\mathcal{R}_{\rm BBH}(z)$  [Gpc$^{-3}$ yr$^{-1}$]")
ax.set_title("Binary black hole merger rate: one simulation vs. GWTC-5.0", loc="left")
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, "part5_Rz_sim_vs_LVK.png"), dpi=130)
plt.close()

# shape slope: R ~ (1+z)^kappa for z < 1
sel = CONV_Z <= 1.0
kappa_sim = np.polyfit(np.log1p(CONV_Z[sel]), np.log(R_sspc[sel]), 1)[0]
print(f"\nshape slope kappa (z < 1): simulation {kappa_sim:.2f},  "
      f"LVK median {np.median(lvk_lamb):.2f}")

# ---------------------------------------------------------------------------
# 4. Figure: dR/dm1 at z = 0.2 vs LVK
# ---------------------------------------------------------------------------
m1_bins_log = np.logspace(np.log10(2.5), np.log10(120), 45)

h_sim, _ = np.histogram(bbh.m1, bins=m1_bins_log, weights=yield_z02)
h_ce,  _ = np.histogram(bbh.m1[~is_smt], bins=m1_bins_log, weights=yield_z02[~is_smt])
h_smt, _ = np.histogram(bbh.m1[is_smt],  bins=m1_bins_log, weights=yield_z02[is_smt])

fig, ax = plt.subplots(figsize=(9.5, 5.6))

d = lvk[("pixelpop", "dR_dm1_z0.2")]
ax.fill_between(d["x"], d["q05"], d["q95"], step="pre", color=C_PP, alpha=0.18, lw=0)
ax.step(d["x"], d["q50"], color=C_PP, lw=1.6, label="LVK GWTC-5.0, PixelPop (median, 90% band)")
d = lvk[("default_bbh", "dR_dm1_z0.2")]
ax.fill_between(d["x"], d["q05"], d["q95"], color=C_LVK, alpha=0.3, lw=0)
ax.plot(d["x"], d["q50"], color=C_LVK, lw=2, label="LVK GWTC-5.0, Default BBH (median, 90% band)")

ax.stairs(h_sim / np.diff(m1_bins_log), m1_bins_log, color=CSYS, lw=4,
          label="van Son et al. fiducial model (this notebook, SSPC)")
ax.stairs(h_ce  / np.diff(m1_bins_log), m1_bins_log, color=C_CE,  lw=2, ls="--",
          label="  common-envelope channel")
ax.stairs(h_smt / np.diff(m1_bins_log), m1_bins_log, color=C_SMT, lw=2, ls="--",
          label="  stable-mass-transfer channel")

for lo, hi in [(2, M1_PE_1ST), (M1_PE_99TH, 200)]:
    ax.axvspan(lo, hi, facecolor="none", edgecolor="grey", hatch="////", alpha=0.4, lw=0)
ax.axvline(M1_PE_1ST,  color="grey", ls="-.", lw=1)
ax.axvline(M1_PE_99TH, color="grey", ls="-.", lw=1)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(2.5, 150); ax.set_ylim(1e-3, 30)
ax.set_xticks([3, 5, 10, 20, 35, 50, 100])
ax.set_xticklabels(["3", "5", "10", "20", "35", "50", "100"])
ax.set_xlabel(r"$m_1$, mass of the more massive black hole [M$_\odot$]")
ax.set_ylabel(r"$\mathrm{d}\mathcal{R}/\mathrm{d}m_1$ at $z = 0.2$  [Gpc$^{-3}$ yr$^{-1}$ M$_\odot^{-1}$]")
ax.set_title("Primary-mass distribution: one simulation vs. GWTC-5.0", loc="left")
ax.legend(loc="upper right", fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, "part5_dRdm1_sim_vs_LVK.png"), dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# 5. Mass-range rate table
# ---------------------------------------------------------------------------
d = lvk[("default_bbh", "dR_dm1_z0.2")]
print("\n--- rate in mass ranges, z = 0.2 ---")
print(f"{'m1 range [Msun]':>18s} {'simulation':>12s} {'LVK median':>12s} {'ratio':>8s}   [Gpc^-3 yr^-1]")
for lo, hi in [(5, 10), (10, 20), (20, 35), (35, 50), (50, 100)]:
    r_sim = yield_z02[(bbh.m1.values >= lo) & (bbh.m1.values < hi)].sum()
    mask  = (d["x"] >= lo) & (d["x"] < hi)
    r_lvk = np.trapz(d["q50"][mask], d["x"][mask])
    ratio = r_sim / r_lvk if r_lvk > 0 else float("nan")
    print(f"{lo:8.0f} - {hi:<7.0f} {r_sim:12.2f} {r_lvk:12.2f} {ratio:8.2f}")

print("\n" + "=" * 78)
print("Part 5 done.")
print("=" * 78)