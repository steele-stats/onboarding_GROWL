"""
GROWL Tutorial 3 - Part 1: the formation efficiency.

Ports Part 1 of tutorial_3_cosmic_integration_predicting_merger_rates.ipynb
to a plain Python script. Run from PowerShell:

    cd C:\GROWL_tutorial3\scripts
    python part1_formation_efficiency.py

Produces (in C:\GROWL_tutorial3\output\figures\):
    part1_imf.png           - IMF by number and by mass, simulated range shaded
    part1_eta_vs_Z.png      - formation efficiency eta_BBH(Z); right panel shows the trap

Also prints diagnostic numbers that should match the notebook. If any
"CHECK" line says MISMATCH, stop and tell Claude - do not continue.
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive; we save PNGs instead of popping up windows
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ---------------------------------------------------------------------------
# Paths - edit these if your folders ever move
# ---------------------------------------------------------------------------
DATA_PATH   = r"C:\GROWL_data\COMPAS_Output_wWeights.h5"
OUT_FIG_DIR = r"C:\GROWL_tutorial3\output\figures"
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants from the notebook
# ---------------------------------------------------------------------------
Z_SUN   = 0.0142          # solar metallicity (Asplund et al. 2009)
BH, NS  = 14, 13          # Hurley stellar type codes

# Columns we want from the DCO table (same list as the notebook)
DCO_COLUMNS = ["SEED", "Mass(1)", "Mass(2)", "Stellar_Type(1)", "Stellar_Type(2)",
               "Merges_Hubble_Time", "Coalescence_Time", "Time", "Metallicity@ZAMS(1)",
               "CE_Event_Counter", "mixture_weight"]

# ---------------------------------------------------------------------------
# Step 0: peek inside the HDF5 file so we see the real column names up front.
#         If any column we want is missing, we fail here with a clear message
#         instead of crashing halfway through.
# ---------------------------------------------------------------------------
print("=" * 78)
print("Opening", DATA_PATH)
print("=" * 78)

with h5py.File(DATA_PATH, "r") as f:
    print("Top-level groups:", list(f.keys()))

    # what's inside BSE_Double_Compact_Objects?
    dco_group = f["BSE_Double_Compact_Objects"]
    dco_cols_available = sorted(dco_group.keys())
    print(f"\nBSE_Double_Compact_Objects has {len(dco_cols_available)} columns. First 40:")
    for c in dco_cols_available[:40]:
        print("   ", c)

    missing = [c for c in DCO_COLUMNS if c not in dco_cols_available]
    if missing:
        print("\n!! MISSING expected columns:", missing)
        print("!! Available columns above; tell Claude the names so we can adjust.")
        raise SystemExit(1)
    print("\nAll expected DCO columns present.")

    sys_group = f["BSE_System_Parameters"]
    if "Metallicity@ZAMS(1)" not in sys_group:
        print("!! BSE_System_Parameters has no 'Metallicity@ZAMS(1)'. Columns:",
              sorted(sys_group.keys())[:40])
        raise SystemExit(1)
    print("BSE_System_Parameters has 'Metallicity@ZAMS(1)'.")

# ---------------------------------------------------------------------------
# Step 1: load the DCO table and the metallicity of every simulated binary
# ---------------------------------------------------------------------------
print("\nLoading DCO table and metallicities (this may take ~30 s)...")

with h5py.File(DATA_PATH, "r") as f:
    dco = pd.DataFrame({c: f["BSE_Double_Compact_Objects"][c][()] for c in DCO_COLUMNS})
    Z_all_systems = f["BSE_System_Parameters"]["Metallicity@ZAMS(1)"][()]

n_binaries_simulated = len(Z_all_systems)

dco = dco.rename(columns={
    "Mass(1)": "M1_raw", "Mass(2)": "M2_raw",
    "Stellar_Type(1)": "type1", "Stellar_Type(2)": "type2",
    "Metallicity@ZAMS(1)": "Z",
    "Merges_Hubble_Time": "merges",
    "Coalescence_Time": "t_coal_Myr",
    "Time": "t_form_Myr",
    "CE_Event_Counter": "n_CE",
    "mixture_weight": "w",
})

dco["is_BBH"]  = (dco.type1 == BH) & (dco.type2 == BH)
dco["is_BHNS"] = (dco.type1 == BH) ^ (dco.type2 == BH)
dco["is_BNS"]  = (dco.type1 == NS) & (dco.type2 == NS)
dco["m1"] = np.maximum(dco.M1_raw, dco.M2_raw)
dco["m2"] = np.minimum(dco.M1_raw, dco.M2_raw)
dco["t_delay_Myr"] = dco.t_form_Myr + dco.t_coal_Myr

bbh = dco[dco.is_BBH & (dco.merges == 1)].copy()

print(f"\n  binaries simulated                 : {n_binaries_simulated:,}")
print(f"  double compact objects             : {len(dco):,}")
print(f"  merging binary black holes (rows)  : {len(bbh):,}")
print(f"  merging binary black holes (sum w) : {bbh.w.sum():,.0f}")
print(f"  metallicity range simulated        : {Z_all_systems.min():.1e} - {Z_all_systems.max():.2e}"
      f"   (log10 Z/Zsun from {np.log10(Z_all_systems.min()/Z_SUN):.2f} to {np.log10(Z_all_systems.max()/Z_SUN):.2f})")

# notebook-published values, for a self-check
_expect = {"n_sim": 10_000_000, "n_dco": 2_523_122,
           "n_bbh_rows": 1_640_550, "bbh_w_sum": 14_911}
print("\n  self-checks (notebook vs ours):")
for k, v in _expect.items():
    ours = {"n_sim": n_binaries_simulated, "n_dco": len(dco),
            "n_bbh_rows": len(bbh), "bbh_w_sum": int(bbh.w.sum())}[k]
    tag = "OK" if abs(ours - v) <= max(2, 0.02 * v) else "MISMATCH"
    print(f"    {k:12s}: expect {v:>12,}   got {ours:>12,}   [{tag}]")

# ---------------------------------------------------------------------------
# Step 2: Kroupa (2001) IMF and the two IMF figures
# ---------------------------------------------------------------------------
IMF_BOUNDS = (0.01, 0.08, 0.5, 200.0)
IMF_SLOPES = (0.3, 1.3, 2.3)
M1_MIN, M1_MAX, M2_MIN, F_BIN = 5.0, 150.0, 0.1, 0.7

def kroupa_imf(m):
    """dN/dm (unnormalised), continuous across the breaks."""
    m = np.asarray(m, dtype=float)
    m0, m1, m2, m3 = IMF_BOUNDS
    a1, a2, a3 = IMF_SLOPES
    c2 = m1 ** (a2 - a1)
    c3 = c2 * m2 ** (a3 - a2)
    out = np.where(m < m1, m ** -a1,
          np.where(m < m2, c2 * m ** -a2, c3 * m ** -a3))
    return np.where((m >= m0) & (m <= m3), out, 0.0)

_norm = quad(kroupa_imf, IMF_BOUNDS[0], IMF_BOUNDS[-1],
             points=IMF_BOUNDS[1:-1])[0]
imf_pdf = lambda m: kroupa_imf(m) / _norm

m_grid = np.logspace(np.log10(IMF_BOUNDS[0]), np.log10(IMF_BOUNDS[-1]), 500)

# colours matching the notebook
CSYS, C_SMT, C_WRONG = "#222222", "#2A9D8F", "#999999"

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].plot(m_grid, m_grid * imf_pdf(m_grid), color=CSYS, lw=2)
axes[0].axvspan(M1_MIN, M1_MAX, color=C_SMT, alpha=0.2,
                label=f"simulated primaries: {M1_MIN:.0f}-{M1_MAX:.0f} M$_\odot$")
axes[0].set_ylabel(r"number of stars per dex, $m\,\mathrm{d}N/\mathrm{d}m$")
axes[0].set_title("By number: almost every star is light", loc="left")
axes[1].plot(m_grid, m_grid**2 * imf_pdf(m_grid), color=CSYS, lw=2)
axes[1].axvspan(M1_MIN, M1_MAX, color=C_SMT, alpha=0.2, label="simulated primaries")
axes[1].set_ylabel(r"stellar mass per dex, $m^2\,\mathrm{d}N/\mathrm{d}m$")
axes[1].set_title("By mass: most stellar mass is ALSO in light stars", loc="left")
for ax in axes:
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"birth mass $m$ [M$_\odot$]")
    ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, "part1_imf.png"), dpi=130)
plt.close()

frac_number = quad(imf_pdf, M1_MIN, M1_MAX)[0]
mass_pdf = lambda m: m * imf_pdf(m)
m_avg = quad(mass_pdf, IMF_BOUNDS[0], IMF_BOUNDS[-1],
             points=IMF_BOUNDS[1:-1])[0]
frac_mass = quad(mass_pdf, M1_MIN, M1_MAX)[0] / m_avg

print("\n--- IMF numbers ---")
print(f"  average mass of an IMF star          : {m_avg:.3f} Msun")
print(f"  fraction by number in 5-150 Msun     : {100*frac_number:.2f}%   (1 in {1/frac_number:.0f})")
print(f"  fraction of stellar MASS in 5-150    : {100*frac_mass:.1f}%")

# ---------------------------------------------------------------------------
# Step 3: star-forming mass represented by ONE simulated binary  (M_rep)
# ---------------------------------------------------------------------------
mass_per_binary_drawn = m_avg * (1.5 + (1 - F_BIN) / F_BIN)
f_sim = quad(lambda m: imf_pdf(m) * (1 - M2_MIN / m), M1_MIN, M1_MAX)[0]
M_REP = mass_per_binary_drawn / f_sim
M_SIM = n_binaries_simulated * M_REP

print("\n--- formation efficiency bookkeeping ---")
print(f"  mass per binary drawn, incl. companion and singles : {mass_per_binary_drawn:.3f} Msun")
print(f"  f_sim  (fraction inside the simulated box)          : {f_sim:.5f}   (1 in {1/f_sim:.0f})")
print(f"  M_rep  (star-forming mass per simulated binary)     : {M_REP:.1f} Msun")
print(f"  M_SIM  (total star-forming mass represented)        : {M_SIM:.3e} Msun")

# cross-check against the analytic COMPAS formula
def analytical_star_forming_mass_per_binary_using_kroupa_imf(
        m1_min, m1_max, m2_min, fbin=1.0,
        imf_mass_bounds=(0.01, 0.08, 0.5, 200)):
    m1, m2, m3, m4 = imf_mass_bounds
    alpha = (-(m4**(-1.3) - m3**(-1.3))/1.3
             - (m3**(-0.3) - m2**(-0.3))/(m3*0.3)
             + (m2**0.7 - m1**0.7)/(m2*m3*0.7))**(-1)
    m_avg_a = alpha * (-(m4**(-0.3) - m3**(-0.3))/0.3
                       + (m3**0.7 - m2**0.7)/(m3*0.7)
                       + (m2**1.7 - m1**1.7)/(m2*m3*1.7))
    fint = (-alpha/1.3 * (m1_max**(-1.3) - m1_min**(-1.3))
            + alpha * m2_min/2.3 * (m1_max**(-2.3) - m1_min**(-2.3)))
    return (1/fint) * m_avg_a * (1.5 + (1 - fbin)/fbin)

M_REP_analytic = analytical_star_forming_mass_per_binary_using_kroupa_imf(
    M1_MIN, M1_MAX, M2_MIN, F_BIN)
print(f"  COMPAS analytic formula                             : {M_REP_analytic:.1f} Msun"
      f"  (diff {100*abs(M_REP/M_REP_analytic-1):.3f}%)")

# ---------------------------------------------------------------------------
# Step 4: per-kind formation efficiency
# ---------------------------------------------------------------------------
print("\n--- formation efficiency per DCO kind ---")
print(f"{'kind':38s} {'sum of weights':>15s} {'eta [1/Msun]':>15s} {'1 per ... Msun':>16s}")
print("-" * 90)
for name, mask in [("binary black holes, all",                dco.is_BBH),
                   ("binary black holes, MERGING",            dco.is_BBH & (dco.merges == 1)),
                   ("black hole - neutron star, merging",     dco.is_BHNS & (dco.merges == 1)),
                   ("binary neutron stars, merging",          dco.is_BNS & (dco.merges == 1))]:
    sw = dco.w[mask].sum()
    print(f"{name:38s} {sw:15,.0f} {sw/M_SIM:15.2e} {M_SIM/sw:16,.0f}")

ETA_BBH_MEAN = bbh.w.sum() / M_SIM
print(f"\n  ETA_BBH_MEAN : {ETA_BBH_MEAN:.2e} per Msun  (one per {1/ETA_BBH_MEAN:,.0f} Msun)")
print(f"  notebook says: 1.47e-05 per Msun (one per ~68,000 Msun)")

# ---------------------------------------------------------------------------
# Step 5: eta(Z), the right way and the wrong way
# ---------------------------------------------------------------------------
LOG_Z_MIN, LOG_Z_MAX = np.log10(Z_all_systems.min()), np.log10(Z_all_systems.max())
DLOGZ_TOT = LOG_Z_MAX - LOG_Z_MIN

def mass_formed_per_bin(log_z_edges):
    return M_SIM * np.diff(log_z_edges) / DLOGZ_TOT

def formation_efficiency_vs_Z(sample, log_z_edges):
    sum_w, _ = np.histogram(np.log10(sample.Z), bins=log_z_edges, weights=sample.w)
    return sum_w / mass_formed_per_bin(log_z_edges)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

for nbins, ls in [(15, "-"), (60, ":")]:
    edges = np.linspace(LOG_Z_MIN, LOG_Z_MAX, nbins + 1)
    axes[0].stairs(formation_efficiency_vs_Z(bbh, edges),
                   edges - np.log10(Z_SUN), color=CSYS, lw=2, ls=ls,
                   label=f"correct: divide by mass in bin ({nbins} bins)")
axes[0].set_title("Right: independent of binning", loc="left")

for nbins, ls in [(15, "-"), (60, ":")]:
    edges = np.linspace(LOG_Z_MIN, LOG_Z_MAX, nbins + 1)
    sum_w, _ = np.histogram(np.log10(bbh.Z), bins=edges, weights=bbh.w)
    axes[1].stairs(sum_w / M_SIM, edges - np.log10(Z_SUN),
                   color=C_WRONG, lw=2, ls=ls,
                   label=f"WRONG: divide by all of M_sim ({nbins} bins)")
edges = np.linspace(LOG_Z_MIN, LOG_Z_MAX, 16)
axes[1].stairs(formation_efficiency_vs_Z(bbh, edges),
               edges - np.log10(Z_SUN), color=CSYS, lw=1, alpha=0.5,
               label="correct, for reference")
axes[1].set_title("Wrong: changes with the number of bins", loc="left")

for ax in axes:
    ax.set_yscale("log"); ax.set_ylim(3e-9, 1e-4)
    ax.set_xlabel(r"birth metallicity $\log_{10}(Z/Z_\odot)$")
    ax.legend(fontsize=9, loc="lower left")
axes[0].set_ylabel(r"formation efficiency $\eta_{\rm BBH}(Z)$  [per M$_\odot$]")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, "part1_eta_vs_Z.png"), dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# Step 6: per-binary eta column, ready for SSPC in Part 3
# ---------------------------------------------------------------------------
N_Z_BINS = 50
LOG_Z_EDGES = np.linspace(LOG_Z_MIN, LOG_Z_MAX, N_Z_BINS + 1)
mass_per_bin = mass_formed_per_bin(LOG_Z_EDGES)

bbh["z_bin"] = np.clip(np.digitize(np.log10(bbh.Z), LOG_Z_EDGES) - 1,
                       0, N_Z_BINS - 1)
bbh["eta"]   = bbh.w / mass_per_bin[bbh.z_bin.values]

frac_of_range = np.diff(LOG_Z_EDGES) / DLOGZ_TOT
eta_mean_from_rows = np.sum(
    bbh.groupby("z_bin").eta.sum().reindex(range(N_Z_BINS), fill_value=0).values
    * frac_of_range)

print("\n--- per-binary eta (consistency check) ---")
print(f"  mean eta from per-row values  : {eta_mean_from_rows:.4e} per Msun")
print(f"  mean eta from sum(w)/M_sim    : {ETA_BBH_MEAN:.4e} per Msun")
print(f"  per-binary eta range          : {bbh.eta.min():.1e} to {bbh.eta.max():.1e} per Msun"
      f"  (median {bbh.eta.median():.1e})")

# save the bbh table with eta for Part 3 to reuse without recomputing
BBH_CACHE = r"C:\GROWL_tutorial3\output\bbh_with_eta.parquet"
try:
    bbh[["m1", "m2", "Z", "t_delay_Myr", "w", "eta", "n_CE", "z_bin"]].to_parquet(
        BBH_CACHE, index=False)
    print(f"\n  cached bbh table with eta  -> {BBH_CACHE}")
except Exception as e:
    print(f"\n  (could not cache to parquet: {e}; Part 3 will recompute)")

print("\n" + "=" * 78)
print("Part 1 done. Figures saved to:", OUT_FIG_DIR)
print("=" * 78)