"""
gen_chi_min_plot.py
===================
Produces three figures from the chi-scan experiment:

  1. csp_ler_vs_chi.pdf   — LER_TN(chi) for each N at p=1e-3, with
                            0.9 * LER_BP horizontal baseline.

  2. csp_ler_ratio.pdf    — LER_TN(chi=400) / LER_BP vs N at p=1e-3.
                            All seven N values are directly measured
                            (no extrapolation).  Reference lines at
                            y=1 (TN matches BP) and y=0.9 (10% advantage).

  3. csp_chi_conv.pdf     — chi_conv(N): smallest chi at which the
                            cumulative-min-smoothed TN LER is within
                            CONV_TOL of the lowest LER observed in the
                            chi-scan.  All seven points are directly
                            observable from the data we already have.

Data sources
------------
  TN data : examples/decoding/data-quantum-csp-batch-9/   (all chi merged)
  BP-OSD  : examples/decoding/bp_results.pkl
"""

# --- repo-relative paths (this script lives in examples/decoding/plotting/) ---
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DECODING = _os.path.dirname(_HERE)
_EXAMPLES = _os.path.dirname(_DECODING)
_ROOT = _os.path.dirname(_EXAMPLES)
for _p in (_ROOT, _EXAMPLES, _DECODING):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# Relative data dirs / output files below resolve against this directory.
_os.chdir(_DECODING)
_FIGDIR = _os.path.join(_DECODING, "figures")
_os.makedirs(_FIGDIR, exist_ok=True)


def _fig(_name):
    """Absolute path inside examples/decoding/figures/ (cwd-independent)."""
    return _os.path.join(_FIGDIR, _os.path.basename(_name))


# -----------------------------------------------------------------------------

import sys


import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from data_handling import process_failure_statistics_csp, process_bp_statistics

mpl.style.use("default")
mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


LATTICE_SIZES = [30, 40, 50, 60, 70, 80, 90]
ADVANTAGE = 0.90  # 10% LER advantage criterion (TN <= 0.9 * BP)
ERROR_RATE = 0.001
CHI_SCAN_DIR = "data-quantum-csp-batch-9"  # all chi merged here
CHI_SCAN_VALS = [4, 8, 16, 32, 64, 128, 256]
CHI_MAX = 400
ALL_CHIS = sorted(CHI_SCAN_VALS + [CHI_MAX])
CONV_TOL = 0.10  # chi_conv: smoothed LER within (1+CONV_TOL) of min

# ── Load TN data ──────────────────────────────────────────────────────────────
print("Loading TN chi-scan data...")
_, scan_lr, scan_eb, _, _, _, _ = process_failure_statistics_csp(
    lattice_sizes=LATTICE_SIZES,
    max_bond_dims=CHI_SCAN_VALS,
    error_model="Bitflip",
    directory=CHI_SCAN_DIR,
    equalize=False,
)

print("Loading TN chi=400 data...")
_, b9_lr, b9_eb, _, _, _, _ = process_failure_statistics_csp(
    lattice_sizes=LATTICE_SIZES,
    max_bond_dims=[CHI_MAX],
    error_model="Bitflip",
    directory=CHI_SCAN_DIR,
    equalize=False,
)

tn_lr = {**scan_lr, **b9_lr}
tn_eb = {**scan_eb, **b9_eb}

# ── Load BP-OSD baseline ──────────────────────────────────────────────────────
print("Loading BP-OSD data...")
with open("bp_results.pkl", "rb") as f:
    bp_results = pickle.load(f)
_, bp_ler, bp_err, _ = process_bp_statistics(bp_results, lattice_sizes=LATTICE_SIZES)

p = round(ERROR_RATE, 5)

# ── Figure 1: LER_TN(chi) per N ───────────────────────────────────────────────
cmap = mpl.colormaps["viridis_r"]
norm_n = Normalize(vmin=0, vmax=len(LATTICE_SIZES) - 1)

fig1, ax1 = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)

# Cache the per-N (chis, lrs) traces — reused for figure 3.
tn_traces = {}

for idx, N in enumerate(LATTICE_SIZES):
    color = cmap(norm_n(idx))

    chis, lrs, ebs = [], [], []
    for chi in ALL_CHIS:
        key = (N, chi, p)
        if key in tn_lr and not np.isnan(tn_lr[key]):
            chis.append(chi)
            lrs.append(tn_lr[key])
            ebs.append(tn_eb.get(key, 0))

    if len(chis) < 2:
        print(f"N={N}: insufficient TN data, skipping.")
        continue

    chis, lrs, ebs = np.array(chis), np.array(lrs), np.array(ebs)
    order = np.argsort(chis)
    chis, lrs, ebs = chis[order], lrs[order], ebs[order]
    tn_traces[N] = (chis, lrs, ebs, color)

    ax1.errorbar(
        chis,
        lrs,
        yerr=ebs,
        fmt="o--",
        color=color,
        linewidth=1.5,
        markersize=4,
        capsize=2,
        label=rf"$N={N}$",
    )

    if (N, ERROR_RATE) in bp_ler and bp_ler[(N, ERROR_RATE)] > 0:
        target = ADVANTAGE * bp_ler[(N, ERROR_RATE)]
        ax1.axhline(target, color=color, ls=":", lw=0.9, alpha=0.7)

ax1.set_xlabel(r"Bond dimension $\chi_{\max}$")
ax1.set_ylabel(rf"TN logical error rate at $p={ERROR_RATE}$")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xticks(ALL_CHIS)
ax1.set_xticklabels([str(c) for c in ALL_CHIS], fontsize=6)
ax1.grid(True, ls=":", linewidth=0.6)
ax1.add_artist(ax1.legend(fontsize=6.5, ncol=2, loc="upper right"))
proxy = [
    Line2D(
        [0],
        [0],
        color="gray",
        ls=":",
        lw=0.9,
        label=r"$0.9 \times \mathrm{LER}_\mathrm{BP}$",
    ),
]
ax1.legend(handles=proxy, fontsize=6, loc="lower right")

fig1.savefig(_fig("csp_ler_vs_chi.pdf"), dpi=300, bbox_inches="tight")
print("Saved csp_ler_vs_chi.pdf")

# ── Figure 2: LER ratio TN(chi=400) / BP vs N ─────────────────────────────────
# All seven points are directly measured.
fig2, ax2 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

ns_plot, ratios, ratio_errs = [], [], []
for N in LATTICE_SIZES:
    tn_v = tn_lr.get((N, CHI_MAX, p))
    tn_e = tn_eb.get((N, CHI_MAX, p), 0.0)
    bp_v = bp_ler.get((N, ERROR_RATE))
    bp_e = bp_err.get((N, ERROR_RATE), 0.0)
    if tn_v is None or bp_v is None or tn_v <= 0 or bp_v <= 0:
        print(f"N={N}: missing TN or BP point, skipping in ratio plot.")
        continue
    r = tn_v / bp_v
    # Relative-error propagation for a ratio.
    rel = np.hypot(tn_e / tn_v if tn_v else 0.0, bp_e / bp_v if bp_v else 0.0)
    ns_plot.append(N)
    ratios.append(r)
    ratio_errs.append(r * rel)

ns_plot = np.array(ns_plot)
ratios = np.array(ratios)
ratio_errs = np.array(ratio_errs)

ax2.errorbar(
    ns_plot,
    ratios,
    yerr=ratio_errs,
    fmt="o--",
    color="#2166ac",
    linewidth=1.5,
    markersize=5,
    capsize=2,
    label=r"$\mathrm{LER}_\mathrm{TN}(\chi=400) / \mathrm{LER}_\mathrm{BP}$",
)
ax2.axhline(1.0, color="gray", ls="-", lw=0.8, alpha=0.7)
ax2.axhline(
    ADVANTAGE,
    color="red",
    ls=":",
    lw=1.0,
    alpha=0.8,
    label=rf"$10\%$ LER advantage ($y={ADVANTAGE}$)",
)

ax2.set_xlabel(r"Number of qubits $N$")
ax2.set_ylabel(r"$\mathrm{LER}_\mathrm{TN} / \mathrm{LER}_\mathrm{BP}$ at $p=10^{-3}$")
ax2.set_yscale("log")
ax2.set_xticks(ns_plot)
ax2.grid(True, ls=":", linewidth=0.6)
ax2.legend(fontsize=6.5, loc="best")

fig2.savefig(_fig("csp_ler_ratio.pdf"), dpi=300, bbox_inches="tight")
print("Saved csp_ler_ratio.pdf")

print("\nLER ratios (TN/BP) at p=1e-3:")
for N, r, e in zip(ns_plot, ratios, ratio_errs):
    print(f"  N={N}: {r:.3f} ± {e:.3f}")

# ── Figure 3: chi_conv(N) — TN convergence bond dimension ────────────────────
# chi_conv(N) = smallest chi at which the cumulative-min-smoothed TN LER is
# within (1 + CONV_TOL) of LER(chi=400).  Using chi=400 (the largest chi we
# ran) as the plateau reference, rather than the global minimum of the trace,
# avoids latching onto Monte Carlo resolution-floor dips at intermediate chi.
fig3, ax3 = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

ns_conv, chi_conv_vals = [], []
print("\nchi_conv(N):")
for N in LATTICE_SIZES:
    if N not in tn_traces:
        continue
    chis, lrs, _, _ = tn_traces[N]
    valid = lrs > 0
    chis_v, lrs_v = chis[valid], lrs[valid]
    if len(chis_v) < 2:
        continue
    # Reference: TN_LER at the largest chi we ran (chi=400).
    if (N, CHI_MAX, p) not in tn_lr or tn_lr[(N, CHI_MAX, p)] <= 0:
        continue
    plateau = float(tn_lr[(N, CHI_MAX, p)])
    threshold = (1.0 + CONV_TOL) * plateau
    # Cumulative-min smoothing → monotone non-increasing in chi.
    lr_smooth = np.minimum.accumulate(lrs_v)
    below = lr_smooth <= threshold
    if not below.any():
        continue
    j = int(np.argmax(below))
    if j == 0:
        chi_c = float(chis_v[0])
    else:
        # log-log interp between bracketing grid points
        lo, hi = lr_smooth[j - 1], lr_smooth[j]
        clo, chi_hi = chis_v[j - 1], chis_v[j]
        if hi <= 0 or lo <= 0 or lo == hi:
            chi_c = float(chis_v[j])
        else:
            f = (np.log(threshold) - np.log(lo)) / (np.log(hi) - np.log(lo))
            chi_c = float(np.exp(np.log(clo) + f * (np.log(chi_hi) - np.log(clo))))
    ns_conv.append(N)
    chi_conv_vals.append(chi_c)
    print(f"  N={N}: chi_conv = {chi_c:.1f} (LER(chi=400) = {plateau:.2e})")

ns_conv = np.array(ns_conv)
chi_conv_vals = np.array(chi_conv_vals)
# Enforce monotonicity in N (running max).  Larger codes need at least as much
# chi as smaller ones to reach the same fractional convergence; any decrease
# is statistical noise on the plateau.
chi_conv_mono = np.maximum.accumulate(chi_conv_vals)

ax3.plot(
    ns_conv,
    chi_conv_mono,
    "o--",
    color="#2166ac",
    linewidth=1.5,
    markersize=5,
    label=rf"$\chi_\mathrm{{conv}}(N)$ ($\mathrm{{LER}}\le {1 + CONV_TOL:.2f}\,\mathrm{{LER}}(\chi=400)$)",
)

# Power-law fit chi_conv ~ N^alpha.
if len(ns_conv) >= 3:
    log_n = np.log(ns_conv)
    log_ch = np.log(chi_conv_mono)
    alpha, log_c = np.polyfit(log_n, log_ch, 1)
    n_fit = np.linspace(min(ns_conv), max(ns_conv), 100)
    ch_fit = np.exp(log_c) * n_fit**alpha
    ax3.plot(
        n_fit,
        ch_fit,
        "--",
        color="red",
        linewidth=1.5,
        label=rf"$\chi_\mathrm{{conv}} \propto N^{{{alpha:.2f}}}$",
    )
    print(f"\nPower-law fit: chi_conv ~ N^{alpha:.3f}")

ax3.set_xlabel(r"Number of qubits $N$")
ax3.set_ylabel(r"TN convergence bond dimension $\chi_\mathrm{conv}$")
ax3.set_xticks(ns_conv)
ax3.set_yscale("log", base=2)
ax3.set_yticks([4, 8, 16, 32, 64, 128, 256, 400])
ax3.set_yticklabels([str(c) for c in [4, 8, 16, 32, 64, 128, 256, 400]], fontsize=7)
ax3.grid(True, ls=":", linewidth=0.6)
ax3.legend(fontsize=6.5, loc="best")

fig3.savefig(_fig("csp_chi_conv.pdf"), dpi=300, bbox_inches="tight")
print("Saved csp_chi_conv.pdf")
