"""
Generate csp-bp-failure-rate.pdf — TN (MPS-MPO, chi=400) vs BP-OSD.
One subplot per lattice size; solid = TN, dashed = BP-OSD.
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
from matplotlib.lines import Line2D

from data_handling import process_failure_statistics_csp, process_bp_statistics

mpl.style.use("default")
mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


LATTICE_SIZES = [30, 40, 50, 60, 70, 80, 90]
CHI_MAX = 400

# ── Load TN data ──────────────────────────────────────────────────────────────
print("Loading TN data...")
(
    tn_error_rates_dict,
    tn_failure_rates,
    tn_error_bars,
    _,
    _,
    _,
    tn_num_codes,
) = process_failure_statistics_csp(
    lattice_sizes=LATTICE_SIZES,
    max_bond_dims=[CHI_MAX],
    error_model="Bitflip",
    directory="data-quantum-csp-batch-9",
    equalize=True,
)

# ── Load BP-OSD data ──────────────────────────────────────────────────────────
print("Loading BP-OSD data...")
with open("bp_results.pkl", "rb") as f:
    bp_results = pickle.load(f)

bp_ps, bp_ler, bp_err, bp_num_codes = process_bp_statistics(
    bp_results, lattice_sizes=LATTICE_SIZES
)
reps_dict = bp_results.get("reps_low_p", {})
base_s = bp_results.get("base_samples", 50_000)

# ── Layout: 4 columns × 2 rows ────────────────────────────────────────────────
NCOLS, NROWS = 4, 2
fig, axes = plt.subplots(
    NROWS,
    NCOLS,
    figsize=(7.0, 3.8),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
axes_flat = axes.flatten()

# Hide the unused last panel
for ax in axes_flat[len(LATTICE_SIZES) :]:
    ax.set_visible(False)

COLOR_TN = "#2166ac"  # blue
COLOR_BP = "#d6604d"  # red

for idx, N in enumerate(LATTICE_SIZES):
    ax = axes_flat[idx]

    # TN decoder
    tn_key = (N, CHI_MAX)
    if tn_key in tn_error_rates_dict:
        ers = tn_error_rates_dict[tn_key]
        lers = [tn_failure_rates[N, CHI_MAX, er] for er in ers]
        errs = [tn_error_bars[N, CHI_MAX, er] for er in ers]
        ax.errorbar(
            ers,
            lers,
            yerr=errs,
            fmt="o--",
            color=COLOR_TN,
            linewidth=1.5,
            markersize=4,
            capsize=2,
            label="TN",
        )

    # BP-OSD — non-zero points
    bp_ers_nonzero = [p for p in bp_ps if (N, p) in bp_ler and bp_ler[(N, p)] > 0]
    if bp_ers_nonzero:
        ax.errorbar(
            bp_ers_nonzero,
            [bp_ler[(N, p)] for p in bp_ers_nonzero],
            yerr=[bp_err[(N, p)] for p in bp_ers_nonzero],
            fmt="s--",
            color=COLOR_BP,
            linewidth=1.5,
            markersize=4,
            capsize=2,
            alpha=0.85,
            label="BP-OSD",
        )

    # BP-OSD — upper bounds where 0 failures observed
    for p in bp_ps:
        if (N, p) not in bp_ler or bp_ler[(N, p)] > 0:
            continue
        reps = reps_dict.get(N, 1)
        n_s = reps * base_s if p == 0.0001 else base_s
        upper = -np.log(0.05) / n_s  # 95 % Poisson upper bound
        ax.annotate(
            "",
            xy=(p, upper * 0.3),
            xytext=(p, upper),
            arrowprops=dict(arrowstyle="-|>", color=COLOR_BP, alpha=0.85, lw=1.2),
        )
        ax.plot(p, upper, "s", color=COLOR_BP, ms=3, alpha=0.85)

    # Pseudo-threshold
    all_ps = sorted(set(list(bp_ps) + (tn_error_rates_dict.get(tn_key) or [])))
    ax.plot(all_ps, all_ps, "k:", lw=0.9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e1)
    ax.grid(True, ls=":", linewidth=0.6)

    nc_tn = tn_num_codes.get((N, CHI_MAX), 0)
    nc_bp = bp_num_codes.get(N, 0)
    ax.set_title(rf"$N={N}$  ($n_c={nc_tn}$)", fontsize=7.5)

# Shared axis labels
for ax in axes[-1]:
    ax.set_xlabel(r"Physical error rate $p$")
for row in axes:
    row[0].set_ylabel(r"Logical error rate $p_L$")

# Global legend
style_handles = [
    Line2D(
        [0],
        [0],
        color=COLOR_TN,
        marker="o",
        ls="-",
        lw=1.3,
        ms=3.5,
        label=r"TN ($\chi_{\max}=400$)",
    ),
    Line2D(
        [0],
        [0],
        color=COLOR_BP,
        marker="s",
        ls="--",
        lw=1.3,
        ms=3,
        alpha=0.85,
        label="BP-OSD",
    ),
    Line2D(
        [0],
        [0],
        color=COLOR_BP,
        marker="s",
        ls="none",
        ms=3,
        alpha=0.85,
        label=r"BP-OSD $95\%$ upper bound",
    ),
    Line2D([0], [0], color="k", ls=":", lw=0.9, label=r"$p_L = p$"),
]
fig.legend(
    handles=style_handles,
    loc="lower right",
    bbox_to_anchor=(1.0, 0.0),
    fontsize=6.5,
    framealpha=0.9,
    ncol=1,
)

fig.savefig(_fig("csp-bp-failure-rate.pdf"), dpi=300, bbox_inches="tight")
print("Saved csp-bp-failure-rate.pdf")
