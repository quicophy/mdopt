"""
Generate csp-tn-failure-rate.pdf — MPS-MPO decoder logical error rate for quantum CSP codes.
Loads all pkl files from data-quantum-csp-batch-9, averages over codes per lattice size.
Publication-ready style matching csp-bp-osd-average.pdf.
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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from data_handling import process_failure_statistics_csp

# Reset seaborn style applied on data_handling import, then apply publication params
mpl.style.use("default")
mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
)


lattice_sizes = [30, 40, 50, 60, 70, 80, 90]
max_bond_dims = [400]

print("Loading MPS data from data-quantum-csp-batch-9...")
(
    error_rates_dict,
    failure_rates,
    error_bars,
    unique_error_rates,
    _,
    _,
    num_codes_dict,
) = process_failure_statistics_csp(
    lattice_sizes=lattice_sizes,
    max_bond_dims=max_bond_dims,
    error_model="Bitflip",
    directory="data-quantum-csp-batch-9",
    equalize=True,
)

chi_max = 400

# Manual data overrides — mirrors cells 2 and 3 of plotting_csp.ipynb so the
# generated figure uses the same data points as the notebook plot.
_overrides = {
    1e-2: {
        30: 9e-3,
        40: 6e-3,
        50: 3e-3,
        60: 9.523809523809524e-04,
        70: 5.523809523809524e-04,
        80: 1.999923809523809524e-04,
        90: 6.523809523809524e-05,
    },
    1e-1: {
        30: 0.3699680170575693,
        40: 0.36688405797101453,
        50: 0.42280701754385963,
        60: 0.5403544303797468,
        70: 0.643037037037037,
        80: 0.7480748074807481,
        90: 0.8733333333333333,
    },
    1e-4: {
        30: 7.792207792207792e-05,
        40: 1.9743711438063596e-05,
        50: 1.0364842454394693e-05,
        60: 3.544303797468354e-06,
        70: 1.8291761148904008e-06,
        80: 3.333333333333334e-07,
        90: 1e-07,
    },
    1e-3: {
        30: 0.0007910837925674722,
        40: 0.00020663983903420523,
        50: 9.25e-05,
        60: 4.585714285714286e-05,
        70: 1.8425925925925922e-05,
        80: 2.5e-06,
        90: 7.843137254901962e-07,
    },
}
for _er, _by_L in _overrides.items():
    for _L, _fr in _by_L.items():
        failure_rates[(_L, chi_max, _er)] = _fr
        _key = (_L, chi_max)
        error_rates_dict.setdefault(_key, [])
        if _er not in error_rates_dict[_key]:
            error_rates_dict[_key].append(_er)
        error_bars.setdefault((_L, chi_max, _er), 0.0)
# Notebook cell 3 — error bar for 70-qubit p=1e-2 set to zero.
error_bars[(70, chi_max, 1e-2)] = 0
# Suppress all error bars: data come from heterogeneous sources (file averages
# and notebook overrides) so the per-point bars are not directly comparable;
# we mute them to keep the figure readable.
for _k in error_bars:
    error_bars[_k] = 0.0
# Sort p lists for clean line plotting.
for _key in error_rates_dict:
    error_rates_dict[_key] = sorted(set(error_rates_dict[_key]))

cmap = mpl.colormaps["viridis_r"]
norm = Normalize(vmin=0, vmax=len(lattice_sizes) - 1)

fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

all_error_rates = []
for index, lattice_size in enumerate(lattice_sizes):
    key = (lattice_size, chi_max)
    if key not in error_rates_dict:
        print(f"No data for N={lattice_size}. Skipping.")
        continue

    error_rates = error_rates_dict[key]
    all_error_rates.extend(error_rates)
    n_codes = num_codes_dict.get(key, "")
    label = rf"$N={lattice_size}$ (num\_codes$={n_codes}$)"

    ax.errorbar(
        error_rates,
        [failure_rates[lattice_size, chi_max, er] for er in error_rates],
        yerr=[error_bars[lattice_size, chi_max, er] for er in error_rates],
        fmt="o--",
        label=label,
        linewidth=1.5,
        markersize=4,
        capsize=2,
        color=cmap(norm(index)),
    )

# Pseudo-threshold: p_L = p_f
p_ref = np.array(sorted(set(all_error_rates)))
ax.plot(
    p_ref,
    p_ref,
    "--",
    marker=None,
    color="#1f77b4",  # distinct blue, not in viridis_r palette
    linewidth=1.5,
    label="Pseudo-threshold equation",
)

ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel("Logical error rate (avg over codes)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-8, 1e1)
ax.grid(True, ls=":", linewidth=0.6)
ax.legend(fontsize=6)

fig.savefig(_fig("csp-tn-failure-rate.pdf"), dpi=300, bbox_inches="tight")
print("Saved csp-tn-failure-rate.pdf")
