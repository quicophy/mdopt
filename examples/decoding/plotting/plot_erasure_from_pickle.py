"""Re-plot 5qubit-erasure-failure-rate.pdf from saved pickle."""

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

import sys, pickle
from math import comb


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

PICKLE = "failure_rate_5qubit_erasure_data.pkl"

with open(PICKLE, "rb") as fh:
    data = pickle.load(fh)

failure_rates_sim = data["failure_rates"]
error_bars_sim = data["error_bars"]
erasure_rates = data["erasure_rates"]
max_bond_dims_sim = data["max_bond_dims"]


def p_fail_analytical(p):
    return sum(comb(5, k) * p**k * (1 - p) ** (5 - k) for k in range(3, 6))


cmap = colormaps["viridis_r"]
norm_colors = Normalize(vmin=0, vmax=len(max_bond_dims_sim) - 1)

fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
for index, CHI_MAX in enumerate(max_bond_dims_sim):
    lbl = (
        r"$\chi_{\max} = \infty$"
        if np.isinf(CHI_MAX)
        else rf"$\chi_{{\max}} = {int(CHI_MAX)}$"
    )
    ax.errorbar(
        erasure_rates,
        [failure_rates_sim[CHI_MAX, er] for er in erasure_rates],
        yerr=[error_bars_sim[CHI_MAX, er] for er in erasure_rates],
        fmt="o--",
        label=lbl,
        linewidth=1.5,
        markersize=4,
        capsize=2,
        color=cmap(norm_colors(index)),
    )
p_ref = np.linspace(0, 0.80, 300)
ax.plot(
    p_ref,
    [p_fail_analytical(p) for p in p_ref],
    label=r"$\sum_{k=3}^{5}\binom{5}{k}p^k(1-p)^{5-k}$",
    color="red",
    linewidth=1.5,
    ls="--",
)
ax.set_xlabel(r"Erasure rate $p$")
ax.set_ylabel("Logical error rate")
ax.grid(True, ls=":", linewidth=0.6)
ax.legend()
fig.savefig(_fig("5qubit-erasure-failure-rate.pdf"), dpi=300, bbox_inches="tight")
print("Saved 5qubit-erasure-failure-rate.pdf")
