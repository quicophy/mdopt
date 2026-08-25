"""Generate publication-ready surface code PDFs, matching the notebook pipelines exactly."""

# --- asset paths (the code lives in the package; the data does not) ---
import os as _os

from mdopt.examples.paths import decoding_assets as _decoding_assets, figure as _fig

# Relative data dirs below resolve against the repo-level examples/decoding/.
_DECODING = str(_decoding_assets())
_os.chdir(_DECODING)
# -----------------------------------------------------------------------------

import sys
import os
import numpy as np


from mdopt.examples.decoding.data_handling import (
    process_failure_statistics,
    plot_failure_statistics,
    plot_bond_dimension_scaling,
)

import seaborn as sns
import matplotlib as mpl

# Undo seaborn's global style modifications
sns.reset_orig()
mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)

# ── 1. Threshold plot — matches plotting_surface.ipynb exactly ────────────────
print("Loading surface code data (bond_dim=250)...")
lattice_sizes = [3, 5, 7, 9, 11]
max_bond_dims = [250]

err_rates_dict, failure_rates, error_bars, _, _, _ = process_failure_statistics(
    lattice_sizes=lattice_sizes,
    max_bond_dims=max_bond_dims,
    error_model="Bitflip",
    directory="data-quantum-surface",
)

plot_failure_statistics(
    error_rates_dict=err_rates_dict,
    failure_rates=failure_rates,
    error_bars=error_bars,
    lattice_sizes=lattice_sizes,
    max_bond_dims=max_bond_dims,
    mode="bond_dim",
    yscale="log",
    bb=False,
    save_path=_fig("surface-failure-rate.pdf"),
)
print("Saved surface-failure-rate.pdf")

# ── 2. Bond dimension scaling — matches plotting_bdim_scaling.ipynb exactly ───
print("Loading bond dimension scaling data...")
lattice_sizes_bdim = [3, 4, 5, 6, 7, 8, 9, 10]
max_bond_dims_bdim = list(np.arange(1, 100))

err_rates_dict2, failure_rates2, error_bars2, _, _, _ = process_failure_statistics(
    lattice_sizes=lattice_sizes_bdim,
    max_bond_dims=max_bond_dims_bdim,
    error_model="Bitflip",
    directory="data-quantum-surface-bdimscaling",
)

# Re-apply style before this plot's figure creation
sns.reset_orig()
mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)

previous_cwd = os.getcwd()
try:
    os.chdir(_FIGDIR)
    plot_bond_dimension_scaling(
        failure_rates=failure_rates2,
        target_error_rate=0.06,
        threshold=0.06,
    )
finally:
    os.chdir(previous_cwd)
print("Saved surface-bdim-scaling.pdf")
