"""Generate publication-ready PDFs for the 3-qubit repetition code."""

# --- asset paths (the code lives in the package; the data does not) ---
import os as _os

from mdopt.examples.paths import decoding_assets as _decoding_assets, figure as _fig

# Relative data dirs below resolve against the repo-level examples/decoding/.
_DECODING = str(_decoding_assets())
_os.chdir(_DECODING)
# -----------------------------------------------------------------------------

import sys


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
from scipy.stats import sem
from tqdm import tqdm

from mdopt.mps.utils import create_custom_product_state
from mdopt.optimiser.utils import SWAP, COPY_LEFT, XOR_BULK, XOR_LEFT, XOR_RIGHT
from mdopt.examples.decoding.decoding import (
    custom_code_constraint_sites,
    custom_code_logicals_sites,
    apply_bitflip_bias,
    apply_constraints,
    decode_custom,
    generate_pauli_error_string,
)

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

# Bit-flip repetition code: Z-type stabilisers detect the X errors
# (textbook symplectic pairing).
stabilizer_generators = ["ZZI", "IZZ"]
x_logical_operators = ["XXX"]
z_logical_operators = ["ZZZ"]
logical_operators = x_logical_operators + z_logical_operators
num_qubits = 3
num_logicals = 2
num_sites = 2 * num_qubits + num_logicals  # 8

constraint_sites = custom_code_constraint_sites(
    stabilizer_generators, logical_operators
)
logicals_sites = custom_code_logicals_sites(x_logical_operators, z_logical_operators)
constraints_tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]
logicals_tensors = [COPY_LEFT, XOR_BULK, SWAP, XOR_RIGHT]
sites_to_bias = list(range(num_logicals, num_sites))

# ── 1. Heatmaps ──────────────────────────────────────────────────────────────
print("Computing heatmaps...")
error_state = "0" * (num_sites - num_logicals)
logicals_state = "+" * num_logicals
error_mps = create_custom_product_state(
    string=logicals_state + error_state, tolerance=1e-17, form="Left-canonical"
)
error_mps = apply_bitflip_bias(
    mps=error_mps, prob_bias_list=0.01, sites_to_bias=sites_to_bias
)

entropies, bond_dims_heatmap = [], []
for i in [0, 1]:
    error_mps, entrps, bnd_dims = apply_constraints(
        error_mps,
        logicals_sites[i],
        logicals_tensors,
        cut=1e-17,
        chi_max=1e4,
        renormalise=True,
        strategy="Naive",
        return_entropies_and_bond_dims=True,
    )
    entropies += entrps
    bond_dims_heatmap += bnd_dims

error_mps, entrps, bnd_dims = apply_constraints(
    error_mps,
    constraint_sites,
    constraints_tensors,
    cut=1e-17,
    chi_max=1e4,
    renormalise=True,
    strategy="Naive",
    return_entropies_and_bond_dims=True,
)
entropies += entrps
bond_dims_heatmap += bnd_dims

entropies = np.array(entropies)
bd_arr = np.array(bond_dims_heatmap, dtype=float)
bd_arr[bd_arr < 1] = 1
n_steps = entropies.shape[0]

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), constrained_layout=True)

im0 = axes[0].imshow(entropies, cmap="viridis", aspect="auto")
fig.colorbar(im0, ax=axes[0], label="Entanglement entropy")
axes[0].set_xlabel("Bond index")
axes[0].set_ylabel("Constraint step")
axes[0].set_yticks(range(n_steps))
axes[0].set_yticklabels(range(n_steps))

vmin_bd, vmax_bd = 2**1, 2**10
im1 = axes[1].imshow(
    bd_arr, cmap="viridis", norm=LogNorm(vmin=vmin_bd, vmax=vmax_bd), aspect="auto"
)
fig.colorbar(
    im1,
    ax=axes[1],
    format=FuncFormatter(lambda x, pos: rf"$2^{{{int(round(np.log2(max(x, 1))))}}}$"),
    ticks=[2**i for i in range(1, 11, 3)],
    label="Bond dimension",
)
axes[1].set_xlabel("Bond index")
axes[1].set_ylabel("Constraint step")
axes[1].set_yticks(range(n_steps))
axes[1].set_yticklabels(range(n_steps))

fig.savefig(_fig("3qubit-heatmaps.pdf"), dpi=300, bbox_inches="tight")
print("Saved 3qubit-heatmaps.pdf")

# ── 2. Truncation error ───────────────────────────────────────────────────────
print("Computing truncation error...")
bond_dim_vals = [np.inf, 16, 8, 4]
inv_bond_dims = [1 / bd for bd in bond_dim_vals]
compress_vals = [10000 if np.isinf(chi) else int(chi) for chi in bond_dim_vals]
trunc_errors = [
    np.linalg.norm(
        error_mps.compress(
            chi_max=chi, renormalise=True, return_truncation_errors=True
        )[1]
    )
    for chi in compress_vals
]

fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
ax.plot(inv_bond_dims, trunc_errors, marker="o", linewidth=1.5, markersize=4)
ax.set_xlabel(r"$1/\chi_{\max}$")
ax.set_ylabel("Truncation error")
ax.set_yscale("log")
ax.grid(True, ls=":", linewidth=0.6)
fig.savefig(_fig("3qubit-truncation-error.pdf"), dpi=300, bbox_inches="tight")
print("Saved 3qubit-truncation-error.pdf")

# ── 3. Logical operator probabilities ─────────────────────────────────────────
print("Computing logical probabilities...")
logical_values = [[] for _ in range(4)]
for chi in bond_dim_vals:
    chi_use = 10000 if np.isinf(chi) else chi
    mps = create_custom_product_state(
        string=logicals_state + error_state, tolerance=1e-17, form="Right-canonical"
    )
    mps = apply_bitflip_bias(mps=mps, prob_bias_list=0.1, sites_to_bias=sites_to_bias)
    for i in [0, 1]:
        mps = apply_constraints(
            mps,
            logicals_sites[i],
            logicals_tensors,
            renormalise=True,
            strategy="Optimised",
            chi_max=chi_use,
            silent=True,
            cut=1e-17,
        )
    mps = apply_constraints(
        mps,
        constraint_sites,
        constraints_tensors,
        renormalise=True,
        strategy="Optimised",
        chi_max=chi_use,
        silent=True,
        cut=1e-17,
    )
    logical = mps.marginal(
        sites_to_marginalise=list(range(num_logicals, len(error_state) + num_logicals))
    ).dense(flatten=True, renormalise=True, norm=1)
    for i in range(4):
        logical_values[i].append(logical[i])

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), constrained_layout=True)
axes[0].plot(inv_bond_dims, logical_values[0], marker="o", linewidth=1.5, markersize=4)
axes[0].set_xlabel(r"$1/\chi_{\max}$")
axes[0].set_ylabel(r"$\Pr(I)$")
axes[0].grid(True, ls=":", linewidth=0.6)
for vals, lbl, mks in zip(
    logical_values[1:], [r"$\Pr(X)$", r"$\Pr(Z)$", r"$\Pr(Y)$"], ["o", "s", "^"]
):
    axes[1].plot(
        inv_bond_dims, vals, marker=mks, linewidth=1.5, markersize=4, label=lbl
    )
axes[1].set_xlabel(r"$1/\chi_{\max}$")
axes[1].set_ylabel("Logical operator probability")
axes[1].grid(True, ls=":", linewidth=0.6)
axes[1].legend()
fig.savefig(_fig("3qubit-logical-probs.pdf"), dpi=300, bbox_inches="tight")
print("Saved 3qubit-logical-probs.pdf")

# ── 4. Failure rate (Monte Carlo) ─────────────────────────────────────────────
print("Running failure rate simulation (2000 experiments)...")
NUM_EXPERIMENTS = 2000
SEED = 123
error_rates = np.linspace(0.01, 0.50, 23)
max_bond_dims_sim = [np.inf, 16, 8, 4]
seed_seq = np.random.SeedSequence(SEED)

errors_per_rate = {
    er: [
        generate_pauli_error_string(
            num_qubits=num_qubits,
            error_rate=er,
            rng=np.random.default_rng(seed_seq.spawn(1)[0]),
            error_model="Bitflip",
        )
        for _ in range(NUM_EXPERIMENTS)
    ]
    for er in error_rates
}

failure_rates_sim = {}
error_bars_sim = {}
for CHI_MAX in max_bond_dims_sim:
    chi_use = 10000 if np.isinf(CHI_MAX) else CHI_MAX
    print(f"  CHI_MAX={CHI_MAX}")
    for er in tqdm(error_rates):
        failures = []
        for error in errors_per_rate[er]:
            _, success = decode_custom(
                stabilizers=stabilizer_generators,
                x_logicals=x_logical_operators,
                z_logicals=z_logical_operators,
                error=error,
                chi_max=chi_use,
                bias_type="Bitflip",
                bias_prob=0.1,
                cut=1e-17,
                tolerance=1e-17,
                renormalise=True,
                silent=True,
                contraction_strategy="Naive",
            )
            failures.append(1 - success)
        failure_rates_sim[CHI_MAX, er] = np.mean(failures)
        error_bars_sim[CHI_MAX, er] = sem(failures)

cmap = colormaps["viridis_r"]
norm_colors = Normalize(vmin=0, vmax=len(max_bond_dims_sim) - 1)
f = lambda p: 3 * p**2 - 2 * p**3

fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
for index, CHI_MAX in enumerate(max_bond_dims_sim):
    lbl = (
        r"$\chi_{\max} = \infty$"
        if np.isinf(CHI_MAX)
        else rf"$\chi_{{\max}} = {int(CHI_MAX)}$"
    )
    ax.errorbar(
        error_rates,
        [failure_rates_sim[CHI_MAX, er] for er in error_rates],
        yerr=[error_bars_sim[CHI_MAX, er] for er in error_rates],
        fmt="o--",
        label=lbl,
        linewidth=1.5,
        markersize=4,
        capsize=2,
        color=cmap(norm_colors(index)),
    )
ax.plot(
    error_rates,
    [f(p) for p in error_rates],
    label=r"$3p^2 - 2p^3$",
    color="red",
    linewidth=1.5,
    ls="--",
)
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel("Logical error rate")
ax.grid(True, ls=":", linewidth=0.6)
ax.legend()
fig.savefig(_fig("3qubit-failure-rate.pdf"), dpi=300, bbox_inches="tight")
print("Saved 3qubit-failure-rate.pdf")
print("All 3-qubit PDFs done.")
