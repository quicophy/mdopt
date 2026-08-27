"""
Generate ldpc-heatmaps-n16.pdf — entropies and bond dims during decoding, NO truncation.
Averages over NUM_EXPERIMENTS samples for a clean heatmap.
"""

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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import qecstruct as qec
from tqdm import tqdm

from mdopt.mps.utils import create_custom_product_state
from mdopt.optimiser.utils import SWAP, XOR_BULK, XOR_LEFT, XOR_RIGHT
from mdopt.examples.decoding.decoding import (
    linear_code_constraint_sites,
    linear_code_prepare_message,
    apply_bitflip_bias,
    apply_constraints,
)

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)

# ── Parameters ────────────────────────────────────────────────────────────────
NUM_BITS = 16
CHECK_DEGREE = 4
BIT_DEGREE = 3
NUM_CHECKS = int(BIT_DEGREE * NUM_BITS / CHECK_DEGREE)  # 12
ERROR_RATE = 0.15
CHI_MAX = int(1e6)  # effectively no truncation for n=16 (max chi = 2^8 = 256)
NUM_EXPERIMENTS = 10
SEED = 123
tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]

# ── Run experiments ───────────────────────────────────────────────────────────
seed_seq = np.random.SeedSequence(SEED)
entropies_all = np.zeros((NUM_EXPERIMENTS, NUM_CHECKS, NUM_BITS - 1))
bond_dims_all = np.zeros((NUM_EXPERIMENTS, NUM_CHECKS, NUM_BITS - 1))

print(f"Running {NUM_EXPERIMENTS} experiments (n={NUM_BITS}, no truncation)...")
for l in tqdm(range(NUM_EXPERIMENTS)):
    rng = np.random.default_rng(seed_seq.spawn(1)[0])
    seed = int(rng.integers(1, 10**8 + 1))
    code = qec.random_regular_code(
        NUM_BITS, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qec.Rng(seed)
    )
    constraint_sites = linear_code_constraint_sites(code)
    _, perturbed_cw = linear_code_prepare_message(
        code, ERROR_RATE, error_model=qec.BinarySymmetricChannel, seed=seed
    )
    mps = create_custom_product_state(perturbed_cw, form="Right-canonical")
    mps = apply_bitflip_bias(mps=mps, sites_to_bias="All", prob_bias_list=ERROR_RATE)
    mps, entrs, bons = apply_constraints(
        mps,
        constraint_sites,
        tensors,
        chi_max=CHI_MAX,
        renormalise=True,
        strategy="Naive",
        silent=True,
        return_entropies_and_bond_dims=True,
    )
    entropies_all[l] = np.array(entrs)
    bond_dims_all[l] = np.array(bons, dtype=float)

# Average over experiments
entropies = entropies_all.mean(axis=0)  # shape: (NUM_CHECKS, NUM_BITS-1)
bond_dims = bond_dims_all.mean(axis=0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), constrained_layout=True)
n_steps = entropies.shape[0]

for ax, data, label in zip(
    axes,
    [entropies, bond_dims],
    ["Entanglement entropy", "Bond dimension"],
):
    im = ax.imshow(
        data, aspect="auto", origin="upper", cmap="viridis", interpolation="nearest"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.03)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("Bond index")
    ax.set_ylabel("Constraint step")
    # Integer y-ticks starting from 0
    ax.set_yticks(range(n_steps))
    ax.set_yticklabels(range(n_steps))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

os.makedirs("data-classical-ldpc", exist_ok=True)
out = _fig("data-classical-ldpc/ldpc-heatmaps-n16.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
