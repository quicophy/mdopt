"""
Generate ldpc-bdim-scaling-n24.pdf — logical error rate vs bond dimension for n=24.
Saves one pkl per (chi, error_rate) for crash-resilience. Skips already-done combos.
Plots whatever data is available at the end.
"""

# --- asset paths (the code lives in the package; the data does not) ---
import os as _os

from mdopt.examples.paths import decoding_assets as _decoding_assets, figure as _fig

# Relative data dirs below resolve against the repo-level examples/decoding/.
_DECODING = str(_decoding_assets())
_os.chdir(_DECODING)
# -----------------------------------------------------------------------------

import sys


import os, pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import qecstruct as qec
from scipy.stats import sem
from tqdm import tqdm

from mdopt.mps.utils import create_custom_product_state
from mdopt.optimiser.utils import SWAP, XOR_BULK, XOR_LEFT, XOR_RIGHT
from mdopt.examples.decoding.decoding import (
    linear_code_constraint_sites,
    linear_code_prepare_message,
    apply_bitflip_bias,
    apply_constraints,
    decode_message,
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

# ── Parameters ────────────────────────────────────────────────────────────────
NUM_BITS = 24
CHECK_DEGREE = 4
BIT_DEGREE = 3
NUM_CHECKS = int(BIT_DEGREE * NUM_BITS / CHECK_DEGREE)
NUM_EXPERIMENTS = 30
SEED = 123
max_bond_dims = [8, 16, 32, 64, 128]
error_rates = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]
DATA_DIR = "data/classical-ldpc"
os.makedirs(DATA_DIR, exist_ok=True)


def pkl_path(chi, er):
    return os.path.join(DATA_DIR, f"bd_n{NUM_BITS}_chi{chi}_er{er:.2f}.pkl")


# ── Pre-generate codes (reuse across chi values) ───────────────────────────────
print("Pre-generating codes...")
seed_seq = np.random.SeedSequence(SEED)
codes_all, icw_all, pcw_all = {}, {}, {}
for er in error_rates:
    codes_all[er], icw_all[er], pcw_all[er] = [], [], []
    for _ in range(NUM_EXPERIMENTS):
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        seed = int(rng.integers(1, 10**8 + 1))
        code = qec.random_regular_code(
            NUM_BITS, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qec.Rng(seed)
        )
        icw, pcw = linear_code_prepare_message(
            code, er, error_model=qec.BinarySymmetricChannel, seed=seed
        )
        codes_all[er].append(code)
        icw_all[er].append(icw)
        pcw_all[er].append(pcw)

# ── Run experiments ───────────────────────────────────────────────────────────
for chi in max_bond_dims:
    print(f"\nchi_max={chi}")
    for er in tqdm(error_rates):
        path = pkl_path(chi, er)
        if os.path.exists(path):
            print(f"  chi={chi}, er={er:.2f}: already done, skipping")
            continue
        fails = []
        for l in range(NUM_EXPERIMENTS):
            cs = linear_code_constraint_sites(codes_all[er][l])
            mps = create_custom_product_state(pcw_all[er][l], form="Right-canonical")
            mps = apply_bitflip_bias(mps=mps, sites_to_bias="All", prob_bias_list=er)
            mps = apply_constraints(
                mps,
                cs,
                tensors,
                chi_max=chi,
                renormalise=True,
                strategy="Optimised",
                silent=True,
            )
            _, success = decode_message(
                message=mps,
                codeword=create_custom_product_state(
                    icw_all[er][l], form="Right-canonical"
                ),
                num_runs=1,
                chi_max_dmrg=chi,
                silent=True,
            )
            fails.append(1 - success)
        with open(path, "wb") as f:
            pickle.dump({"chi": chi, "er": er, "failures": fails}, f)

# ── Load all available results and plot ───────────────────────────────────────
cmap = mpl.colormaps["viridis_r"]
norm = Normalize(vmin=0, vmax=len(max_bond_dims) - 1)
fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

for idx, chi in enumerate(max_bond_dims):
    y_vals, yerr_vals, x_vals = [], [], []
    for er in error_rates:
        path = pkl_path(chi, er)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        y_vals.append(np.mean(d["failures"]))
        yerr_vals.append(sem(d["failures"]))
        x_vals.append(er)
    if not x_vals:
        continue
    ax.errorbar(
        x_vals,
        y_vals,
        yerr=yerr_vals,
        fmt="o--",
        label=rf"$\chi = {chi}$",
        linewidth=1.5,
        markersize=4,
        capsize=2,
        color=cmap(norm(idx)),
    )

ax.set_xlabel("Physical bit-flip rate $p$")
ax.set_ylabel("Logical error rate")
ax.grid(True, ls=":", linewidth=0.6)
ax.legend()
out = _fig("ldpc-bdim-scaling-n24.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved {out}")
