"""
Generate ldpc-mps-vs-bp-n20.pdf — MPS-MPO decoder vs BP for n=20.
Saves one pkl per (p, batch) for crash-resilience. Loads existing data.
"""

# --- asset paths (the code lives in the package; the data does not) ---
import os as _os

from mdopt.examples.paths import decoding_assets as _decoding_assets, figure as _fig

# Relative data dirs below resolve against the repo-level examples/decoding/.
_DECODING = str(_decoding_assets())
_os.chdir(_DECODING)
# -----------------------------------------------------------------------------

import sys


import os, pickle, glob
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
NUM_BITS = 20
CHECK_DEGREE = 4
BIT_DEGREE = 3
NUM_CHECKS = int(BIT_DEGREE * NUM_BITS / CHECK_DEGREE)
CHI_MAX = 50
NUM_EXPERIMENTS = 50  # per p value
SEED = 42
p_grid = np.array([0.0001, 0.001, 0.01, 0.1])
tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]
DATA_DIR = "data/classical-ldpc"
os.makedirs(DATA_DIR, exist_ok=True)


def mps_pkl_path(p):
    return os.path.join(DATA_DIR, f"mps_n{NUM_BITS}_chi{CHI_MAX}_p{p:.4f}.pkl")


# ── Load existing failure data for a given p ──────────────────────────────────
def load_existing_failures(p):
    """Load all existing pkl files for this p value (any seed/format)."""
    failures = []
    # New-style files from this script
    path = mps_pkl_path(p)
    if os.path.exists(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        failures.extend(d["failures"])
        return failures
    # Legacy format: numbits{N}_bonddim{chi}_errorrate{p}_seed{seed}.pkl
    pattern = os.path.join(
        DATA_DIR, f"numbits{NUM_BITS}_bonddim{CHI_MAX}_errorrate{p}_seed*.pkl"
    )
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        failures.extend(d["failures"])
    return failures


# ── Run MPS experiments ───────────────────────────────────────────────────────
seed_seq = np.random.SeedSequence(SEED)
mps_results = {}  # p -> list of failures

for p in p_grid:
    existing = load_existing_failures(p)
    n_done = len(existing)
    n_needed = NUM_EXPERIMENTS - n_done
    print(f"p={p:.4f}: {n_done} experiments already done, need {n_needed} more")
    if n_needed <= 0:
        mps_results[p] = existing[:NUM_EXPERIMENTS]
        continue

    fails = list(existing)
    for _ in tqdm(range(n_needed), desc=f"p={p:.4f}"):
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        seed = int(rng.integers(1, 10**8 + 1))
        code = qec.random_regular_code(
            NUM_BITS, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qec.Rng(seed)
        )
        icw, pcw = linear_code_prepare_message(
            code, p, error_model=qec.BinarySymmetricChannel, seed=seed
        )
        cs = linear_code_constraint_sites(code)
        mps = create_custom_product_state(pcw, form="Right-canonical")
        mps = apply_bitflip_bias(mps=mps, sites_to_bias="All", prob_bias_list=p)
        mps = apply_constraints(
            mps,
            cs,
            tensors,
            chi_max=CHI_MAX,
            renormalise=True,
            strategy="Optimised",
            silent=True,
        )
        _, success = decode_message(
            message=mps,
            codeword=create_custom_product_state(icw, form="Right-canonical"),
            num_runs=1,
            chi_max_dmrg=CHI_MAX,
            silent=True,
        )
        fails.append(1 - success)

    # Save after each p value completes
    with open(mps_pkl_path(p), "wb") as f:
        pickle.dump({"p": p, "failures": fails}, f)
    print(f"  LER={np.mean(fails):.4f} (saved)")
    mps_results[p] = fails

# ── Load BP data ──────────────────────────────────────────────────────────────
bp_path = os.path.join(DATA_DIR, f"bp_numbits{NUM_BITS}_seed42.pkl")
with open(bp_path, "rb") as f:
    bp = pickle.load(f)
bp_p = bp["p_grid"]
bp_ler = bp["ler"]
bp_err = bp["ler_err"]

# ── Plot ──────────────────────────────────────────────────────────────────────
cmap = mpl.colormaps["viridis_r"]
norm = Normalize(vmin=0, vmax=1)

fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

ax.errorbar(
    bp_p,
    bp_ler,
    yerr=bp_err,
    fmt="o--",
    linewidth=1.5,
    markersize=4,
    capsize=2,
    color=cmap(norm(0.15)),
    label=rf"BP ($n={NUM_BITS}$, {bp['max_iter']} iter.)",
)

# MPS-MPO: only plot p values with data
mps_p_list, mps_ler_list, mps_err_list = [], [], []
for p in p_grid:
    if p in mps_results and len(mps_results[p]) > 0:
        mps_p_list.append(p)
        mps_ler_list.append(np.mean(mps_results[p]))
        mps_err_list.append(sem(mps_results[p]) if len(mps_results[p]) > 1 else 0)

if mps_p_list:
    ax.errorbar(
        mps_p_list,
        mps_ler_list,
        yerr=mps_err_list,
        fmt="s--",
        linewidth=1.5,
        markersize=4,
        capsize=2,
        color=cmap(norm(0.75)),
        label=rf"MPS-MPO ($n={NUM_BITS}$, $\chi={CHI_MAX}$)",
    )

ax.plot(bp_p, bp_p, ":", color="gray", linewidth=1.0, label=r"$p_L = p$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Physical bit-flip rate $p$")
ax.set_ylabel("Logical error rate")
ax.legend()
ax.grid(True, ls=":", linewidth=0.6)

out = _fig("ldpc-mps-vs-bp-n20.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
