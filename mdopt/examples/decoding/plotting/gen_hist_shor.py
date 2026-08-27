"""
Generate shor-correction-histograms.pdf for Shor's 9-qubit code.
Shows correction statistics for all weight-1 (27), weight-2 (324), weight-3 (2268) Pauli errors.
Each panel has 4 bars (I, X, Z, Y) counting how many errors decoded to each logical class.
Hyperparameters match shor.ipynb exactly.
"""

# --- asset paths (the code lives in the package; the data does not) ---
import os as _os

from mdopt.examples.paths import decoding_assets as _decoding_assets, figure as _fig

# Relative data dirs below resolve against the repo-level examples/decoding/.
_DECODING = str(_decoding_assets())
_os.chdir(_DECODING)
# -----------------------------------------------------------------------------

import sys


import itertools
import numpy as np
import qecstruct as qec
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

from mdopt.examples.decoding.decoding import decode_css, map_distribution_to_pauli

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

# ── Shor code ─────────────────────────────────────────────────────────────────
code = qec.shor_code()
num_qubits = len(code)  # 9

# ── Build exhaustive error sets ───────────────────────────────────────────────
OPS = ["X", "Z", "Y"]


def make_errors(weight):
    errors = []
    for positions in itertools.combinations(range(num_qubits), weight):
        for ops in itertools.product(OPS, repeat=weight):
            err = ["I"] * num_qubits
            for pos, op in zip(positions, ops):
                err[pos] = op
            errors.append("".join(err))
    return errors


errors_w1 = make_errors(1)  # 27 errors
errors_w2 = make_errors(2)  # 324 errors
errors_w3 = make_errors(3)  # 2268 errors

print(
    f"Weight-1: {len(errors_w1)}, Weight-2: {len(errors_w2)}, Weight-3: {len(errors_w3)}"
)

# ── Decode every error (hyperparameters from shor.ipynb) ──────────────────────
common = dict(bias_type="Bitflip", renormalise=True, silent=True)

print("Decoding weight-1 errors (bias_prob=0)...")
dists_w1 = [decode_css(code, err, bias_prob=0, **common)[0] for err in tqdm(errors_w1)]

print("Decoding weight-2 errors (bias_prob=1e-2)...")
dists_w2 = [
    decode_css(code, err, bias_prob=1e-2, **common)[0] for err in tqdm(errors_w2)
]

print("Decoding weight-3 errors (bias_prob=1e-2)...")
dists_w3 = [
    decode_css(code, err, bias_prob=1e-2, **common)[0] for err in tqdm(errors_w3)
]

corrections_w1 = map_distribution_to_pauli(dists_w1)
corrections_w2 = map_distribution_to_pauli(dists_w2)
corrections_w3 = map_distribution_to_pauli(dists_w3)

# ── Plot ──────────────────────────────────────────────────────────────────────
labels_order = ["I", "X", "Z", "Y"]

datasets = [
    (corrections_w1, f"1-qubit errors ({len(corrections_w1)})"),
    (corrections_w2, f"2-qubit errors ({len(corrections_w2)})"),
    (corrections_w3, f"3-qubit errors ({len(corrections_w3)})"),
]

fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), constrained_layout=True)

for ax, (corrections, title) in zip(axes, datasets):
    counts = Counter(corrections)
    vals = [counts.get(l, 0) for l in labels_order]
    ax.bar(labels_order, vals, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Correction")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=9)
    ax.grid(True, ls=":", linewidth=0.6, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig(_fig("shor-correction-histograms.pdf"), dpi=300, bbox_inches="tight")
print("Saved shor-correction-histograms.pdf")
print(f"Weight-1: {dict(Counter(corrections_w1))}")
print(f"Weight-2: {dict(Counter(corrections_w2))}")
print(f"Weight-3: {dict(Counter(corrections_w3))}")
