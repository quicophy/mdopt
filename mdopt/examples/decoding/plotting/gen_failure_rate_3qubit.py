"""Generate 3qubit-failure-rate.pdf with physical error rate up to 0.5."""

# --- asset paths (the code lives in the package; the data does not) ---
import os as _os

from mdopt.examples.paths import decoding_assets as _decoding_assets, figure as _fig

# Relative data dirs below resolve against the repo-level examples/decoding/.
_DECODING = str(_decoding_assets())
_os.chdir(_DECODING)
# -----------------------------------------------------------------------------

import sys, pickle


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from scipy.stats import sem
from tqdm import tqdm

from mdopt.examples.decoding.decoding import decode_custom, generate_pauli_error_string

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
num_qubits = 3

PICKLE = "data/cache/failure_rate_3qubit_data.pkl"
_os.makedirs(_os.path.dirname(PICKLE), exist_ok=True)
NUM_EXPERIMENTS = 10000
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
    chi_use = 10000 if np.isinf(CHI_MAX) else int(CHI_MAX)
    print(f"CHI_MAX={CHI_MAX}")
    for er in tqdm(error_rates):
        failures = [
            1
            - decode_custom(
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
            )[1]
            for error in errors_per_rate[er]
        ]
        failure_rates_sim[CHI_MAX, er] = np.mean(failures)
        error_bars_sim[CHI_MAX, er] = sem(failures)

with open(PICKLE, "wb") as fh:
    pickle.dump(
        {
            "failure_rates": failure_rates_sim,
            "error_bars": error_bars_sim,
            "error_rates": error_rates,
            "max_bond_dims": max_bond_dims_sim,
        },
        fh,
    )

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
