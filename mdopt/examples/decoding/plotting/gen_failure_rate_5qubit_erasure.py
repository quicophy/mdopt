"""Generate 5qubit-erasure-failure-rate.pdf — LER vs erasure rate for the 5-qubit code."""

# --- asset paths (the code lives in the package; the data does not) ---
import os as _os

from mdopt.examples.paths import decoding_assets as _decoding_assets, figure as _fig

# Relative data dirs below resolve against the repo-level examples/decoding/.
_DECODING = str(_decoding_assets())
_os.chdir(_DECODING)
# -----------------------------------------------------------------------------

import os, sys, pickle
from math import comb
from multiprocessing import Pool

# Pin BLAS to a single thread per worker to avoid oversubscription with multiprocessing.
for var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[var] = "1"


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
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

stabilizer_generators = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
x_logical_operators = ["XXXXX"]
z_logical_operators = ["ZZZZZ"]
num_qubits = 5

PICKLE = "failure_rate_5qubit_erasure_data.pkl"
NUM_EXPERIMENTS = 100000
SEED = 42
NUM_WORKERS = 8
erasure_rates = np.linspace(0.01, 0.50, 25)
max_bond_dims_sim = [np.inf, 64, 16, 8]


def _decode_one_cell(args):
    """Run NUM_EXPERIMENTS decodes for a single (chi, er) pair."""
    chi, er, errors, tiebreak_seed = args
    chi_use = 10000 if np.isinf(chi) else int(chi)
    rng = np.random.default_rng(tiebreak_seed)
    failures = []
    for error in errors:
        dist, _ = decode_custom(
            stabilizers=stabilizer_generators,
            x_logicals=x_logical_operators,
            z_logicals=z_logical_operators,
            error=error,
            chi_max=chi_use,
            bias_prob=0.0,
            bias_type="Bitflip",
            cut=1e-17,
            tolerance=1e-17,
            renormalise=True,
            silent=True,
            contraction_strategy="Naive",
        )
        max_amp = float(np.max(dist))
        eps = 1e-9 * max(max_amp, 1e-30)
        maximizers = [i for i, x in enumerate(dist) if x >= max_amp - eps]
        chosen = int(rng.choice(maximizers))
        failures.append(0 if chosen == 0 else 1)
    return chi, er, float(np.mean(failures)), float(sem(failures))


def main():
    # Pre-generate the error strings per erasure rate, shared across all chi values
    # so the comparison is on identical samples.
    seed_seq = np.random.SeedSequence(SEED)
    errors_per_rate = {}
    for er in tqdm(erasure_rates, desc="Generating errors"):
        errors_per_rate[er] = [
            generate_pauli_error_string(
                num_qubits=num_qubits,
                error_rate=0.0,
                erasure_rate=float(er),
                rng=np.random.default_rng(seed_seq.spawn(1)[0]),
                error_model="Erasure",
            )
            for _ in range(NUM_EXPERIMENTS)
        ]

    # Build job list: one (chi, er) cell per worker job.
    jobs = []
    tiebreak_ss = np.random.SeedSequence(SEED + 1)
    for chi in max_bond_dims_sim:
        for er in erasure_rates:
            jobs.append(
                (
                    chi,
                    er,
                    errors_per_rate[er],
                    int(tiebreak_ss.spawn(1)[0].generate_state(1)[0]),
                )
            )

    failure_rates_sim = {}
    error_bars_sim = {}
    with Pool(processes=NUM_WORKERS) as pool:
        for chi, er, fail, err in tqdm(
            pool.imap_unordered(_decode_one_cell, jobs),
            total=len(jobs),
            desc="Decoding cells",
        ):
            failure_rates_sim[chi, er] = fail
            error_bars_sim[chi, er] = err

    with open(PICKLE, "wb") as fh:
        pickle.dump(
            {
                "failure_rates": failure_rates_sim,
                "error_bars": error_bars_sim,
                "erasure_rates": erasure_rates,
                "max_bond_dims": max_bond_dims_sim,
                "num_experiments": NUM_EXPERIMENTS,
                "seed": SEED,
            },
            fh,
        )

    # Analytical upper bound: failure when >= 3 out of 5 qubits are erased
    # (5-qubit code, d=3, corrects up to d-1=2 erasures).
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
    p_ref = np.linspace(0, 0.50, 300)
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


if __name__ == "__main__":
    main()
