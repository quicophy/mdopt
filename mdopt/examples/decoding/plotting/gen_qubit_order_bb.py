"""Generate Figure 8.14: qubit ordering comparison for the [[72, 12, 6]] BB code.

Two-panel figure (LER vs p, avg time vs p), one curve per chi_max in
{8, 16, 24, 32, 40}, Natural ordering shown as solid lines, RCM-optimised
ordering as dashed. The two orderings see the *same* sampled error strings
(per p) so the comparison is paired.
"""

# --- asset paths (the code lives in the package; the data does not) ---
import os as _os

from mdopt.examples.paths import decoding_assets as _decoding_assets, figure as _fig

# Relative data dirs below resolve against the repo-level examples/decoding/.
_DECODING = str(_decoding_assets())
_os.chdir(_DECODING)
# -----------------------------------------------------------------------------

import os, sys, pickle, time
from multiprocessing import Pool

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
from tqdm import tqdm

from mdopt.examples.decoding.decoding import (
    create_bb_code,
    generate_pauli_error_string,
    css_code_checks,
    decode_css,
)
from mdopt.optimiser.utils import optimise_qubit_order

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

ORDER_X = 6
ORDER_Y = 6
POLY_A = "x**3 + y + y**2"
POLY_B = "y**3 + x + x**2"

PICKLE = "qubit_order_bb_data.pkl"
PLOT = _fig("bb-qubit-ordering.pdf")
NUM_EXPERIMENTS = 10000
SEED = 42
NUM_WORKERS = 8
chi_max_list = [8, 16, 24, 32, 40]
p_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
orderings = ["Natural", "Optimised"]
CUT = 1e-10
TOLERANCE = 1e-10
BIAS_TYPE = "Bitflip"
CODE_PARAMS = (ORDER_X, ORDER_Y, POLY_A, POLY_B)


def _decode_one_cell(args):
    """Decode NUM_EXPERIMENTS errors under one (chi, p, ordering) cell."""
    chi, p, ordering, errors = args
    code = create_bb_code(*CODE_PARAMS)
    n_fail = 0
    t0 = time.perf_counter()
    for e in errors:
        _, success = decode_css(
            code=code,
            error=e,
            chi_max=chi,
            cut=CUT,
            bias_type=BIAS_TYPE,
            bias_prob=p,
            renormalise=True,
            silent=True,
            contraction_strategy="Naive",
            qubit_order_strategy=ordering,
            tolerance=TOLERANCE,
        )
        if not success:
            n_fail += 1
    dt = time.perf_counter() - t0
    N = len(errors)
    ler = n_fail / N
    return chi, p, ordering, ler, float(np.sqrt(ler * (1 - ler) / N)), dt / N


def main():
    # ── Code construction and bandwidth metrics ────────────────────────────────
    code = create_bb_code(*CODE_PARAMS)
    n = len(code)
    pm_x = code.x_stabs_binary()
    H_x = np.zeros((pm_x.num_rows(), pm_x.num_columns()), dtype=int)
    for r, cols in enumerate(pm_x.rows()):
        for c in cols:
            H_x[r, c] = 1
    pm_z = code.z_stabs_binary()
    H_z = np.zeros((pm_z.num_rows(), pm_z.num_columns()), dtype=int)
    for r, cols in enumerate(pm_z.rows()):
        for c in cols:
            H_z[r, c] = 1
    H = np.vstack([H_x, H_z])
    perm = optimise_qubit_order(H)

    cx_nat, cz_nat = css_code_checks(code, qubit_perm=None)
    cx_opt, cz_opt = css_code_checks(code, qubit_perm=perm)
    max_nat = max(max(c) - min(c) for c in cx_nat + cz_nat if len(c) >= 2)
    max_opt = max(max(c) - min(c) for c in cx_opt + cz_opt if len(c) >= 2)
    sum_nat = sum(max(c) - min(c) for c in cx_nat + cz_nat if len(c) >= 2)
    sum_opt = sum(max(c) - min(c) for c in cx_opt + cz_opt if len(c) >= 2)

    print(f"BB code [[n={n}, k={code.num_x_logicals()}]]")
    print(f"  Natural:   max_span={max_nat:4d},  total_span={sum_nat:6d}")
    print(f"  Optimised: max_span={max_opt:4d},  total_span={sum_opt:6d}")
    print(
        f"  Reduction: max {100*(max_nat-max_opt)/max_nat:.1f}%, "
        f"total {100*(sum_nat-sum_opt)/sum_nat:.1f}%"
    )

    # ── Generate shared errors per p ───────────────────────────────────────────
    seed_seq = np.random.SeedSequence(SEED)
    errors_per_p = {}
    for p in tqdm(p_list, desc="Generating errors"):
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        errors_per_p[p] = [
            generate_pauli_error_string(
                num_qubits=n,
                error_rate=p,
                error_model=BIAS_TYPE,
                rng=rng,
            )
            for _ in range(NUM_EXPERIMENTS)
        ]

    # ── Build job list and run in parallel ─────────────────────────────────────
    jobs = []
    for chi in chi_max_list:
        for p in p_list:
            for ordering in orderings:
                jobs.append((chi, p, ordering, errors_per_p[p]))

    results = {}
    with Pool(processes=NUM_WORKERS) as pool:
        for chi, p, ordering, ler, ler_sem, avg_t in tqdm(
            pool.imap_unordered(_decode_one_cell, jobs),
            total=len(jobs),
            desc="Decoding cells",
        ):
            results[chi, p, ordering] = (ler, ler_sem, avg_t)

    with open(PICKLE, "wb") as fh:
        pickle.dump(
            {
                "results": results,
                "chi_max_list": chi_max_list,
                "p_list": p_list,
                "orderings": orderings,
                "num_experiments": NUM_EXPERIMENTS,
                "bandwidth_natural": (max_nat, sum_nat),
                "bandwidth_optimised": (max_opt, sum_opt),
                "bias_type": BIAS_TYPE,
                "code_params": CODE_PARAMS,
            },
            fh,
        )

    # ── Plotting ───────────────────────────────────────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    cmap = colormaps["viridis_r"]
    norm = Normalize(vmin=0, vmax=len(chi_max_list) - 1)

    # Natural solid, RCM dashed, as the module docstring above promises.
    style = {"Natural": ("-", "o"), "Optimised": ("--", "s")}
    for idx, chi in enumerate(chi_max_list):
        color = cmap(norm(idx))
        for ordering in orderings:
            ls, marker = style[ordering]
            lers = [results[chi, p, ordering][0] for p in p_list]
            sems = [results[chi, p, ordering][1] for p in p_list]
            times = [results[chi, p, ordering][2] for p in p_list]
            ax_l.errorbar(
                p_list,
                lers,
                yerr=sems,
                fmt=marker,
                ls=ls,
                color=color,
                markersize=4,
                capsize=2,
                linewidth=1.5,
            )
            ax_r.plot(
                p_list,
                times,
                marker=marker,
                ls=ls,
                color=color,
                markersize=4,
                linewidth=1.5,
            )

    # Two-column legend: chi colours (via natural-style proxy lines) + ordering styles.
    chi_handles = [
        plt.Line2D(
            [],
            [],
            color=cmap(norm(i)),
            ls="--",
            marker="o",
            markersize=4,
            label=rf"$\chi_{{\max}}={chi}$",
        )
        for i, chi in enumerate(chi_max_list)
    ]
    ord_handles = [
        plt.Line2D(
            [], [], color="k", ls="-", marker="o", markersize=4, label="Natural"
        ),
        plt.Line2D([], [], color="k", ls="--", marker="s", markersize=4, label="RCM"),
    ]
    leg1 = ax_l.legend(handles=chi_handles, loc="lower right", framealpha=0.9)
    ax_l.add_artist(leg1)
    ax_l.legend(handles=ord_handles, loc="upper left", framealpha=0.9)

    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel(r"Physical error rate $p$")
    ax_l.set_ylabel("Logical error rate")
    ax_l.grid(True, ls=":", linewidth=0.6)

    ax_r.set_xscale("log")
    ax_r.set_xlabel(r"Physical error rate $p$")
    ax_r.set_ylabel(r"Avg.\ time per decode (s)")
    ax_r.grid(True, ls=":", linewidth=0.6)

    fig.savefig(PLOT, dpi=300, bbox_inches="tight")
    print(f"Saved {PLOT}")


if __name__ == "__main__":
    main()
