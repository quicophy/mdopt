"""Generate the 2x2 (vertical x horizontal) strategy comparison for the
distance-L surface code via hypergraph product of two repetition codes.

Local, parallel, paired-error timing comparison.
"""

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
# -----------------------------------------------------------------------------

import os, sys, pickle, time
from multiprocessing import Pool

for v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[v] = "1"


import numpy as np
import qecstruct as qec
from tqdm import tqdm

from examples.decoding.decoding import (
    decode_css,
    generate_pauli_error_string,
    css_code_checks,
)
from mdopt.optimiser.utils import optimise_qubit_order

LATTICE_SIZE = 5
PICKLE = f"vert_horiz_surface_L{LATTICE_SIZE}_data.pkl"
SEED = 42
NUM_WORKERS = 8
chi_max_list = [4, 8, 16, 32]
p_list = [1e-3, 3e-3, 1e-2, 3e-2]
orderings = ["Natural", "Optimised"]
verticals = ["Naive", "Optimised"]
CUT = 1e-10
TOLERANCE = 1e-10
BIAS_TYPE = "Bitflip"
# Tiered trials per p: more trials at low p (fast cells). 5x previous tier.
TRIALS_PER_P = {1e-3: 10000, 3e-3: 5000, 1e-2: 2500, 3e-2: 1500}


def _build_code():
    rep = qec.repetition_code(LATTICE_SIZE)
    return qec.hypergraph_product(rep, rep)


def _decode_one_cell(args):
    chi, p, ordering, vertical, errors = args
    code = _build_code()
    n_fail = 0
    per_decode_t = []
    for e in errors:
        t0 = time.perf_counter()
        _, success = decode_css(
            code=code,
            error=e,
            chi_max=chi,
            cut=CUT,
            bias_type=BIAS_TYPE,
            bias_prob=p,
            renormalise=True,
            silent=True,
            contraction_strategy=vertical,
            qubit_order_strategy=ordering,
            tolerance=TOLERANCE,
        )
        per_decode_t.append(time.perf_counter() - t0)
        if not success:
            n_fail += 1
    arr = np.array(per_decode_t)
    N = len(errors)
    ler = n_fail / N
    return (
        chi,
        p,
        ordering,
        vertical,
        ler,
        float(np.sqrt(ler * (1 - ler) / N)),
        float(arr.mean()),
        float(arr.std(ddof=1) / np.sqrt(N)),
        float(np.median(arr)),
        N,
        float(arr.sum()),
        arr.astype(np.float32),
    )  # keep per-decode wall-times for paired analysis


def main():
    code = _build_code()
    n = len(code)
    print(f"Surface L={LATTICE_SIZE}: n={n}, k={code.num_x_logicals()}")

    pm_x = code.x_stabs_binary()
    Hx = np.zeros((pm_x.num_rows(), pm_x.num_columns()), dtype=int)
    for r, cols in enumerate(pm_x.rows()):
        for c in cols:
            Hx[r, c] = 1
    pm_z = code.z_stabs_binary()
    Hz = np.zeros((pm_z.num_rows(), pm_z.num_columns()), dtype=int)
    for r, cols in enumerate(pm_z.rows()):
        for c in cols:
            Hz[r, c] = 1
    H = np.vstack([Hx, Hz])
    perm = optimise_qubit_order(H)

    cx_nat, cz_nat = css_code_checks(code, qubit_perm=None)
    cx_opt, cz_opt = css_code_checks(code, qubit_perm=perm)
    ms = lambda cs: max(max(c) - min(c) for c in cs if len(c) >= 2)
    ts = lambda cs: sum(max(c) - min(c) for c in cs if len(c) >= 2)
    bw_nat = (ms(cx_nat + cz_nat), ts(cx_nat + cz_nat))
    bw_opt = (ms(cx_opt + cz_opt), ts(cx_opt + cz_opt))
    print(f"  Natural:   max_span={bw_nat[0]:4d}, total_span={bw_nat[1]:6d}")
    print(f"  Optimised: max_span={bw_opt[0]:4d}, total_span={bw_opt[1]:6d}")
    print(
        f"  Reduction: max {100*(bw_nat[0]-bw_opt[0])/bw_nat[0]:.1f}%, "
        f"total {100*(bw_nat[1]-bw_opt[1])/bw_nat[1]:.1f}%"
    )

    # Pre-generate errors per p, shared across all (chi, ordering, vertical).
    seed_seq = np.random.SeedSequence(SEED)
    errors_per_p = {}
    for p in tqdm(p_list, desc="Generating errors"):
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        N = TRIALS_PER_P[p]
        errors_per_p[p] = [
            generate_pauli_error_string(
                num_qubits=n, error_rate=p, error_model=BIAS_TYPE, rng=rng
            )
            for _ in range(N)
        ]

    jobs = []
    for chi in chi_max_list:
        for p in p_list:
            for ordering in orderings:
                for vertical in verticals:
                    jobs.append((chi, p, ordering, vertical, errors_per_p[p]))

    print(f"\nDispatching {len(jobs)} cells over {NUM_WORKERS} workers…")
    results = {}
    with Pool(processes=NUM_WORKERS) as pool:
        for (
            chi,
            p,
            ordering,
            vertical,
            ler,
            ler_sem,
            t_mean,
            t_sem,
            t_med,
            N,
            wall,
            per_decode_t,
        ) in tqdm(
            pool.imap_unordered(_decode_one_cell, jobs),
            total=len(jobs),
            desc="Decoding cells",
        ):
            results[(chi, p, ordering, vertical)] = {
                "chi": chi,
                "p": p,
                "ordering": ordering,
                "vertical": vertical,
                "ler": ler,
                "ler_sem": ler_sem,
                "avg_time_s": t_mean,
                "avg_time_sem": t_sem,
                "median_time_s": t_med,
                "n_trials": N,
                "wallclock_total_s": wall,
                "per_decode_t": per_decode_t,
            }

    with open(PICKLE, "wb") as fh:
        pickle.dump(
            {
                "results": results,
                "chi_max_list": chi_max_list,
                "p_list": p_list,
                "orderings": orderings,
                "verticals": verticals,
                "trials_per_p": TRIALS_PER_P,
                "lattice_size": LATTICE_SIZE,
                "bandwidth_natural": bw_nat,
                "bandwidth_optimised": bw_opt,
                "n_qubits": n,
            },
            fh,
        )
    print(f"\nSaved {PICKLE}")


if __name__ == "__main__":
    main()
