"""Run the surface-code 2x2 strategy comparison at chi in {64, 128, 256, 512}.

Designed to run overnight alongside an existing background job that already
uses ~6 cores: this script uses only NUM_WORKERS=4 and processes chi groups
strictly sequentially, saving a separate pickle after each chi group so partial
progress is preserved.
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
)

LATTICE_SIZE = 5
SEED = 42
NUM_WORKERS = 4
chi_max_list = [64, 128, 256, 512]  # processed in order
p_list = [1e-3, 3e-3, 1e-2, 3e-2]
orderings = ["Natural", "Optimised"]
verticals = ["Naive", "Optimised"]
CUT = 1e-10
TOLERANCE = 1e-10
BIAS_TYPE = "Bitflip"

# Per-(chi, p) trial counts — scaled down as chi grows so slowest cells
# finish in ~30 min wall-time each at NUM_WORKERS=4.
TRIALS_PER_CHI_P = {
    (64, 1e-3): 5000,
    (64, 3e-3): 2500,
    (64, 1e-2): 1000,
    (64, 3e-2): 500,
    (128, 1e-3): 3000,
    (128, 3e-3): 1500,
    (128, 1e-2): 600,
    (128, 3e-2): 300,
    (256, 1e-3): 1500,
    (256, 3e-3): 800,
    (256, 1e-2): 400,
    (256, 3e-2): 200,
    (512, 1e-3): 600,
    (512, 3e-3): 300,
    (512, 1e-2): 150,
    (512, 3e-2): 80,
}


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
    )


def _pickle_path(chi):
    return f"vert_horiz_surface_L{LATTICE_SIZE}_chi{chi}_data.pkl"


def run_chi_group(chi, errors_per_p, n_qubits):
    pkl = _pickle_path(chi)
    if os.path.exists(pkl):
        print(f"[chi={chi}] {pkl} already exists, skipping.")
        return

    print(f"[chi={chi}] starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_start = time.perf_counter()

    jobs = []
    for p in p_list:
        N = TRIALS_PER_CHI_P[(chi, p)]
        errs = errors_per_p[p][:N]
        for ordering in orderings:
            for vertical in verticals:
                jobs.append((chi, p, ordering, vertical, errs))

    results = {}
    with Pool(processes=NUM_WORKERS) as pool:
        for (
            c,
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
            desc=f"chi={chi}",
        ):
            results[(c, p, ordering, vertical)] = {
                "chi": c,
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

    elapsed = time.perf_counter() - t_start
    with open(pkl, "wb") as fh:
        pickle.dump(
            {
                "results": results,
                "chi_max_list": [chi],
                "p_list": p_list,
                "orderings": orderings,
                "verticals": verticals,
                "trials_per_chi_p": {
                    k: v for k, v in TRIALS_PER_CHI_P.items() if k[0] == chi
                },
                "lattice_size": LATTICE_SIZE,
                "n_qubits": n_qubits,
                "wallclock_s": elapsed,
            },
            fh,
        )
    print(f"[chi={chi}] saved {pkl}  (wall {elapsed/60:.1f} min)")


def main():
    code = _build_code()
    n = len(code)
    print(f"Surface L={LATTICE_SIZE}: n={n}, k={code.num_x_logicals()}")

    # Pre-generate paired errors per p, sized to the largest N at any chi
    # for that p. Each chi group then slices off its own prefix.
    seed_seq = np.random.SeedSequence(SEED)
    errors_per_p = {}
    for p in p_list:
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        N_max = max(TRIALS_PER_CHI_P[(c, p)] for c in chi_max_list)
        errors_per_p[p] = [
            generate_pauli_error_string(
                num_qubits=n, error_rate=p, error_model=BIAS_TYPE, rng=rng
            )
            for _ in range(N_max)
        ]
    print(f"Generated paired errors per p.")

    for chi in chi_max_list:
        run_chi_group(chi, errors_per_p, n)

    print(f"\nAll chi groups complete.")


if __name__ == "__main__":
    main()
