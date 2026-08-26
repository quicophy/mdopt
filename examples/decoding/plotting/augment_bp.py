"""Augment bp_results.pkl with more BP-OSD samples at p=1e-3 for high N.

The original cache runs only 1 rep × BASE_SAMPLES at p=1e-3.  At N=90 that
means only ~150 000 pooled samples — with BP_LER on the order of 1e-5, zero
failures get observed and the BP-OSD baseline is unusable.

This script adds EXTRA_REPS_AT_P001 reps for selected lattice sizes at
p=1e-3 (independent of, and additive to, any existing reps for that cell),
re-pickles in v4 format that records reps per (N, p), and leaves the
remaining cells untouched.

Parallelised across codes per N.
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

import os
import sys
import json
import pickle
import re
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

CACHE_FILE = "bp_results.pkl"
BATCH = 9
TARGET_P = 0.001

# How many more reps × BASE_SAMPLES to run at p=1e-3 for each N that needs it.
# Cells not listed here are left untouched.
EXTRA_REPS_AT_P001 = {
    80: 9,  # adds 9 × 50k = 450 k samples per code  → ~12× total
    90: 49,  # adds 49 × 50k = 2.45 M samples per code → ~50× total
}

NUM_WORKERS = 6  # codes decoded in parallel per N


def list_to_parity_matrix(stabs, num_qubits):
    H = np.zeros((len(stabs), num_qubits), dtype=int)
    for row, stab in enumerate(stabs):
        H[row, stab] = 1
    return H % 2


def load_csp_code(num_qubits, batch, code_id):
    for prefix in ["data-csp-codes", "examples/decoding/data-csp-codes"]:
        path = os.path.join(
            prefix,
            f"batch_{batch}",
            "codes",
            f"qubits_{num_qubits}",
            f"code_{code_id}.json",
        )
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            return data["num_qubits"], data["x_stabs"], data["z_stabs"]
    raise FileNotFoundError(f"N={num_qubits} batch={batch} id={code_id}")


def _worker(args):
    N, code_id, base_samples, target_p, max_p, n_reps = args
    from qldpc.codes import CSSCode

    _, xs, zs = load_csp_code(N, BATCH, code_id)
    Hx = list_to_parity_matrix(xs, N)
    Hz = list_to_parity_matrix(zs, N)
    code = CSSCode(Hx, Hz)
    ler_func = code.get_logical_error_rate_func(
        num_samples=base_samples,
        max_error_rate=max_p,
        pauli_bias=[1, 0, 0],
    )
    extra_failures = 0
    for _ in range(n_reps):
        ler = ler_func(target_p)[0]
        extra_failures += round(ler * base_samples)
    return code_id, extra_failures


def main():
    if not os.path.isfile(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found.", file=sys.stderr)
        sys.exit(1)
    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)

    base = data.get("base_samples", 50_000)
    ps = list(data.get("ps", []))
    ns = data.get("ns", [])
    if TARGET_P not in ps:
        print(f"ERROR: p={TARGET_P} not in cache ps={ps}", file=sys.stderr)
        sys.exit(1)
    i_p = ps.index(TARGET_P)
    max_p = max(ps)

    # Track per-(N, p) reps explicitly (cache v4).
    reps_per_np = data.get("reps_per_np")
    if reps_per_np is None:
        # Bootstrap from v3: p=1e-4 uses reps_low_p[N], everything else uses 1.
        reps_low_p = data.get("reps_low_p", {})
        reps_per_np = {}
        for N in ns:
            for p in ps:
                if p == 0.0001:
                    reps_per_np[(N, p)] = reps_low_p.get(N, 1)
                else:
                    reps_per_np[(N, p)] = 1
        data["reps_per_np"] = reps_per_np
        data["version"] = 4

    # Run augmentation per N.
    for N, n_extra in EXTRA_REPS_AT_P001.items():
        if N not in ns or N not in data:
            print(f"N={N}: not in cache, skipping.")
            continue
        code_dict = data[N]
        cur_reps = reps_per_np.get((N, TARGET_P), 1)
        target_total = cur_reps + n_extra
        print(
            f"N={N}: current reps {cur_reps} → target {target_total} "
            f"(adding {n_extra} × {base:,} samples per code, "
            f"{len(code_dict)} codes)"
        )

        t0 = time.perf_counter()
        jobs = [
            (N, code_id, base, TARGET_P, max_p, n_extra)
            for code_id in sorted(code_dict)
        ]
        with Pool(processes=min(NUM_WORKERS, len(jobs))) as pool:
            for code_id, extra_failures in tqdm(
                pool.imap_unordered(_worker, jobs),
                total=len(jobs),
                desc=f"N={N}",
            ):
                code_dict[code_id][i_p] += extra_failures

        reps_per_np[(N, TARGET_P)] = target_total
        elapsed = time.perf_counter() - t0
        print(f"N={N}: wall {elapsed/60:.1f} min")

        # Persist after each N so a partial run is recoverable.
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"N={N}: saved.")

    # Final pooled BP_LER report for sanity.
    print("\nUpdated BP_LER at p=1e-3:")
    for N in ns:
        cd = data.get(N, {})
        if not cd:
            continue
        reps = reps_per_np.get((N, TARGET_P), 1)
        n_samp_per_code = reps * base
        lers = [arr[i_p] / n_samp_per_code for arr in cd.values()]
        pooled = float(np.mean(lers))
        total_fail = int(sum(arr[i_p] for arr in cd.values()))
        total_samp = n_samp_per_code * len(cd)
        print(
            f"  N={N}: BP_LER={pooled:.3e}  "
            f"(total {total_fail} failures / {total_samp:,} samples)"
        )


if __name__ == "__main__":
    main()
