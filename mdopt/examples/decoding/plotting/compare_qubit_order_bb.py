"""
Compare natural vs. optimised qubit ordering for the [[72, 12, 6]] BB code.

Metrics reported:
  - Bandwidth of the combined parity check matrix (max span of any check)
    before and after qubit reordering.
  - Logical error rate (avg over trials).

Usage:
    poetry run python examples/decoding/compare_qubit_order_bb.py
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
import time
import numpy as np
from tqdm import tqdm


from mdopt.decoding.decoding import (
    create_bb_code,
    generate_pauli_error_string,
    css_code_checks,
    decode_css,
)
from mdopt.optimiser.utils import optimise_qubit_order

# ── BB code parameters (Bravyi et al. 2024 [[72, 12, 6]] code) ────────────────
ORDER_X = 6
ORDER_Y = 6
POLY_A = "x**3 + y + y**2"
POLY_B = "y**3 + x + x**2"

# ── Experiment settings ────────────────────────────────────────────────────────
CHI_MAX = 40
ERROR_RATE = 1e-3
BIAS_PROB = 1e-3
BIAS_TYPE = "Bitflip"
NUM_TRIALS = 50_000
SEED = 42
CUT = 1e-10
TOLERANCE = 1e-10


def bandwidth(checks_x, checks_z):
    spans = [max(c) - min(c) for c in checks_x + checks_z if len(c) >= 2]
    return max(spans) if spans else 0


def total_span(checks_x, checks_z):
    return sum(max(c) - min(c) for c in checks_x + checks_z if len(c) >= 2)


def run_trials(code, error_rate, num_trials, rng, chi_max, qubit_order_strategy, label):
    """Run `num_trials` decode attempts with a live tqdm progress bar."""
    n_errors = 0
    elapsed_total = 0.0

    bar = tqdm(
        total=num_trials,
        desc=f"{label:10s}",
        unit="trial",
        dynamic_ncols=True,
    )
    for i in range(num_trials):
        error = generate_pauli_error_string(
            num_qubits=len(code),
            error_rate=error_rate,
            error_model=BIAS_TYPE,
            rng=rng,
        )
        t0 = time.perf_counter()
        result = decode_css(
            code=code,
            error=error,
            chi_max=chi_max,
            cut=CUT,
            bias_type=BIAS_TYPE,
            bias_prob=BIAS_PROB,
            renormalise=True,
            silent=True,
            contraction_strategy="Naive",
            qubit_order_strategy=qubit_order_strategy,
            tolerance=TOLERANCE,
        )
        elapsed_total += time.perf_counter() - t0

        _, success = result
        if not success:
            n_errors += 1

        # Update postfix every trial so the user sees live numbers.
        bar.set_postfix(
            failures=n_errors,
            LER=f"{n_errors / (i + 1):.4f}",
            avg_t=f"{elapsed_total / (i + 1):.2f}s",
            refresh=False,
        )
        bar.update(1)

    bar.close()
    ler = n_errors / num_trials
    sem = np.sqrt(ler * (1 - ler) / num_trials)
    return ler, sem, elapsed_total / num_trials


def main():
    print(f"BB code: orders=({ORDER_X},{ORDER_Y}), A={POLY_A}, B={POLY_B}")
    code = create_bb_code(ORDER_X, ORDER_Y, POLY_A, POLY_B)
    n = len(code)
    kx = code.num_x_logicals()
    kz = code.num_z_logicals()
    print(f"[[n={n}, k={kx}]] CSS code  ({kx} X-logicals, {kz} Z-logicals)")

    # ── Parity matrices ────────────────────────────────────────────────────────
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

    # ── Bandwidth ──────────────────────────────────────────────────────────────
    checks_x_nat, checks_z_nat = css_code_checks(code, qubit_perm=None)
    perm = optimise_qubit_order(H)
    checks_x_opt, checks_z_opt = css_code_checks(code, qubit_perm=perm)

    bw_nat = bandwidth(checks_x_nat, checks_z_nat)
    bw_opt = bandwidth(checks_x_opt, checks_z_opt)
    ts_nat = total_span(checks_x_nat, checks_z_nat)
    ts_opt = total_span(checks_x_opt, checks_z_opt)

    print(f"\n── Bandwidth (max MPO span) ──────────────────────")
    print(f"  Natural:   max_span={bw_nat:4d},  total_span={ts_nat:6d}")
    print(f"  Optimised: max_span={bw_opt:4d},  total_span={ts_opt:6d}")
    print(
        f"  Reduction: max_span {100*(bw_nat-bw_opt)/bw_nat:.1f}%,"
        f"  total_span {100*(ts_nat-ts_opt)/ts_nat:.1f}%"
    )

    # ── Decoding trials ────────────────────────────────────────────────────────
    print(
        f"\n── Decoding trials (chi_max={CHI_MAX}, p={ERROR_RATE:.0e},"
        f" N={NUM_TRIALS:,} trials) ──"
    )

    rng_nat = np.random.default_rng(SEED)
    rng_opt = np.random.default_rng(SEED)  # identical seed → identical errors

    ler_nat, sem_nat, t_nat = run_trials(
        code, ERROR_RATE, NUM_TRIALS, rng_nat, CHI_MAX, "Natural", "Natural"
    )
    ler_opt, sem_opt, t_opt = run_trials(
        code, ERROR_RATE, NUM_TRIALS, rng_opt, CHI_MAX, "Optimised", "Optimised"
    )

    print(f"\n── Results ───────────────────────────────────────")
    print(
        f"  Natural:   LER={ler_nat:.6f} ± {sem_nat:.6f},  avg time/decode={t_nat:.3f}s"
    )
    print(
        f"  Optimised: LER={ler_opt:.6f} ± {sem_opt:.6f},  avg time/decode={t_opt:.3f}s"
    )
    if ler_nat > 0 and ler_opt > 0:
        print(f"  LER ratio (nat/opt): {ler_nat/ler_opt:.2f}×")


if __name__ == "__main__":
    main()
