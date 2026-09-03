"""Profiling benchmark suite for mdopt's hot paths.

Each workload is deterministic (fixed seeds), sized to run in tens of seconds,
and returns a correctness fingerprint. The fingerprints are the contract for
the optimisation work on this branch: any change that moves a fingerprint
beyond 1e-10 is a behaviour change, not an optimisation.

Run:  python benchmarks/bench_suite.py [--profile] [--workload NAME]
Profiles land in benchmarks/results/<workload>.pstats plus a text top-30.
"""

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import numpy as np
import qecstruct as qec

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def wl_surface_bitflip():
    """Code-capacity surface-code decode: the quantum_surface workload."""
    from mdopt.examples.decoding.decoding import (
        decode_css,
        generate_pauli_error_string,
    )

    code = qec.hypergraph_product(qec.repetition_code(5), qec.repetition_code(5))
    rng = np.random.default_rng(51)
    outputs = []
    for _ in range(6):
        error = generate_pauli_error_string(
            len(code), 0.05, rng=rng, error_model="Bitflip"
        )
        _, success = decode_css(
            code,
            error,
            chi_max=64,
            bias_type="Bitflip",
            bias_prob=0.05,
            renormalise=True,
            silent=True,
            contraction_strategy="Optimised",
        )
        outputs.append(float(success))
    return outputs


def wl_shor_depolarising():
    """Small-code depolarising decode: dense readout path end to end."""
    from mdopt.examples.decoding.decoding import (
        decode_css,
        generate_pauli_error_string,
    )

    code = qec.shor_code()
    rng = np.random.default_rng(7)
    outputs = []
    for _ in range(40):
        error = generate_pauli_error_string(len(code), 0.1, rng=rng)
        _, success = decode_css(
            code,
            error,
            chi_max=128,
            bias_type="Depolarising",
            bias_prob=0.1,
            renormalise=True,
            silent=True,
        )
        outputs.append(float(success))
    return outputs


def wl_classical_ldpc():
    """Classical LDPC pipeline: constraints + Dephasing DMRG readout."""
    from mdopt.examples.decoding.decoding import (
        apply_bitflip_bias,
        apply_constraints,
        decode_message,
        linear_code_constraint_sites,
        linear_code_prepare_message,
    )
    from mdopt.mps.utils import create_custom_product_state
    from mdopt.optimiser.utils import SWAP, XOR_BULK, XOR_LEFT, XOR_RIGHT

    outputs = []
    for seed in (11, 12, 13):
        code = qec.random_regular_code(48, 36, 3, 4, qec.Rng(seed))
        first, second = linear_code_prepare_message(
            code, 0.1, error_model=qec.BinarySymmetricChannel, seed=seed
        )
        sites = linear_code_constraint_sites(code)
        start = create_custom_product_state(first, form="Right-canonical")
        state = create_custom_product_state(second, form="Right-canonical")
        state = apply_bitflip_bias(mps=state, sites_to_bias="All", prob_bias_list=0.1)
        state = apply_constraints(
            state,
            sites,
            [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT],
            chi_max=64,
            renormalise=True,
            strategy="Optimised",
            silent=True,
        )
        _, overlap = decode_message(
            message=state,
            codeword=start,
            num_runs=1,
            chi_max_dmrg=64,
            silent=True,
        )
        outputs.append(float(overlap))
    return outputs


def wl_dmrg_ground_state():
    """Plain DMRG on a transverse-field Ising chain (optimiser hot path)."""
    from mdopt.mps.utils import create_simple_product_state
    from mdopt.optimiser.dmrg import DMRG

    num_sites = 24
    identity = np.eye(2)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    mpo = []
    for site in range(num_sites):
        tensor = np.zeros((3, 3, 2, 2))
        tensor[0, 0] = identity
        tensor[2, 2] = identity
        tensor[0, 1] = pauli_z
        tensor[1, 2] = pauli_z
        tensor[0, 2] = pauli_x
        if site == 0:
            mpo.append(tensor[0:1, :, :, :])
        elif site == num_sites - 1:
            mpo.append(tensor[:, 2:3, :, :])
        else:
            mpo.append(tensor)
    from mdopt.contractor.contractor import mps_mpo_contract
    from mdopt.mps.utils import inner_product

    mps = create_simple_product_state(num_sites, which="+")
    engine = DMRG(mps, mpo, chi_max=48, cut=1e-12, mode="SA", silent=True)
    engine.run(2)
    # The energy depends on the optimised state everywhere the norm does not:
    # renormalised bond updates make norm() ~ 1.0 for any state, correct or
    # not, so it cannot serve as the correctness fingerprint.
    ground = engine.mps
    h_ground = mps_mpo_contract(ground, mpo, chi_max=int(1e4), renormalise=False)
    energy = float(np.real(inner_product(ground, h_ground)))
    return [round(energy, 10)]


WORKLOADS = {
    "surface_bitflip": wl_surface_bitflip,
    "shor_depolarising": wl_shor_depolarising,
    "classical_ldpc": wl_classical_ldpc,
    "dmrg_ground_state": wl_dmrg_ground_state,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default=None)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    names = [args.workload] if args.workload else sorted(WORKLOADS)
    summary = {}
    for name in names:
        func = WORKLOADS[name]
        started = time.perf_counter()
        if args.profile:
            profiler = cProfile.Profile()
            fingerprint = profiler.runcall(func)
            wall = time.perf_counter() - started
            profiler.dump_stats(RESULTS / f"{name}.pstats")
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats("cumulative").print_stats(30)
            (RESULTS / f"{name}.top30.txt").write_text(stream.getvalue())
        else:
            fingerprint = func()
            wall = time.perf_counter() - started
        summary[name] = {"wall_s": round(wall, 3), "fingerprint": fingerprint}
        print(f"{name:>20}: {wall:7.2f} s  fingerprint={fingerprint}", flush=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
