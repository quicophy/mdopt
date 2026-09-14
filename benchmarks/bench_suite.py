"""Profiling benchmark suite for mdopt's hot paths.

Each workload is deterministic (fixed seeds), sized to run in tens of seconds,
and returns a correctness fingerprint. The fingerprints are the contract for
the optimisation work: ``--check`` compares them against the committed
``benchmarks/baseline.json`` -- exact values (energies, verdicts, overlaps)
must match to 1e-10, and chi-truncated posterior entries must stay within
1e-2: a different but equally valid SVD gauge in a near-degenerate spectrum
changes which directions chi_max keeps, and that moves small class masses at
this level while leaving verdicts and converged results untouched (old and
new decoders agree to 6e-14 at chi=1e5). A real behaviour change moves
verdicts or exact values. ``--write-baseline``
records a new baseline after a change that is validated some other way
(exact-enumeration tests, agreement at converged chi).

Run:  python benchmarks/bench_suite.py [--profile] [--check] [--workload NAME]
Profiles land in benchmarks/results/<workload>.pstats plus a text top-30.
"""

import argparse
import cProfile
import io
import json
import math
import pstats
import time
from pathlib import Path

import numpy as np
import qecstruct as qec
import stim

# Imported once here, not inside the workloads: function-local imports made
# the first-run wall time and profile depend on invocation order (a workload
# run alone paid cold-import cost that a full sorted suite had already paid).
from mdopt.contractor.contractor import mps_mpo_contract
from mdopt.decoding.dem import decode_dem, dem_to_problem
from mdopt.examples.ising.ising import IsingMPO
from mdopt.decoding.decoding import (
    apply_bitflip_bias,
    apply_constraints,
    decode_css,
    decode_message,
    generate_pauli_error_string,
    linear_code_constraint_sites,
    linear_code_prepare_message,
)
from mdopt.mps.utils import (
    create_custom_product_state,
    create_simple_product_state,
    inner_product,
)
from mdopt.optimiser.dmrg import DMRG
from mdopt.optimiser.utils import SWAP, XOR_BULK, XOR_LEFT, XOR_RIGHT

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def wl_surface_bitflip():
    """Code-capacity surface-code decode: the quantum_surface workload."""
    code = qec.hypergraph_product(qec.repetition_code(5), qec.repetition_code(5))
    rng = np.random.default_rng(51)
    outputs = []
    for _ in range(6):
        error = generate_pauli_error_string(
            len(code), 0.05, rng=rng, error_model="Bitflip"
        )
        dense, success = decode_css(
            code,
            error,
            chi_max=64,
            bias_type="Bitflip",
            bias_prob=0.05,
            renormalise=True,
            silent=True,
            contraction_strategy="Optimised",
        )
        # The full posterior, not just the verdict: a wrong posterior
        # with an unmoved argmax must still move the fingerprint.
        outputs.append([float(success)] + [round(float(x), 10) for x in dense])
    return outputs


def wl_css_optimised():
    """Surface-code decode under the RCM qubit ordering (optimise_qubit_order)."""
    code = qec.hypergraph_product(qec.repetition_code(5), qec.repetition_code(5))
    rng = np.random.default_rng(52)
    outputs = []
    for _ in range(2):
        error = generate_pauli_error_string(
            len(code), 0.05, rng=rng, error_model="Bitflip"
        )
        dense, success = decode_css(
            code,
            error,
            chi_max=64,
            bias_type="Bitflip",
            bias_prob=0.05,
            renormalise=True,
            silent=True,
            contraction_strategy="Optimised",
            qubit_order_strategy="Optimised",
        )
        outputs.append([float(success)] + [round(float(x), 10) for x in dense])
    return outputs


def _dem_case(task, distance, rounds, p, seed, num_sampled, num_keep):
    """A circuit-level DEM plus its busiest sampled syndromes.

    The busiest syndromes (most detection events) carry nontrivial posteriors
    and drive the slowest contractions, so they are the ones worth timing and
    fingerprinting; a quiet syndrome decodes to (1, 1e-10) and would not move.
    """
    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{task}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    problem = dem_to_problem(
        circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    )
    sampler = circuit.compile_detector_sampler(seed=seed)
    detections, _ = sampler.sample(num_sampled, separate_observables=True)
    order = np.argsort(-detections.sum(axis=1), kind="stable")[:num_keep]
    return problem, detections[order].astype(int)


def _dem_rows(problem, syndromes, chi_max):
    outputs = []
    for syndrome in syndromes:
        masses, flips = decode_dem(problem, syndrome, chi_max=chi_max)
        posterior = masses / masses.sum()
        # [verdict, *normalised class masses]: the verdict is exact, the
        # masses are chi-truncated posterior entries.
        outputs.append([float(flips[0])] + [round(float(x), 10) for x in posterior])
    return outputs


def wl_dem_d3():
    """Circuit-level DEM decode, d=3 r=3 p=0.8% memory-X (the Fig. 1d cell)."""
    problem, syndromes = _dem_case("x", 3, 3, 0.008, seed=3, num_sampled=64, num_keep=8)
    return _dem_rows(problem, syndromes, chi_max=32)


def wl_dem_d5():
    """Circuit-level DEM decode, d=5 r=5 p=0.5% memory-Z (the campaign cell)."""
    problem, syndromes = _dem_case("z", 5, 5, 0.005, seed=5, num_sampled=32, num_keep=1)
    return _dem_rows(problem, syndromes, chi_max=32)


def wl_shor_depolarising():
    """Small-code depolarising decode: dense readout path end to end."""
    code = qec.shor_code()
    rng = np.random.default_rng(7)
    outputs = []
    for _ in range(40):
        error = generate_pauli_error_string(len(code), 0.1, rng=rng)
        dense, success = decode_css(
            code,
            error,
            chi_max=128,
            bias_type="Depolarising",
            bias_prob=0.1,
            renormalise=True,
            silent=True,
        )
        # The full posterior, not just the verdict: a wrong posterior
        # with an unmoved argmax must still move the fingerprint.
        outputs.append([float(success)] + [round(float(x), 10) for x in dense])
    return outputs


def wl_classical_ldpc():
    """Classical LDPC pipeline: constraints + Dephasing DMRG readout."""
    outputs = []
    for seed in (11, 12, 13):
        code = qec.random_regular_code(48, 36, 3, 4, qec.Rng(seed))
        first, second = linear_code_prepare_message(
            code, 0.1, error_model=qec.BinarySymmetricChannel, seed=seed
        )
        sites = linear_code_constraint_sites(code)
        start = create_custom_product_state(first, form="Right-canonical")
        state = create_custom_product_state(second, form="Right-canonical")
        received = state.copy()
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
        # The DMRG verdict alone is blind to the constrained state: it stays
        # 1.0 whenever the MAP codeword is unchanged. Observables of the state
        # the hot path actually produces come first: its overlaps with the
        # transmitted codeword and with the received (biased) message, and
        # the Schmidt spectrum at the middle bond (a corrupted contraction
        # leaves weight outside the codeword, which shows up here).
        middle = state.num_sites // 2
        _, spectra = state.copy().move_orth_centre(
            middle, return_singular_values=True, renormalise=True
        )
        schmidt = sorted(
            (float(v) for v in np.asarray(spectra[-1]).ravel()), reverse=True
        )
        outputs.append(
            [
                round(float(abs(inner_product(start, state))), 10),
                round(float(abs(inner_product(received, state))), 10),
                float(overlap),
                *[round(v, 10) for v in schmidt[:4]],
            ]
        )
    return outputs


def wl_dmrg_ground_state():
    """Plain DMRG on a transverse-field Ising chain (optimiser hot path)."""
    num_sites = 24
    mpo = IsingMPO(num_sites=num_sites, h_magnetic=1.0).hamiltonian_mpo()
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


BASELINE = HERE / "baseline.json"
# Workloads whose fingerprint rows are [verdict, *posterior entries at chi_max].
POSTERIOR_WORKLOADS = {
    "surface_bitflip",
    "shor_depolarising",
    "css_optimised",
    "dem_d3",
    "dem_d5",
}

WORKLOADS = {
    "surface_bitflip": wl_surface_bitflip,
    "css_optimised": wl_css_optimised,
    "shor_depolarising": wl_shor_depolarising,
    "classical_ldpc": wl_classical_ldpc,
    "dmrg_ground_state": wl_dmrg_ground_state,
    "dem_d3": wl_dem_d3,
    "dem_d5": wl_dem_d5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default=None)
    # Mutually exclusive: writing the baseline first and then checking against
    # it would compare every fingerprint with itself and always pass.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="compare against baseline.json"
    )
    mode.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the fingerprints of the workloads run by this invocation",
    )
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    names = [args.workload] if args.workload else sorted(WORKLOADS)
    # Merge into the previous summary so a --workload run does not discard the
    # other entries; profiled runs are marked, since cProfile inflates wall time.
    summary = {}
    if (RESULTS / "summary.json").exists():
        summary = json.loads((RESULTS / "summary.json").read_text())
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
        summary[name] = {
            "wall_s": round(wall, 3),
            "fingerprint": fingerprint,
            "profiled": bool(args.profile),
        }
        print(f"{name:>20}: {wall:7.2f} s  fingerprint={fingerprint}", flush=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.write_baseline:
        # Update only the workloads this invocation ran: a targeted run must
        # neither shrink the committed baseline to the selected workload nor
        # promote stale cached fingerprints of the others.
        baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        for name in names:
            baseline[name] = summary[name]["fingerprint"]
        BASELINE.write_text(json.dumps(baseline, indent=2))
        print(f"baseline written: {BASELINE}")
    if args.check:
        baseline = json.loads(BASELINE.read_text())
        failures = []
        # Only the workloads this invocation ran; summary also carries cached
        # entries from earlier runs, which a targeted --check must not judge.
        for name in names:
            failures += _compare(name, summary[name]["fingerprint"], baseline[name])
        if failures:
            print("FINGERPRINT MISMATCH:\n  " + "\n  ".join(failures))
            raise SystemExit(1)
        print("fingerprints match baseline")


def _compare(name, got, want, path=""):
    """Exact for scalars/verdicts; 1e-2 for chi-truncated posterior entries."""
    tolerance = 1e-2 if name in POSTERIOR_WORKLOADS else 1e-10
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            return [f"{name}{path}: shape changed"]
        return [
            f
            for i, (g, w) in enumerate(zip(got, want))
            for f in _compare(name, g, w, f"{path}[{i}]")
        ]
    # the leading verdict of a posterior row is exact; the entries are not
    if name in POSTERIOR_WORKLOADS and path.endswith("[0]") and path.count("[") == 2:
        tolerance = 1e-10
    got_value, want_value = float(got), float(want)
    # A NaN would otherwise pass: abs(nan) > tolerance is False.
    if not (math.isfinite(got_value) and math.isfinite(want_value)):
        return [f"{name}{path}: non-finite value {got} (baseline {want})"]
    if abs(got_value - want_value) > tolerance:
        return [f"{name}{path}: {got} vs baseline {want}"]
    return []


if __name__ == "__main__":
    main()
