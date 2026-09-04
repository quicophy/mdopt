"""Tests for the detector-error-model decoder.

The ground truth throughout is exact enumeration of the syndrome coset: every
mechanism set consistent with the observed syndrome, weighted by its prior
mass, grouped by observable pattern. The decoder must reproduce those class
masses exactly at converged bond dimension, for syndromes sampled from real
stim circuits.
"""

import itertools

import numpy as np
import pymatching
import pytest
import stim

from mdopt.decoding.dem import (
    DemProblem,
    decode_dem,
    dem_to_problem,
    solve_representative,
)


def _nullspace_gf2(problem: DemProblem):
    """Basis of mechanism sets invisible to every detector."""
    rows, cols = problem.num_detectors, problem.num_mechanisms
    matrix = np.zeros((rows, cols), dtype=int)
    for det, row in enumerate(problem.detector_rows):
        matrix[det, row] = 1
    work, pivots, r = matrix.copy(), [], 0
    for col in range(cols):
        piv = next((i for i in range(r, rows) if work[i, col]), None)
        if piv is None:
            continue
        work[[r, piv]] = work[[piv, r]]
        for i in range(rows):
            if i != r and work[i, col]:
                work[i] = (work[i] + work[r]) % 2
        pivots.append(col)
        r += 1
        if r == rows:
            break
    basis = []
    for free in [c for c in range(cols) if c not in pivots]:
        vec = np.zeros(cols, dtype=int)
        vec[free] = 1
        for i, col in enumerate(pivots):
            if i < r:
                vec[col] = work[i, free]
        basis.append(vec)
    return basis


def _exact_class_masses(problem: DemProblem, syndrome: np.ndarray) -> np.ndarray:
    """Brute force over the syndrome coset, exact probability masses."""
    base = solve_representative(problem, syndrome)
    basis = _nullspace_gf2(problem)
    assert len(basis) <= 22, "coset too large to enumerate"
    probs = np.asarray(problem.probs)
    masses = np.zeros(2**problem.num_observables)
    for coeffs in itertools.product((0, 1), repeat=len(basis)):
        mech = base.copy()
        for c, vec in zip(coeffs, basis):
            if c:
                mech = (mech + vec) % 2
        weight = float(np.prod(np.where(mech == 1, probs, 1 - probs)))
        index = 0
        for j, row in enumerate(problem.observable_rows):
            if row and int(np.sum(mech[row]) % 2):
                index |= 1 << j
        masses[index] += weight
    return masses


def _repetition_problem(distance=3, rounds=2, p=0.02):
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
    )
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    return circuit, dem_to_problem(dem)


def test_dem_parsing_merges_duplicate_mechanisms():
    """Identical detector/observable sets must merge with p1(1-p2)+p2(1-p1)."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1
        error(0.2) D1 D0
        error(0.3) D1 L0
        detector D0
        detector D1
        """)
    problem = dem_to_problem(dem)
    assert problem.num_mechanisms == 2
    merged = [p for p, row in zip(problem.probs, range(2))]
    assert any(np.isclose(p, 0.1 * 0.8 + 0.2 * 0.9) for p in problem.probs)
    assert any(np.isclose(p, 0.3) for p in problem.probs)


def test_dem_rejects_decomposed_errors():
    """Maximum likelihood wants hyperedges, not matching-friendly splits."""
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 ^ D2 D3")
    with pytest.raises(ValueError, match="decompose"):
        dem_to_problem(dem)


def test_representative_solves_the_syndrome():
    """The GF(2) solve must reproduce any sampled syndrome exactly."""
    circuit, problem = _repetition_problem(distance=5, rounds=3)
    sampler = circuit.compile_detector_sampler(seed=7)
    detections, _ = sampler.sample(20, separate_observables=True)
    matrix = np.zeros((problem.num_detectors, problem.num_mechanisms), dtype=int)
    for det, row in enumerate(problem.detector_rows):
        matrix[det, row] = 1
    for syndrome in detections:
        rep = solve_representative(problem, syndrome.astype(int))
        assert np.array_equal(matrix @ rep % 2, syndrome.astype(int) % 2)


def test_dem_class_masses_match_brute_force():
    """The decoder's class masses equal exact coset enumeration.

    Syndromes come from sampling the actual repetition-code circuit, so this
    exercises the full path: stim DEM -> merged problem -> representative ->
    biased MPS -> constraints -> marginal readout.
    """
    circuit, problem = _repetition_problem(distance=3, rounds=2)
    sampler = circuit.compile_detector_sampler(seed=11)
    detections, _ = sampler.sample(8, separate_observables=True)
    seen = set()
    for syndrome in detections:
        key = tuple(syndrome.astype(int))
        if key in seen:
            continue
        seen.add(key)
        masses, flips = decode_dem(problem, syndrome.astype(int), chi_max=int(1e4))
        exact = _exact_class_masses(problem, syndrome.astype(int))
        got = masses / masses.sum()
        want = exact / exact.sum()
        assert np.allclose(got, want, atol=1e-9), (key, got, want)
        assert flips[0] == int(np.argmax(want) & 1)


def test_dem_class_masses_are_representative_independent():
    """The posterior is a function of the syndrome, not the representative."""
    circuit, problem = _repetition_problem(distance=3, rounds=2)
    sampler = circuit.compile_detector_sampler(seed=3)
    detections, _ = sampler.sample(4, separate_observables=True)
    rng = np.random.default_rng(0)
    basis = _nullspace_gf2(problem)
    syndrome = detections[np.argmax(detections.sum(axis=1))].astype(int)
    base = solve_representative(problem, syndrome)
    reference, _ = decode_dem(problem, syndrome, representative=base)
    reference = reference / reference.sum()
    for _ in range(3):
        shifted = base.copy()
        for vec in basis:
            if rng.random() < 0.5:
                shifted = (shifted + vec) % 2
        masses, _ = decode_dem(problem, syndrome, representative=shifted)
        assert np.allclose(masses / masses.sum(), reference, atol=1e-9)


def test_dem_decoder_is_at_least_as_accurate_as_pymatching():
    """On a matchable repetition-code circuit, MAP must not lose to MWPM.

    Per-shot ties can break either way, so the comparison is on totals with a
    one-shot slack rather than per shot.
    """
    circuit, problem = _repetition_problem(distance=5, rounds=5, p=0.08)
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=42)
    detections, observables = sampler.sample(60, separate_observables=True)

    ours = mwpm = 0
    for syndrome, actual in zip(detections, observables):
        _, flips = decode_dem(problem, syndrome.astype(int), chi_max=128)
        ours += int(np.array_equal(flips, actual.astype(int)))
        predicted = matcher.decode(syndrome)
        mwpm += int(np.array_equal(predicted % 2, actual.astype(int)))
    assert ours >= mwpm - 1, (ours, mwpm)


def test_two_observable_bit_ordering_matches_the_enumeration():
    """Bit j of the class index must be observable j, for a 2-observable DEM.

    The repetition-code tests all have one observable, so the multi-observable
    index convention (fixed by the .reverse() before readout) was otherwise
    unpinned. The model is deliberately asymmetric between L0 and L1.
    """
    dem = stim.DetectorErrorModel("""
        error(0.3) D0 L0
        error(0.05) D0 D1 L1
        error(0.1) D1 L0 L1
        error(0.02) D0
        error(0.15) D1
        """)
    problem = dem_to_problem(dem)
    for syndrome in ([0, 0], [1, 0], [0, 1], [1, 1]):
        masses, flips = decode_dem(problem, np.array(syndrome))
        exact = _exact_class_masses(problem, np.array(syndrome))
        assert np.allclose(
            masses / masses.sum(), exact / exact.sum(), atol=1e-9
        ), syndrome
        assert np.array_equal(
            flips,
            [(int(np.argmax(exact)) >> j) & 1 for j in range(2)],
        )


def test_surface_code_circuit_level_matches_brute_force():
    """First circuit-level surface-code validation: d=3, one round.

    23 mechanisms, 8 detectors, nullspace dimension 15 -- small enough to
    enumerate the full syndrome coset exactly. Every sampled syndrome's class
    masses must match, which exercises genuine hyperedges (undecomposed
    depolarising errors flip up to 6 detectors here).
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=1,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    problem = dem_to_problem(
        circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    )
    sampler = circuit.compile_detector_sampler(seed=5)
    detections, _ = sampler.sample(6, separate_observables=True)
    seen = set()
    for syndrome in detections:
        key = tuple(syndrome.astype(int))
        if key in seen:
            continue
        seen.add(key)
        masses, _ = decode_dem(problem, syndrome.astype(int), chi_max=int(1e4))
        exact = _exact_class_masses(problem, syndrome.astype(int))
        assert np.allclose(masses / masses.sum(), exact / exact.sum(), atol=1e-9), key


def test_ordering_leaves_the_posterior_invariant():
    """Reordering mechanisms relabels the chain; the class masses must not move."""
    from mdopt.decoding.dem import order_mechanisms

    circuit, problem = _repetition_problem(distance=3, rounds=2)
    sampler = circuit.compile_detector_sampler(seed=9)
    detections, _ = sampler.sample(5, separate_observables=True)
    reordered, _ = order_mechanisms(problem, "bandwidth")
    for syndrome in detections:
        a, fa = decode_dem(problem, syndrome.astype(int))
        b, fb = decode_dem(reordered, syndrome.astype(int))
        assert np.allclose(a / a.sum(), b / b.sum(), atol=1e-10)
        assert np.array_equal(fa, fb)


def test_natural_order_spans_scale_with_a_round_not_the_chain():
    """stim's time order is already near the time-locality floor.

    Measured finding, locked in as a scaling property rather than a wish: RCM
    bandwidth ordering makes circuit-DEM spans WORSE (the high-degree
    detectors form cliques that defeat it), while the native order keeps the
    mean span at roughly one round's worth of mechanisms regardless of how
    many rounds run. What grows the floor is mechanisms-per-round, not
    duration.
    """
    from mdopt.decoding.dem import constraint_spans

    spans = {}
    for rounds in (3, 9):
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=rounds,
            after_clifford_depolarization=0.005,
            before_measure_flip_probability=0.005,
            after_reset_flip_probability=0.005,
        )
        problem = dem_to_problem(
            circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
        )
        spans[rounds] = (
            constraint_spans(problem).mean(),
            problem.num_mechanisms / rounds,
        )
    for rounds, (mean_span, per_round) in spans.items():
        assert mean_span < 2.5 * per_round, (rounds, mean_span, per_round)
    # Tripling the number of rounds must not triple the span floor.
    assert spans[9][0] < 2 * spans[3][0], spans


def test_degree_one_detectors_pin_their_mechanism():
    """A one-mechanism detector fixes f_i = 0; masses must stay exact.

    The model mixes a pinned mechanism that also carries an observable with
    ordinary weight-2+ detectors, and every syndrome is checked against coset
    enumeration -- including syndromes where the pinned detector fired.
    """
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) D1 D2
        error(0.15) D1 D2 L1
        error(0.05) D2
        error(0.3) D1 L1
        """)
    problem = dem_to_problem(dem)
    for s0 in (0, 1):
        for s1 in (0, 1):
            for s2 in (0, 1):
                syndrome = np.array([s0, s1, s2])
                masses, flips = decode_dem(problem, syndrome)
                exact = _exact_class_masses(problem, syndrome)
                assert np.allclose(
                    masses / masses.sum(), exact / exact.sum(), atol=1e-9
                ), (s0, s1, s2)
                assert np.array_equal(
                    flips,
                    [(int(np.argmax(exact)) >> j) & 1 for j in range(2)],
                ), (s0, s1, s2)


def test_untouched_observable_is_deterministically_unflipped():
    """A declared observable no mechanism touches must carry flip 0.

    Initialising it as |+> would split its mass 50/50 over an impossible
    flip; the L1 slot here is declared (via L2's existence... the gap) but
    never referenced by any error.
    """
    dem = stim.DetectorErrorModel("""
        error(0.3) D0 L0
        error(0.1) D0 D1 L2
        error(0.2) D1
        """)
    problem = dem_to_problem(dem)
    assert problem.num_observables == 3
    assert problem.observable_rows[1] == []
    for syndrome in ([0, 0], [1, 0], [0, 1], [1, 1]):
        masses, flips = decode_dem(problem, np.array(syndrome))
        exact = _exact_class_masses(problem, np.array(syndrome))
        assert np.allclose(
            masses / masses.sum(), exact / exact.sum(), atol=1e-9
        ), syndrome
        assert flips[1] == 0, syndrome
        # No mass may sit on any class with the untouched bit set.
        norm = masses / masses.sum()
        assert norm[np.array([i for i in range(8) if (i >> 1) & 1])].max() < 1e-12


def test_dem_error_paths_raise_precisely():
    """The guard rails: inconsistent syndromes, dead detectors, bad
    representatives, chi collapse, and unknown ordering strategies."""
    from mdopt.decoding.dem import order_mechanisms

    problem = DemProblem(
        probs=[0.1, 0.2, 0.15],
        detector_rows=[[0, 1], [1, 2]],
        observable_rows=[[0]],
        num_detectors=2,
        num_observables=1,
    )

    # Two identical detector rows with conflicting syndrome bits: no solve.
    conflicted = DemProblem(
        probs=[0.1],
        detector_rows=[[0], [0]],
        observable_rows=[[]],
        num_detectors=2,
        num_observables=1,
    )
    with pytest.raises(ValueError):
        solve_representative(conflicted, np.array([1, 0]))

    # A declared detector no mechanism touches cannot fire.
    dead = DemProblem(
        probs=[0.1, 0.2],
        detector_rows=[[], [0, 1]],
        observable_rows=[[0]],
        num_detectors=2,
        num_observables=1,
    )
    with pytest.raises(ValueError, match="fired"):
        decode_dem(dead, np.array([1, 0]))

    # A representative that does not reproduce the syndrome is rejected.
    with pytest.raises(ValueError, match="representative"):
        decode_dem(problem, np.array([1, 0]), representative=np.array([0, 0, 0]))

    # chi_max=1 destroys the class-mass vector, which must be loud --
    # either as a collapse or as a materially negative mass, never as a
    # silently abs()'d posterior.
    with pytest.raises(ArithmeticError, match="collapsed|not converged"):
        decode_dem(problem, np.array([1, 0]), chi_max=1)

    # Ordering: natural is the identity, unknown strategies are rejected.
    natural, perm = order_mechanisms(problem, "natural")
    assert np.array_equal(perm, np.arange(3))
    assert natural.detector_rows == problem.detector_rows
    with pytest.raises(ValueError):
        order_mechanisms(problem, "alphabetical")


def test_empty_and_mechanism_free_models_decode_trivially():
    """No mechanisms means all mass on the no-flip class, never a crash.

    A declared detector with no error mechanisms previously reached
    create_custom_product_state("") and raised IndexError; a declared
    observable with no mechanisms must come out deterministically unflipped.
    """
    empty = dem_to_problem(stim.DetectorErrorModel("detector D0"))
    masses, flips = decode_dem(empty, np.array([0]))
    assert masses.tolist() == [1.0]
    assert flips.size == 0
    with pytest.raises(ValueError, match="fired"):
        decode_dem(empty, np.array([1]))

    obs_only = DemProblem(
        probs=[],
        detector_rows=[[]],
        observable_rows=[[], []],
        num_detectors=1,
        num_observables=2,
    )
    masses, flips = decode_dem(obs_only, np.array([0]))
    assert np.array_equal(masses, [1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(flips, [0, 0])
