"""Tests for the class-summed detector-error-model decoder.

The contraction is checked against dense linear algebra and against exact
enumeration of every solution inside an active set. The decoder is checked on
small circuits against exact maximum likelihood and, on the bivariate bicycle
[[72, 12, 6]] memory circuit, against BP-OSD on the same shots.
"""

import itertools

import numpy as np
import pytest
import stim
import sympy
from ldpc import BpOsdDecoder
from qldpc import circuits as qldpc_circuits
from qldpc.codes import BBCode

from mdopt.decoding.class_sum import (
    DEFAULT_BPOSD_OPTIONS,
    ClassSumDecoder,
    ParityChain,
    active_set_class_masses,
    basis_truncated_class_masses,
    multiplet_safe_rank,
    problem_matrices,
)
from mdopt.decoding.dem import DemProblem, decode_dem, dem_to_problem


def _random_problem(seed, num_detectors=8, num_mechanisms=14, num_observables=3):
    """A random model in which every mechanism flips two or three detectors."""
    rng = np.random.default_rng(seed)
    detector_rows = [[] for _ in range(num_detectors)]
    observable_rows = [[] for _ in range(num_observables)]
    for mechanism in range(num_mechanisms):
        for det in rng.choice(num_detectors, size=rng.integers(2, 4), replace=False):
            detector_rows[int(det)].append(mechanism)
        for obs in range(num_observables):
            if rng.random() < 0.4:
                observable_rows[obs].append(mechanism)
    return DemProblem(
        probs=list(rng.uniform(0.02, 0.3, size=num_mechanisms)),
        detector_rows=detector_rows,
        observable_rows=observable_rows,
        num_detectors=num_detectors,
        num_observables=num_observables,
    )


def _enumerate_active_set(problem, syndrome, active):
    """Exact class masses over every solution supported on ``active``."""
    check, observables = problem_matrices(problem)
    check = np.asarray(check.todense(), dtype=int)
    observables = np.asarray(observables.todense(), dtype=int)
    probs = np.asarray(problem.probs)
    masses = np.zeros(2**problem.num_observables)
    powers = 1 << np.arange(problem.num_observables)
    for bits in itertools.product((0, 1), repeat=len(active)):
        mechanisms = np.zeros(problem.num_mechanisms, dtype=int)
        mechanisms[list(active)] = bits
        if ((check @ mechanisms) % 2 != syndrome).any():
            continue
        weight = float(np.prod(np.where(mechanisms == 1, probs, 1 - probs)))
        masses[int(((observables @ mechanisms) % 2) @ powers)] += weight
    return masses


def _sample_from_model(problem, shots, seed):
    """Sample mechanisms from the priors with numpy (stable across versions)."""
    check, observables = problem_matrices(problem)
    rng = np.random.default_rng(seed)
    errors = (rng.random((shots, problem.num_mechanisms)) < problem.probs).astype(
        np.uint8
    )
    syndromes = (check @ errors.T).T % 2
    flips = (observables @ errors.T).T % 2
    return syndromes.astype(np.uint8), flips.astype(np.uint8)


def _bb72_problem(rounds=1, p=0.01):
    x, y = sympy.symbols("x y")
    code = BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    circuit = qldpc_circuits.get_memory_experiment(
        code,
        "Z",
        num_rounds=rounds,
        noise_model=qldpc_circuits.DepolarizingNoiseModel(p),
    )
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    return dem_to_problem(dem)


def test_rank_is_plain_truncation_without_degeneracy():
    values = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    assert multiplet_safe_rank(values, 3) == 3
    assert multiplet_safe_rank(values, 10) == 5
    assert multiplet_safe_rank(np.array([1.0, 0.5, 0.0, 0.0]), 4) == 2
    assert multiplet_safe_rank(np.zeros(3), 2) == 0
    assert multiplet_safe_rank(np.zeros(0), 2) == 0


def test_rank_never_splits_a_multiplet():
    """A cut inside a degenerate block moves to the block's boundary."""
    values = np.array([1.0, 0.5, 0.5, 0.5, 0.1])
    assert multiplet_safe_rank(values, 2) == 4
    assert multiplet_safe_rank(values, 3) == 4
    assert multiplet_safe_rank(values, 4) == 4
    # A block too large to keep whole is dropped whole instead ...
    values = np.concatenate([[1.0, 0.9], np.full(40, 0.5), [0.1]])
    assert multiplet_safe_rank(values, 8) == 2
    # ... unless it sits at the top of the spectrum.
    values = np.concatenate([np.full(40, 0.5), [0.1]])
    assert multiplet_safe_rank(values, 8) == 40


def test_rank_at_the_edges_of_its_rule():
    """The tolerance, the allowance and the scale, each at its boundary."""
    tolerance = 1e-7
    # A relative difference of exactly the tolerance is not a degeneracy.
    assert multiplet_safe_rank(np.array([1.0, 0.5, 0.5 * (1 - tolerance), 0.1]), 2) == 2
    inside = np.array([1.0, 0.5, 0.5 * (1 - 0.5 * tolerance), 0.1])
    assert multiplet_safe_rank(inside, 2) == 3
    # At chi_max = 8 the cut may move up by four values, not by five.
    head = np.linspace(2.0, 1.5, 6)
    assert multiplet_safe_rank(np.concatenate([head, np.full(6, 1.0), [0.1]]), 8) == 12
    assert multiplet_safe_rank(np.concatenate([head, np.full(7, 1.0), [0.1]]), 8) == 6
    # A whole spectrum that is one multiplet is kept, whatever chi_max says.
    assert multiplet_safe_rank(np.ones(40), 4) == 40
    # The rule depends on ratios only.
    for scale in (1e-200, 1.0, 1e200):
        assert multiplet_safe_rank(scale * np.array([1.0, 1.0, 0.5, 0.25]), 1) == 2
        assert multiplet_safe_rank(scale * np.array([1.0, 0.5, 0.25]), 1) == 1
    # The window is anchored at the cut: a run that drifts by 0.6 tolerance
    # per step is one multiplet only within one tolerance of the cut.
    drift = (1 - 0.6 * tolerance) ** np.arange(60)
    assert multiplet_safe_rank(drift, 8) == 9


def test_truncation_reports_the_lost_fraction_at_any_scale():
    """The lost fraction is a ratio, so a chain whose singular values are too
    small to square reports the same loss as the same chain at unit scale."""
    losses = []
    for scale in (1.0, 1e-200):
        chain = ParityChain([np.array([0.8, 0.6])] * 3)
        chain.apply_parity([0, 2], 4, (1.0, 0.5))
        chain.move_center(2)
        chain.tensors[2] = chain.tensors[2] * scale
        chain.lost = 0.0
        chain._truncate_bond(2, 1)
        losses.append(chain.lost)
    assert 0.0 < losses[0] < 1.0
    assert np.isclose(losses[1], losses[0], rtol=1e-12)
    empty = ParityChain([np.array([0.8, 0.6])] * 2)
    empty.move_center(1)
    empty.tensors[1] = np.zeros_like(empty.tensors[1])
    with pytest.raises(ArithmeticError, match="vanished"):
        empty._truncate_bond(1, 4)


def _dense(chain):
    state = np.ones((1, 1))
    for tensor in chain.tensors:
        state = np.einsum("ia,apb->ipb", state, tensor).reshape(-1, tensor.shape[2])
    return state[:, 0] * np.exp(chain.log_norm)


@pytest.mark.parametrize("seed", range(5))
def test_parity_chain_matches_dense_algebra(seed):
    """Exact parity factors at unbounded bond dimension reproduce the vector."""
    rng = np.random.default_rng(seed)
    num_sites = 7
    vectors = rng.uniform(0.2, 1.0, size=(num_sites, 2))
    chain = ParityChain(list(vectors))
    reference = np.ones(1)
    for vector in vectors:
        reference = np.kron(reference, vector)
    bits = np.array(list(itertools.product((0, 1), repeat=num_sites)))
    assert np.allclose(_dense(chain), reference)
    for _ in range(8):
        size = int(rng.integers(1, num_sites + 1))
        sites = rng.choice(num_sites, size=size, replace=False)
        weights = rng.uniform(0.1, 1.0, size=2)
        if rng.random() < 0.3:
            weights = np.array([1.0, 0.0])
        chain.apply_parity(sites, 2**num_sites, weights)
        reference = reference * weights[bits[:, sites].sum(axis=1) % 2]
        assert np.allclose(_dense(chain), reference)
    assert chain.lost < 1e-20
    chain.project_bit(3, 1)
    assert np.allclose(_dense(chain), reference * bits[:, 3])


def test_parity_chain_records_the_truncated_weight():
    rng = np.random.default_rng(11)
    vectors = list(rng.uniform(0.2, 1.0, size=(8, 2)))
    exact, truncated = ParityChain(vectors), ParityChain(vectors)
    for sites in ([0, 3, 7], [1, 2, 5, 6], [0, 4, 6], [2, 3, 7]):
        weights = rng.uniform(0.2, 1.0, size=2)
        exact.apply_parity(sites, 256, weights)
        truncated.apply_parity(sites, 2, weights)
    assert exact.lost < 1e-20
    assert truncated.lost > 0.0
    assert truncated.max_bond() < exact.max_bond()


def test_parity_chain_holds_a_product_whose_norm_is_outside_float64():
    """Three thousand sites of (1, 1) have a norm of 2 ** 1500; its logarithm
    is an ordinary number. Tiny sites must not read as an annihilated state."""
    chain = ParityChain([np.ones(2)] * 3000)
    assert np.isclose(chain.log_norm, 1500 * np.log(2.0))
    tiny = ParityChain([np.array([1e-200, 1e-200])] * 40)
    assert np.isclose(tiny.log_norm, 40 * np.log(np.sqrt(2.0) * 1e-200))
    # Entries near the top of float64: the norm is infinite, its log is not.
    huge = ParityChain([np.array([1.5e308, 1.5e308])])
    assert np.isclose(huge.log_norm, np.log(1.5e308) + 0.5 * np.log(2.0))
    assert np.allclose(huge.tensors[0].ravel(), np.sqrt(0.5))
    with pytest.raises(ArithmeticError):
        ParityChain([np.ones(2), np.zeros(2)])


def test_parity_chain_normalizes_amplitudes_too_small_to_square():
    chain = ParityChain([np.ones(2)])
    chain.apply_parity([0], 4, (1e-200, 1e-200))
    assert np.isclose(chain.log_norm, np.log(np.sqrt(2.0) * 1e-200))


def test_an_active_set_of_probability_zero_is_rejected():
    """An inactive mechanism that certainly fires, or a pivot that must take
    a value of probability zero: there is nothing to normalize."""
    certain = DemProblem(
        probs=[0.2, 1.0],
        detector_rows=[[0]],
        observable_rows=[[0]],
        num_detectors=1,
        num_observables=1,
    )
    with pytest.raises(ArithmeticError, match="positive probability"):
        active_set_class_masses(certain, np.ones(1, dtype=np.uint8), [0], chi_max=4)
    impossible = DemProblem(
        probs=[0.0],
        detector_rows=[[0]],
        observable_rows=[[]],
        num_detectors=1,
        num_observables=1,
    )
    with pytest.raises(ArithmeticError, match="positive probability"):
        active_set_class_masses(impossible, np.ones(1, dtype=np.uint8), [0], chi_max=4)
    masses, info = active_set_class_masses(
        impossible, np.zeros(1, dtype=np.uint8), [0], chi_max=4
    )
    assert np.allclose(masses, [1.0, 0.0]) and abs(info["log_mass"]) < 1e-12


def test_parity_chain_rejects_bad_input():
    with pytest.raises(ValueError, match="at least one site"):
        ParityChain([])
    chain = ParityChain([np.array([1.0, 1.0])] * 3)
    with pytest.raises(ValueError, match="at least one site"):
        chain.apply_parity([], 4)
    with pytest.raises(ValueError, match="outside the chain"):
        chain.apply_parity([1, 3], 4)
    with pytest.raises(ValueError, match="more than once"):
        chain.apply_parity([0, 1, 1], 4)
    with pytest.raises(ValueError, match="bond dimension"):
        chain.apply_parity([0, 1], 0)
    with pytest.raises(ValueError, match="two weights"):
        chain.apply_parity([0, 1], 4, (1.0, 0.5, 0.25))
    chain.apply_parity([0, 1], 4)
    with pytest.raises(ArithmeticError, match="annihilated"):
        chain.apply_parity([0, 1], 4, (0.0, 1.0))


@pytest.mark.parametrize("seed", range(6))
def test_class_masses_match_enumeration_on_the_full_model(seed):
    """With every mechanism active the result is the exact posterior."""
    problem = _random_problem(seed)
    syndromes, _ = _sample_from_model(problem, 3, seed + 100)
    for syndrome in syndromes:
        active = list(range(problem.num_mechanisms))
        exact = _enumerate_active_set(problem, syndrome, active)
        masses, info = active_set_class_masses(problem, syndrome, active, chi_max=256)
        assert info["lost"] < 1e-20
        assert np.allclose(masses, exact / exact.sum(), atol=1e-12)
        assert np.isclose(info["log_mass"], np.log(exact.sum()))
        assert info["rank"] + info["free"] == problem.num_mechanisms


@pytest.mark.parametrize("seed", range(6))
def test_class_masses_match_enumeration_on_a_restricted_active_set(seed):
    """Mechanisms outside the active set are pinned off, in any pivot order."""
    problem = _random_problem(seed, num_mechanisms=16)
    rng = np.random.default_rng(seed)
    check, _ = problem_matrices(problem)
    error = np.zeros(problem.num_mechanisms, dtype=np.uint8)
    error[rng.choice(problem.num_mechanisms, size=3, replace=False)] = 1
    syndrome = (check @ error) % 2
    extra = rng.choice(problem.num_mechanisms, size=7, replace=False)
    active = list(dict.fromkeys([int(i) for i in np.flatnonzero(error)] + list(extra)))
    exact = _enumerate_active_set(problem, syndrome, active)
    for order in (active, active[::-1], sorted(active)):
        masses, info = active_set_class_masses(problem, syndrome, order, chi_max=256)
        assert np.allclose(masses, exact / exact.sum(), atol=1e-12)
        assert np.isclose(info["log_mass"], np.log(exact.sum()))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("stages", [2, 3, 5])
def test_tempered_class_masses_match_enumeration(seed, stages):
    """Without truncation the tempered contraction is the same exact sum."""
    problem = _random_problem(seed, num_mechanisms=16)
    rng = np.random.default_rng(seed)
    check, _ = problem_matrices(problem)
    error = np.zeros(problem.num_mechanisms, dtype=np.uint8)
    error[rng.choice(problem.num_mechanisms, size=3, replace=False)] = 1
    syndrome = (check @ error) % 2
    extra = rng.choice(problem.num_mechanisms, size=7, replace=False)
    restricted = list(
        dict.fromkeys([int(i) for i in np.flatnonzero(error)] + list(extra))
    )
    for active in (restricted, list(range(problem.num_mechanisms))):
        exact = _enumerate_active_set(problem, syndrome, active)
        masses, info = active_set_class_masses(
            problem, syndrome, active, chi_max=1024, stages=stages
        )
        assert info["lost"] < 1e-20
        assert np.allclose(masses, exact / exact.sum(), atol=1e-12)
        assert np.isclose(info["log_mass"], np.log(exact.sum()))


def test_tempered_contraction_survives_thousands_of_free_mechanisms():
    """The uniform start must not overflow: 2 ** 1100 is beyond float64."""
    size, prob = 2200, 0.01
    problem = DemProblem(
        probs=[prob] * size,
        detector_rows=[list(range(size))],
        observable_rows=[[0]],
        num_detectors=1,
        num_observables=1,
    )
    masses, info = active_set_class_masses(
        problem, np.zeros(1, dtype=np.uint8), list(range(size)), chi_max=4, stages=2
    )
    assert info["free"] == size - 1
    assert np.isfinite(masses).all() and np.isclose(masses.sum(), 1.0)
    # Even parity over all mechanisms: (1 + (1 - 2p) ** n) / 2.
    assert np.isclose(info["log_mass"], np.log((1 + (1 - 2 * prob) ** size) / 2))


def _two_route_problem(free: int, prob: float):
    """Two equally likely solutions in opposite classes, far apart in weight order.

    Free mechanisms ``f_i``; detector ``g_i`` holds a pivot ``G_i`` and
    ``f_i`` and fired; detector ``n_ij`` holds a pivot ``N_ij``, ``f_i`` and
    ``f_j`` and did not fire. The observable is ``f_0``. "Every ``G_i``" and
    "every ``f_i``" both have ``free`` firing mechanisms, so the two classes
    carry exactly the same mass by the symmetry that exchanges them, but on
    the way from one to the other ``k (free - k)`` of the ``N_ij`` fire.
    """
    pairs = list(itertools.combinations(range(free), 2))
    first_free = free + len(pairs)
    rows = [[i, first_free + i] for i in range(free)]
    rows += [
        [free + k, first_free + i, first_free + j] for k, (i, j) in enumerate(pairs)
    ]
    problem = DemProblem(
        probs=[prob] * (first_free + free),
        detector_rows=rows,
        observable_rows=[[first_free]],
        num_detectors=len(rows),
        num_observables=1,
    )
    syndrome = np.array([1] * free + [0] * len(pairs), dtype=np.uint8)
    return problem, syndrome, list(range(first_free + free))


def test_tempering_keeps_a_component_the_one_pass_loses_to_the_zero_cutoff():
    """Tempering keeps the second route, which a one-pass contraction can lose.

    With ten free mechanisms at probability 1e-3 the second solution sits
    below ``ZERO_CUTOFF`` of the first in the state a single pass compresses,
    whatever the bond dimension, and the pivot factors applied afterwards
    bring it back to the same weight: the single pass returns one class with
    almost no discarded weight. The exact masses are one half each.
    """
    problem, syndrome, active = _two_route_problem(10, 1e-3)
    tempered, _ = active_set_class_masses(
        problem, syndrome, active, chi_max=10**6, stages=2
    )
    assert np.allclose(tempered, [0.5, 0.5], atol=1e-6)
    tempered, _ = active_set_class_masses(
        problem, syndrome, active, chi_max=16, stages=2
    )
    assert np.allclose(tempered, [0.5, 0.5], atol=1e-6)


def test_tempering_rejects_bad_input():
    problem = _random_problem(0)
    syndrome = np.zeros(problem.num_detectors, dtype=np.uint8)
    with pytest.raises(ValueError, match="tempering stages"):
        active_set_class_masses(problem, syndrome, [0, 1], stages=0)
    with pytest.raises(ValueError, match="tempering stages"):
        ClassSumDecoder(problem, temper_stages=0)
    with pytest.raises(ValueError, match="tempering threshold"):
        ClassSumDecoder(problem, temper_threshold=-1.0)


def _small_surface_problem(p):
    """Distance 3, one round: 23 mechanisms with genuine hyperedges."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=1,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    return dem_to_problem(dem)


def test_class_masses_agree_with_the_full_chain_decoder():
    """Same posterior as decode_dem on a stim surface-code circuit."""
    problem = _small_surface_problem(0.01)
    syndromes, _ = _sample_from_model(problem, 12, seed=5)
    for syndrome in syndromes:
        reference, _ = decode_dem(problem, syndrome.astype(int), chi_max=4096)
        masses, info = active_set_class_masses(
            problem, syndrome, range(problem.num_mechanisms), chi_max=4096
        )
        assert info["lost"] < 1e-20
        assert np.allclose(masses, reference / reference.sum(), atol=1e-9)


def test_a_mechanism_listed_twice_in_a_row_cancels():
    """The matrices are over GF(2): two entries of one mechanism are none."""
    problem = DemProblem(
        probs=[0.1, 0.2],
        detector_rows=[[0, 0, 1], [1]],
        observable_rows=[[1, 1]],
        num_detectors=2,
        num_observables=1,
    )
    check, observables = problem_matrices(problem)
    assert np.array_equal(check.toarray(), [[0, 1], [0, 1]])
    assert np.array_equal(observables.toarray(), [[0, 0]])
    assert check.nnz == 2 and observables.nnz == 0


@pytest.mark.parametrize("seed", range(4))
def test_class_masses_under_real_truncation(seed):
    """End to end at bond dimensions that truncate: the masses stay a
    probability vector, the mass estimate and the masses are off by an amount
    that the reported loss controls, and everything converges to the
    enumeration."""
    problem = _random_problem(
        seed, num_detectors=6, num_mechanisms=17, num_observables=3
    )
    syndromes, _ = _sample_from_model(problem, 1, seed + 100)
    active = list(range(problem.num_mechanisms))
    exact = _enumerate_active_set(problem, syndromes[0], active)
    target, log_exact = exact / exact.sum(), np.log(exact.sum())
    truncated = 0
    for chi_max in (2, 4, 8, 2048):
        masses, info = active_set_class_masses(
            problem, syndromes[0], active, chi_max=chi_max
        )
        assert np.all(masses >= 0.0) and np.isclose(masses.sum(), 1.0)
        distance = 0.5 * np.abs(masses - target).sum()
        log_error = abs(info["log_mass"] - log_exact)
        # Measured relations, with a wide margin: over forty models the log
        # error was at most 1.75 times the loss and the distance at most
        # 0.66 times its root.
        assert log_error <= 3.0 * info["lost"] + 1e-9
        assert distance <= 2.0 * np.sqrt(info["lost"]) + 1e-9
        if chi_max == 2048:
            assert info["lost"] < 1e-20 and distance < 1e-10 and log_error < 1e-10
        elif info["lost"] > 1e-3:
            truncated += 1
            assert info["bond"] <= chi_max + max(4, chi_max // 4)
    assert truncated >= 2


def test_a_truncated_mass_estimate_can_exceed_the_exact_mass():
    """``log_mass`` is an estimate, not a lower bound: a soft factor applied
    after a truncation can amplify what the truncation changed."""
    problem = DemProblem(
        probs=[0.1, 0.1, 0.5, 0.5],
        detector_rows=[[0, 2, 3], [1, 2, 3]],
        observable_rows=[[2]],
        num_detectors=2,
        num_observables=1,
    )
    syndrome = np.array([0, 1], dtype=np.uint8)
    active = [0, 1, 2, 3]
    exact = _enumerate_active_set(problem, syndrome, active).sum()
    assert np.isclose(exact, 0.09)
    _, info = active_set_class_masses(problem, syndrome, active, chi_max=1)
    assert info["lost"] > 0.1
    assert np.exp(info["log_mass"]) > 1.5 * exact
    _, info = active_set_class_masses(problem, syndrome, active, chi_max=2)
    assert np.isclose(np.exp(info["log_mass"]), exact)


def test_certain_mechanisms_are_decided_before_the_contraction():
    """A mechanism of probability zero or one is a hard constraint. Left in
    the chain, a truncation could remove every valid solution and raise on an
    instance of positive probability; decided beforehand, the result is exact
    at a large bond dimension and never raises at a small one."""
    checked = impossible = 0
    for seed in range(60):
        base = _random_problem(
            seed, num_detectors=5, num_mechanisms=11, num_observables=2
        )
        rng = np.random.default_rng(seed)
        probs = np.array(base.probs)
        certain = rng.choice(11, size=int(rng.integers(1, 4)), replace=False)
        probs[certain] = rng.choice([0.0, 1.0], size=len(certain))
        problem = DemProblem(
            probs=list(probs),
            detector_rows=base.detector_rows,
            observable_rows=base.observable_rows,
            num_detectors=5,
            num_observables=2,
        )
        syndromes, _ = _sample_from_model(problem, 1, seed + 7)
        active = list(range(11))
        exact = _enumerate_active_set(problem, syndromes[0], active)
        if exact.sum() == 0.0:
            impossible += 1
            with pytest.raises(ArithmeticError, match="positive probability"):
                active_set_class_masses(problem, syndromes[0], active, chi_max=4)
            continue
        checked += 1
        masses, info = active_set_class_masses(
            problem, syndromes[0], active, chi_max=2048
        )
        assert np.allclose(masses, exact / exact.sum(), atol=1e-12)
        assert np.isclose(info["log_mass"], np.log(exact.sum()))
        assert info["free"] + info["rank"] == 11 - len(certain)
        rough, _ = active_set_class_masses(problem, syndromes[0], active, chi_max=1)
        assert np.all(rough >= 0.0) and np.isclose(rough.sum(), 1.0)
    assert checked >= 40


def test_active_set_without_a_solution_raises():
    problem = DemProblem(
        probs=[0.1, 0.1, 0.1],
        detector_rows=[[0, 1], [1, 2]],
        observable_rows=[[0]],
        num_detectors=2,
        num_observables=1,
    )
    with pytest.raises(ValueError, match="no solution inside the active set"):
        active_set_class_masses(problem, np.array([1, 1]), [0])
    with pytest.raises(ValueError, match="outside the model"):
        active_set_class_masses(problem, np.array([1, 1]), [0, 5])
    with pytest.raises(ValueError, match="bond dimension"):
        active_set_class_masses(problem, np.array([1, 1]), [1], chi_max=0)
    masses, info = active_set_class_masses(problem, np.array([1, 1]), [1])
    assert np.allclose(masses, [1.0, 0.0])
    assert info["free"] == 0 and info["rank"] == 1
    assert np.isclose(info["log_mass"], np.log(0.1 * 0.9 * 0.9))


def test_model_without_observables_has_one_class():
    problem = DemProblem(
        probs=[0.1, 0.2],
        detector_rows=[[0, 1]],
        observable_rows=[],
        num_detectors=1,
        num_observables=0,
    )
    masses, info = active_set_class_masses(problem, np.array([1]), [0, 1])
    assert masses.shape == (1,) and np.isclose(masses[0], 1.0)
    assert np.isclose(info["log_mass"], np.log(0.1 * 0.8 + 0.9 * 0.2))
    masses, info = active_set_class_masses(problem, np.array([1]), [0])
    assert np.isclose(info["log_mass"], np.log(0.1 * 0.8))


def test_decoder_rejects_bad_input():
    problem = _random_problem(0)
    with pytest.raises(ValueError, match="bond dimension"):
        ClassSumDecoder(problem, chi_max=0)
    degenerate = DemProblem(
        probs=[0.1, 1.0],
        detector_rows=[[0, 1]],
        observable_rows=[[0]],
        num_detectors=1,
        num_observables=1,
    )
    with pytest.raises(ValueError, match="at most 1/2"):
        ClassSumDecoder(degenerate)
    likely = DemProblem(
        probs=[0.9, 0.8],
        detector_rows=[[0, 1]],
        observable_rows=[[0]],
        num_detectors=1,
        num_observables=1,
    )
    with pytest.raises(ValueError, match="at most 1/2"):
        ClassSumDecoder(likely)
    empty = DemProblem(
        probs=[],
        detector_rows=[[]],
        observable_rows=[[]],
        num_detectors=1,
        num_observables=1,
    )
    with pytest.raises(ValueError, match="no mechanisms"):
        ClassSumDecoder(empty)
    with pytest.raises(ValueError, match="temper_chi_max"):
        ClassSumDecoder(problem, temper_chi_max=0)
    with pytest.raises(ValueError, match="extra centers"):
        ClassSumDecoder(problem, extra_centers=-1)
    with pytest.raises(ValueError):
        ClassSumDecoder(problem).decode(np.zeros(problem.num_detectors + 1))


def test_decoder_returns_no_flips_for_the_empty_syndrome():
    problem = _random_problem(1)
    flips, info = ClassSumDecoder(problem).decode(np.zeros(problem.num_detectors))
    assert flips.shape == (problem.num_observables,) and not flips.any()
    assert info["predicted_class"] == 0 and info["active"] == 0


def test_a_tie_goes_to_the_bposd_class(monkeypatch):
    """The BP-OSD class wins a tie wherever the sort happens to rank it."""
    problem = _random_problem(4, num_observables=2)
    decoder = ClassSumDecoder(problem)
    monkeypatch.setattr(
        decoder,
        "_class_masses",
        lambda syndrome, active, stages: (np.full(4, 0.25), {"lost": 0.0, "free": 1}),
    )
    syndromes, _ = _sample_from_model(problem, 60, seed=7)
    ranked_elsewhere = 0
    for syndrome in syndromes[syndromes.any(axis=1)]:
        flips, info = decoder.decode(syndrome)
        if info["candidates"] == 0:
            continue
        assert info["predicted_class"] == info["bposd_class"]
        assert int(flips @ (1 << np.arange(2))) == info["bposd_class"]
        # argsort puts class 3 first and class 2 second among equal masses.
        ranked_elsewhere += info["bposd_class"] in (0, 1)
    assert ranked_elsewhere > 0


def test_decoder_matches_maximum_likelihood_on_a_small_surface_code():
    """On a small circuit the candidates' active set holds the posterior."""
    problem = _small_surface_problem(0.02)
    decoder = ClassSumDecoder(problem, chi_max=16)
    syndromes, _ = _sample_from_model(problem, 80, seed=3)
    assert syndromes.any(axis=1).sum() > 20
    agree = 0
    for syndrome in syndromes:
        flips, info = decoder.decode(syndrome)
        _, reference = decode_dem(problem, syndrome.astype(int), chi_max=4096)
        agree += int(np.array_equal(flips, reference))
        assert info["lost"] < 1e-6
        assert int(flips @ (1 << np.arange(flips.size))) == info["predicted_class"]
    assert agree >= 78


def test_a_tempered_pass_that_fails_keeps_the_one_pass_verdict(monkeypatch):
    """The tempered contraction can underflow. The decode then stands on the
    first pass: same verdict, margin and ``lost``, and ``tempered`` is off."""
    problem = _random_problem(4, num_observables=2)
    decoder = ClassSumDecoder(problem, temper_stages=4)
    calls = []

    def truncated_then_failing(syndrome, active, stages):
        calls.append(stages)
        if stages > 1:
            raise ArithmeticError("underflow")
        return np.array([0.1, 0.2, 0.6, 0.1]), {"lost": 0.5, "free": 1}

    monkeypatch.setattr(decoder, "_class_masses", truncated_then_failing)
    syndromes, _ = _sample_from_model(problem, 60, seed=7)
    decoded = 0
    for syndrome in syndromes[syndromes.any(axis=1)]:
        calls.clear()
        flips, info = decoder.decode(syndrome)
        if info["candidates"] == 0:
            continue
        decoded += 1
        assert calls == [1, 4]
        assert not info["tempered"]
        assert info["predicted_class"] == 2
        assert int(flips @ (1 << np.arange(2))) == 2
        assert np.isclose(info["margin"], 0.4) and info["lost"] == 0.5
    assert decoded > 0


def test_decoder_tempers_only_when_the_first_pass_truncated():
    """The tempered pass replaces the one-pass verdict above the threshold."""
    problem = _small_surface_problem(0.02)
    syndromes, _ = _sample_from_model(problem, 40, seed=3)
    syndromes = syndromes[syndromes.any(axis=1)]
    assert len(syndromes) > 10
    default = ClassSumDecoder(problem, chi_max=16)
    always = ClassSumDecoder(problem, chi_max=16, temper_threshold=0.0)
    never = ClassSumDecoder(problem, chi_max=1, temper_stages=1)
    eager = ClassSumDecoder(problem, chi_max=1)
    tempered_at_small_chi = 0
    for syndrome in syndromes:
        flips, info = default.decode(syndrome)
        assert info["lost"] < 1e-8 and not info["tempered"]
        forced_flips, forced = always.decode(syndrome)
        assert forced["tempered"]
        assert np.array_equal(forced_flips, flips)
        assert np.isclose(forced["margin"], info["margin"], atol=1e-9)
        _, plain = never.decode(syndrome)
        assert not plain["tempered"]
        _, second = eager.decode(syndrome)
        assert second["tempered"] == (plain["lost"] >= eager.temper_threshold)
        tempered_at_small_chi += second["tempered"]
    assert tempered_at_small_chi > 0


def test_widened_search_extends_the_candidates():
    """Every extra center adds classes and never loses a better candidate."""
    problem = _random_problem(2, num_detectors=10, num_mechanisms=40, num_observables=6)
    syndromes, _ = _sample_from_model(problem, 40, seed=5)
    syndromes = syndromes[syndromes.any(axis=1)]
    narrow = ClassSumDecoder(problem, extra_centers=0)
    wide = ClassSumDecoder(problem, extra_centers=4)
    grew = 0
    for syndrome in syndromes:
        syndrome = syndrome.astype(np.uint8)
        seed = np.asarray(narrow._bposd.decode(syndrome), dtype=np.uint8)
        if not narrow._satisfies(seed, syndrome):
            continue
        found, tried, _ = narrow._search(syndrome, seed)
        assert narrow.candidates(syndrome, seed).keys() == found.keys()
        assert narrow.widen(syndrome, found) == found
        # Centers that were already used add nothing and do not use the budget.
        only_tried = {label: found[label] for label in tried}
        assert wide.widen(syndrome, only_tried, tried) == only_tried
        repeated = ClassSumDecoder(problem, extra_centers=len(tried))
        again = repeated.widen(syndrome, found)
        assert again.keys() == found.keys()
        wider = wide.widen(syndrome, found, tried)
        assert set(found) <= set(wider)
        for label, (likelihood, solution) in wider.items():
            assert wide._satisfies(solution, syndrome)
            assert wide.class_of(solution) == label
            assert np.isclose(likelihood, wide._llr @ solution)
            if label in found:
                assert likelihood >= found[label][0] - 1e-12
        grew += len(wider) > len(found)
    assert grew > 0


def test_decoder_widens_only_when_the_first_pass_truncated():
    """Hard-syndrome options leave the easy syndromes untouched."""
    problem = _small_surface_problem(0.02)
    syndromes, _ = _sample_from_model(problem, 40, seed=3)
    syndromes = syndromes[syndromes.any(axis=1)]
    default = ClassSumDecoder(problem, chi_max=16)
    options = ClassSumDecoder(problem, chi_max=16, extra_centers=4, temper_chi_max=32)
    assert default.temper_chi_max == 16 and options.temper_chi_max == 32
    forced = ClassSumDecoder(problem, chi_max=1, extra_centers=4, temper_chi_max=64)
    stock = ClassSumDecoder(problem, chi_max=1)
    # One pass and nothing else: its ``lost`` says whether the first pass
    # of the decoders above truncated.
    one_pass = ClassSumDecoder(problem, chi_max=1, temper_stages=1)
    widened = larger = tempered = 0
    for syndrome in syndromes:
        flips, info = default.decode(syndrome)
        same_flips, same = options.decode(syndrome)
        assert not same["widened"] and not same["tempered"]
        assert np.array_equal(flips, same_flips)
        assert same["candidates"] == info["candidates"]
        _, narrow = stock.decode(syndrome)
        hard_flips, hard = forced.decode(syndrome)
        # Only a truncated first pass widens, and only if a class is left
        # that was not a center yet.
        _, first = one_pass.decode(syndrome)
        assert first["lost"] >= forced.temper_threshold or not hard["widened"]
        assert hard["candidates"] >= narrow["candidates"]
        if hard["tempered"]:
            # The tempered pass at chi 64 is exact on this small model.
            assert hard["lost"] < 1e-8
            _, reference = decode_dem(problem, syndrome.astype(int), chi_max=4096)
            larger += int(np.array_equal(hard_flips, reference))
        widened += hard["widened"]
        tempered += hard["tempered"]
    assert widened > 0 and tempered > 5 and larger >= tempered - 2


@pytest.mark.parametrize("seed", range(6))
def test_basis_truncated_masses_match_enumeration(seed):
    """Without pruning the computational-basis contraction is exact."""
    problem = _random_problem(seed)
    syndromes, _ = _sample_from_model(problem, 3, seed + 100)
    for syndrome in syndromes:
        active = list(range(problem.num_mechanisms))
        exact = _enumerate_active_set(problem, syndrome, active)
        masses, info = basis_truncated_class_masses(
            problem, syndrome, active, max_states=4096
        )
        assert info["pruned"] == 0.0
        assert np.allclose(masses, exact / exact.sum(), atol=1e-12)
        assert np.isclose(info["log_mass"], np.log(exact.sum()))
        assert info["rank"] + info["free"] == problem.num_mechanisms


@pytest.mark.parametrize("seed", range(6))
def test_basis_truncated_masses_on_a_restricted_active_set(seed):
    """Pinned mechanisms and the pivot order are handled as in the SVD sum."""
    problem = _random_problem(seed, num_mechanisms=16)
    rng = np.random.default_rng(seed)
    check, _ = problem_matrices(problem)
    error = np.zeros(problem.num_mechanisms, dtype=np.uint8)
    error[rng.choice(problem.num_mechanisms, size=3, replace=False)] = 1
    syndrome = (check @ error) % 2
    extra = rng.choice(problem.num_mechanisms, size=7, replace=False)
    active = list(dict.fromkeys([int(i) for i in np.flatnonzero(error)] + list(extra)))
    exact = _enumerate_active_set(problem, syndrome, active)
    for order in (active, active[::-1], sorted(active)):
        masses, info = basis_truncated_class_masses(problem, syndrome, order)
        reference, svd = active_set_class_masses(problem, syndrome, order, chi_max=256)
        assert np.allclose(masses, exact / exact.sum(), atol=1e-12)
        assert np.allclose(masses, reference, atol=1e-10)
        assert np.isclose(info["log_mass"], np.log(exact.sum()))
        assert np.isclose(info["log_mass"], svd["log_mass"])


def test_basis_truncation_prunes_and_reports_it():
    """A small state budget prunes, says so, and still returns probabilities."""
    problem = _random_problem(3, num_detectors=10, num_mechanisms=40, num_observables=4)
    syndromes, _ = _sample_from_model(problem, 6, seed=11)
    active = list(range(problem.num_mechanisms))
    pruned_once = False
    for syndrome in syndromes:
        full, exact = basis_truncated_class_masses(
            problem, syndrome, active, max_states=1 << 20
        )
        assert exact["pruned"] == 0.0
        few, info = basis_truncated_class_masses(
            problem, syndrome, active, max_states=4
        )
        assert info["states"] <= 8
        assert np.isclose(few.sum(), 1.0) and (few >= 0).all()
        assert 0.0 <= info["pruned"] <= 1.0
        # Pruning drops non-negative terms: the retained mass is a lower
        # bound on the exact one.
        assert info["log_mass"] <= exact["log_mass"] + 1e-12
        pruned_once = pruned_once or info["pruned"] > 0
        more, _ = basis_truncated_class_masses(
            problem, syndrome, active, max_states=512
        )
        assert np.abs(more - full).sum() <= np.abs(few - full).sum() + 1e-12
    assert pruned_once
    with pytest.raises(ValueError, match="basis states"):
        basis_truncated_class_masses(problem, syndromes[0], active, max_states=0)


def test_basis_truncation_survives_hundreds_of_small_factors_at_one_site():
    """Four hundred pivot factors close on one free mechanism: their product
    underflows float64, the class masses and the log mass do not."""
    size, prob = 400, 0.01
    problem = DemProblem(
        probs=[prob] * (size + 1),
        detector_rows=[[d, size] for d in range(size)],
        observable_rows=[[size]],
        num_detectors=size,
        num_observables=1,
    )
    syndrome = np.zeros(size, dtype=np.uint8)
    syndrome[: size // 2] = 1
    masses, info = basis_truncated_class_masses(
        problem, syndrome, list(range(size + 1)), max_states=2
    )
    assert info["free"] == 1 and info["pruned"] == 0.0
    assert np.allclose(masses, [1 - prob, prob])
    half = size // 2
    assert np.isclose(info["log_mass"], half * np.log(prob) + half * np.log(1 - prob))


def test_basis_truncation_decides_certain_mechanisms_beforehand():
    """A mechanism that certainly fires is not ranked as if it stayed off.

    Detector 0 holds a pivot of probability 1e-6, a free mechanism of
    probability 0.3 that flips the observable, and a mechanism that certainly
    fires. The free mechanism then almost surely fired too. Ranking the
    states by "every remaining mechanism stays off" would keep the other
    state when a single state is kept.
    """
    problem = DemProblem(
        probs=[1e-6, 0.3, 1.0],
        detector_rows=[[0, 1, 2]],
        observable_rows=[[1]],
        num_detectors=1,
        num_observables=1,
    )
    syndrome = np.zeros(1, dtype=np.uint8)
    exact = _enumerate_active_set(problem, syndrome, [0, 1, 2])
    masses, info = basis_truncated_class_masses(
        problem, syndrome, [0, 1, 2], max_states=1
    )
    assert masses.argmax() == exact.argmax() == 1
    assert info["free"] + info["rank"] == 2
    masses, info = basis_truncated_class_masses(
        problem, syndrome, [0, 1, 2], max_states=4
    )
    assert np.allclose(masses, exact / exact.sum(), atol=1e-12)
    assert np.isclose(info["log_mass"], np.log(exact.sum()))

    checked = 0
    for seed in range(60):
        base = _random_problem(
            seed, num_detectors=5, num_mechanisms=11, num_observables=2
        )
        rng = np.random.default_rng(seed)
        probs = np.array(base.probs)
        certain = rng.choice(11, size=int(rng.integers(1, 4)), replace=False)
        probs[certain] = rng.choice([0.0, 1.0], size=len(certain))
        problem = DemProblem(
            probs=list(probs),
            detector_rows=base.detector_rows,
            observable_rows=base.observable_rows,
            num_detectors=5,
            num_observables=2,
        )
        syndromes, _ = _sample_from_model(problem, 1, seed + 7)
        active = list(range(11))
        exact = _enumerate_active_set(problem, syndromes[0], active)
        if exact.sum() == 0.0:
            with pytest.raises(ArithmeticError, match="positive probability"):
                basis_truncated_class_masses(problem, syndromes[0], active)
            continue
        checked += 1
        masses, info = basis_truncated_class_masses(
            problem, syndromes[0], active, max_states=4096
        )
        assert info["pruned_states"] == 0
        assert np.allclose(masses, exact / exact.sum(), atol=1e-12)
        assert np.isclose(info["log_mass"], np.log(exact.sum()))
        assert info["free"] + info["rank"] == 11 - len(certain)
        # A single state never loses everything: the completion it is
        # ranked by has positive probability.
        rough, _ = basis_truncated_class_masses(
            problem, syndromes[0], active, max_states=1
        )
        assert np.all(rough >= 0.0) and np.isclose(rough.sum(), 1.0)
    assert checked >= 40


def test_basis_truncation_keeps_both_routes_of_the_zero_cutoff_case():
    """Two equally likely solutions in opposite classes, one half each.

    The one-pass SVD contraction can lose one of them to its zero cutoff (see
    the tempering test on the same model); the recursion in the bit basis
    works with probabilities and keeps both.
    """
    problem, syndrome, active = _two_route_problem(10, 1e-3)
    for max_states in (256, 10**6):
        masses, _ = basis_truncated_class_masses(
            problem, syndrome, active, max_states=max_states
        )
        assert np.allclose(masses, [0.5, 0.5], atol=1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_basis_truncation_accepts_zero_probability_factors(seed):
    """Deterministic mechanisms are valid input: zero factors remove
    assignments, they do not raise."""
    certain = DemProblem(
        probs=[0.0],
        detector_rows=[[0]],
        observable_rows=[[]],
        num_detectors=1,
        num_observables=1,
    )
    masses, info = basis_truncated_class_masses(certain, np.zeros(1), [0])
    assert np.allclose(masses, [1.0, 0.0]) and info["log_mass"] == 0.0
    # The same mechanism cannot explain a set detector: nothing to normalize.
    with pytest.raises(ArithmeticError, match="positive probability"):
        basis_truncated_class_masses(certain, np.ones(1), [0])

    problem = _random_problem(seed)
    rng = np.random.default_rng(seed)
    probs = np.array(problem.probs)
    probs[rng.choice(problem.num_mechanisms, size=3, replace=False)] = 0.0
    problem = DemProblem(
        probs=list(probs),
        detector_rows=problem.detector_rows,
        observable_rows=problem.observable_rows,
        num_detectors=problem.num_detectors,
        num_observables=problem.num_observables,
    )
    syndromes, _ = _sample_from_model(problem, 3, seed + 50)
    active = list(range(problem.num_mechanisms))
    for syndrome in syndromes:
        exact = _enumerate_active_set(problem, syndrome, active)
        masses, info = basis_truncated_class_masses(
            problem, syndrome, active, max_states=4096
        )
        assert np.allclose(masses, exact / exact.sum(), atol=1e-12)
        assert np.isclose(info["log_mass"], np.log(exact.sum()))


def test_basis_truncation_keeps_a_weight_that_is_tiny_until_its_pivots_close():
    """After the first free mechanism the winning class is 1200 orders of
    magnitude behind; the pivots still open reverse that. Nothing is pruned,
    so the result must be exact."""
    prob, num_single, per_group, groups = 1e-3, 400, 200, 5
    num_pivots = num_single + per_group * groups
    first = num_pivots
    # Turning the observable mechanism on fires 400 pivots at once. Leaving
    # it off costs more: every one of the five groups then needs its own
    # mechanism of probability 1e-300.
    rows = [[d, first] for d in range(num_single)] + [
        [num_single + g * per_group + i, first, first + 1 + g]
        for g in range(groups)
        for i in range(per_group)
    ]
    problem = DemProblem(
        probs=[prob] * num_pivots + [0.5] + [1e-300] * groups,
        detector_rows=rows,
        observable_rows=[[first]],
        num_detectors=num_pivots,
        num_observables=1,
    )
    syndrome = np.array([0] * num_single + [1] * (per_group * groups), dtype=np.uint8)
    masses, info = basis_truncated_class_masses(
        problem, syndrome, list(range(num_pivots + 1 + groups)), max_states=64
    )
    assert info["free"] == 1 + groups
    assert info["pruned_states"] == 0 and info["pruned"] == 0.0
    assert np.allclose(masses, [0.0, 1.0])
    assert np.isclose(
        info["log_mass"],
        np.log(0.5) + num_single * np.log(prob) + per_group * groups * np.log(1 - prob),
    )


def test_basis_truncation_reports_the_removal_of_a_blocked_state():
    """A state whose open pivot has a zero factor can be unblocked by a later
    mechanism. Removing it is a loss and must be reported as one."""
    problem = DemProblem(
        probs=[0.0, 0.5, 0.5],
        detector_rows=[[0, 1, 2]],
        observable_rows=[[1]],
        num_detectors=1,
        num_observables=1,
    )
    exact, info = basis_truncated_class_masses(problem, np.zeros(1), [0, 1, 2])
    assert np.allclose(exact, [0.5, 0.5]) and info["pruned_states"] == 0
    _, info = basis_truncated_class_masses(
        problem, np.zeros(1), [0, 1, 2], max_states=1
    )
    assert info["pruned_states"] == 1 and info["pruned"] > 0.0


def test_a_tiny_pruned_share_is_reported_as_it_is():
    """A share of 1e-80 must not round to zero: the decoder compares it with
    ``temper_threshold``, which a caller may set far below double precision
    of one."""
    problem = DemProblem(
        probs=[1e-100, 1e-20, 0.5],
        detector_rows=[[0, 1, 2]],
        observable_rows=[[1]],
        num_detectors=1,
        num_observables=1,
    )
    _, info = basis_truncated_class_masses(
        problem, np.ones(1, dtype=np.uint8), [0, 1, 2], max_states=1
    )
    assert info["pruned_states"] > 0
    assert 1e-90 < info["pruned"] < 1e-60


def test_decoder_with_basis_truncation_matches_maximum_likelihood():
    """The decoder option agrees with the full-chain reference and the SVD sum."""
    problem = _small_surface_problem(0.02)
    basis = ClassSumDecoder(problem, truncation="basis")
    svd = ClassSumDecoder(problem, chi_max=16)
    syndromes, _ = _sample_from_model(problem, 80, seed=3)
    agree = same = 0
    for syndrome in syndromes:
        flips, info = basis.decode(syndrome)
        other, other_info = svd.decode(syndrome)
        _, reference = decode_dem(problem, syndrome.astype(int), chi_max=4096)
        agree += int(np.array_equal(flips, reference))
        same += int(np.array_equal(flips, other))
        assert not info["tempered"]
        assert info["lost"] == 0.0 and info["pruned"] == 0.0
        assert other_info["pruned"] == 0.0
    assert agree >= 78 and same == len(syndromes)
    with pytest.raises(ValueError, match="truncation"):
        ClassSumDecoder(problem, truncation="schmidt")
    with pytest.raises(ValueError, match="basis states"):
        ClassSumDecoder(problem, truncation="basis", max_states=0)


def test_widening_recontracts_a_superset_and_decides_from_it(monkeypatch):
    """A truncated first pass widens the search; the class sum is repeated on
    a set that contains the first one, and that second sum gives the verdict."""
    problem = _random_problem(2, num_detectors=10, num_mechanisms=40, num_observables=6)
    decoder = ClassSumDecoder(problem, extra_centers=64, temper_stages=1)
    calls = []

    def truncated_then_decided(syndrome, active, stages):
        calls.append(list(active))
        masses = np.zeros(1 << problem.num_observables)
        # First sum: truncated, verdict 0. Second sum: verdict 5.
        masses[0 if len(calls) == 1 else 5] = 1.0
        return masses, {"lost": 1.0 if len(calls) == 1 else 0.0, "free": 1}

    monkeypatch.setattr(decoder, "_class_masses", truncated_then_decided)
    syndromes, _ = _sample_from_model(problem, 60, seed=5)
    recontracted = 0
    for syndrome in syndromes[syndromes.any(axis=1)]:
        calls.clear()
        flips, info = decoder.decode(syndrome)
        if info["candidates"] == 0:
            continue
        assert len(calls) in (1, 2)
        if len(calls) == 2:
            assert info["widened"]
            assert set(calls[0]) < set(calls[1])
            assert info["active"] == len(calls[1])
            if info["bposd_class"] != 5:
                assert info["predicted_class"] == 5
                assert int(flips @ (1 << np.arange(flips.size))) == 5
            recontracted += 1
        else:
            assert info["active"] == len(calls[0])
    assert recontracted > 0


def test_widening_that_asks_for_no_solve_is_not_reported(monkeypatch):
    """With every solve of the syndrome already made, the untried classes cost
    nothing to visit and change nothing: the decode is not marked as widened
    and the class sum is not run again."""
    problem = _random_problem(2, num_detectors=10, num_mechanisms=40, num_observables=6)
    decoder = ClassSumDecoder(problem, extra_centers=64, temper_stages=1)
    calls = []

    def truncated(syndrome, active, stages):
        calls.append(list(active))
        masses = np.zeros(1 << problem.num_observables)
        masses[3] = 1.0
        return masses, {"lost": 1.0, "free": 1}

    search = decoder._search

    def search_with_every_solve_made(syndrome, seed):
        found, tried, solves = search(syndrome, seed)
        for j in range(problem.num_observables):
            for target in (0, 1):
                solves.setdefault((j, target), None)
        return found, tried, solves

    monkeypatch.setattr(decoder, "_class_masses", truncated)
    monkeypatch.setattr(decoder, "_search", search_with_every_solve_made)
    syndromes, _ = _sample_from_model(problem, 60, seed=5)
    untried = 0
    for syndrome in syndromes[syndromes.any(axis=1)]:
        calls.clear()
        _, info = decoder.decode(syndrome)
        assert not info["widened"] and len(calls) <= 1
        untried += info["candidates"] > decoder.recenter_rounds
    assert untried > 0


def test_a_wider_sum_that_vanishes_keeps_the_first_verdict(monkeypatch):
    """Widening must not abort a decode that already has a verdict."""
    problem = _random_problem(2, num_detectors=10, num_mechanisms=40, num_observables=6)
    decoder = ClassSumDecoder(problem, extra_centers=64, temper_stages=1)
    calls = []

    def truncated_then_vanished(syndrome, active, stages):
        calls.append(list(active))
        if len(calls) > 1:
            raise ArithmeticError("vanished")
        masses = np.zeros(1 << problem.num_observables)
        masses[3] = 1.0
        return masses, {"lost": 1.0, "free": 1}

    monkeypatch.setattr(decoder, "_class_masses", truncated_then_vanished)
    syndromes, _ = _sample_from_model(problem, 60, seed=5)
    retried = 0
    for syndrome in syndromes[syndromes.any(axis=1)]:
        calls.clear()
        _, info = decoder.decode(syndrome)
        if len(calls) == 2:
            retried += 1
            assert info["widened"] and info["active"] == len(calls[0])
            if info["bposd_class"] != 3:
                assert info["predicted_class"] == 3
    assert retried > 0


def test_the_search_never_repeats_a_solve():
    """One syndrome has two distinct solves per observable; a wide search
    makes each of them at most once."""
    problem = _random_problem(2, num_detectors=10, num_mechanisms=40, num_observables=6)
    decoder = ClassSumDecoder(problem, extra_centers=64)
    counts = []

    class Counting:
        def __init__(self, inner):
            self.inner = inner

        def decode(self, extended):
            counts.append(tuple(int(b) for b in extended))
            return self.inner.decode(extended)

    decoder._flipped = [Counting(inner) for inner in decoder._flipped]
    syndromes, _ = _sample_from_model(problem, 30, seed=5)
    widened = 0
    for syndrome in syndromes[syndromes.any(axis=1)]:
        syndrome = syndrome.astype(np.uint8)
        seed = np.asarray(decoder._bposd.decode(syndrome), dtype=np.uint8)
        if not decoder._satisfies(seed, syndrome):
            continue
        counts.clear()
        found, tried, solves = decoder._search(syndrome, seed)
        stock = len(counts)
        wider = decoder.widen(syndrome, found, tried, solves)
        assert len(counts) <= 2 * problem.num_observables
        assert len(counts) == len(solves)
        assert set(found) <= set(wider)
        widened += len(counts) > stock
        # Without a cache from the search, the centers still share one.
        counts.clear()
        decoder.widen(syndrome, found, tried)
        assert len(counts) <= 2 * problem.num_observables
    assert widened > 0


def test_the_center_budget_is_spent_on_centers_that_solve_something():
    """A center whose flips were all solved already does not use up the budget.

    With one extra center, the most likely untried class sometimes needs no
    new solve while a less likely one does. The wider search then moves on
    to that one instead of doing nothing.
    """
    problem = _random_problem(2, num_detectors=12, num_mechanisms=60, num_observables=6)
    decoder = ClassSumDecoder(problem, extra_centers=1)
    syndromes, _ = _sample_from_model(problem, 80, seed=5)
    passed_over = 0
    for syndrome in syndromes[syndromes.any(axis=1)]:
        syndrome = syndrome.astype(np.uint8)
        seed = np.asarray(decoder._bposd.decode(syndrome), dtype=np.uint8)
        if not decoder._satisfies(seed, syndrome):
            continue
        found, tried, solves = decoder._search(syndrome, seed)
        centers = sorted(
            (label for label in found if label not in tried),
            key=lambda label: -found[label][0],
        )
        missing = [
            sum(
                (j, ((center >> j) & 1) ^ 1) not in solves
                for j in range(problem.num_observables)
            )
            for center in centers
        ]
        useful = next((n for n in missing if n > 0), 0)
        before = len(solves)
        decoder.widen(syndrome, found, tried, solves)
        # Exactly the first center with something to solve is used.
        assert len(solves) - before == useful
        passed_over += bool(missing) and missing[0] == 0 and useful > 0
    assert passed_over > 0


def test_decoder_beats_bposd_on_the_bb72_memory_circuit():
    """Fewer logical errors than BP-OSD on the same circuit-level shots.

    One syndrome-extraction round of the [[72, 12, 6]] bivariate bicycle code
    at p = 0.01, where BP-OSD fails on about 9 % of the shots and the class sum
    on about 7 %.
    """
    problem = _bb72_problem(rounds=1, p=0.01)
    assert problem.num_detectors == 72 and problem.num_observables == 12
    check, observables = problem_matrices(problem)
    syndromes, truth = _sample_from_model(problem, 250, seed=2026)
    decoder = ClassSumDecoder(problem, chi_max=16)
    basis = ClassSumDecoder(problem, truncation="basis", max_states=64)
    bposd = BpOsdDecoder(
        check, error_channel=list(problem.probs), **DEFAULT_BPOSD_OPTIONS
    )
    class_sum_failures = bposd_failures = fixes = breaks = basis_failures = 0
    for syndrome, expected in zip(syndromes, truth):
        flips, info = decoder.decode(syndrome)
        basis_flips, _ = basis.decode(syndrome)
        basis_failures += int(not np.array_equal(basis_flips, expected))
        reference = (observables @ bposd.decode(syndrome)) % 2
        assert info["bposd_class"] == int(reference @ (1 << np.arange(12)))
        ours_wrong = not np.array_equal(flips, expected)
        theirs_wrong = not np.array_equal(reference, expected)
        class_sum_failures += ours_wrong
        bposd_failures += theirs_wrong
        fixes += theirs_wrong and not ours_wrong
        breaks += ours_wrong and not theirs_wrong
    assert bposd_failures > 0
    assert class_sum_failures < bposd_failures
    assert fixes > breaks
    assert basis_failures < bposd_failures
    assert basis_failures <= class_sum_failures + 2
