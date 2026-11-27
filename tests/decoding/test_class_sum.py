"""Tests for the class sums of a detector error model on an active set.

The contraction is checked against dense linear algebra and against exact
enumeration of every solution inside an active set.
"""

import itertools

import numpy as np
import pytest
import stim

from mdopt.decoding.class_sum import (
    ParityChain,
    active_set_class_masses,
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
