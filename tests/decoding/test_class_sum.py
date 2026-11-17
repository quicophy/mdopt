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
