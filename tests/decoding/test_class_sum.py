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
            problem, syndrome, active, chi_max=256, stages=stages
        )
        assert info["lost"] < 1e-20
        assert np.allclose(masses, exact / exact.sum(), atol=1e-12)
        assert np.isclose(info["log_mass"], np.log(exact.sum()))


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
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        ClassSumDecoder(degenerate)
    empty = DemProblem(
        probs=[],
        detector_rows=[[]],
        observable_rows=[[]],
        num_detectors=1,
        num_observables=1,
    )
    with pytest.raises(ValueError, match="no mechanisms"):
        ClassSumDecoder(empty)
    with pytest.raises(ValueError, match="bond dimension"):
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


def test_empty_syndrome_with_likely_mechanisms_is_searched():
    """Priors above 1/2 make a flip the likeliest answer to the empty syndrome."""
    problem = DemProblem(
        probs=[0.9, 0.8],
        detector_rows=[[0, 1]],
        observable_rows=[[0]],
        num_detectors=1,
        num_observables=1,
    )
    flips, info = ClassSumDecoder(problem).decode(np.zeros(1))
    assert flips.tolist() == [1] and info["predicted_class"] == 1


def test_a_tie_goes_to_the_bposd_class():
    """Two mechanisms with the same detectors and prior tie two classes."""
    problem = DemProblem(
        probs=[0.1, 0.1, 0.1],
        detector_rows=[[0, 1, 2]],
        observable_rows=[[0], [1]],
        num_detectors=1,
        num_observables=2,
    )
    decoder = ClassSumDecoder(problem)
    flips, info = decoder.decode(np.ones(1))
    assert info["margin"] < 1e-9
    assert info["predicted_class"] == info["bposd_class"]
    assert int(flips @ (1 << np.arange(2))) == info["bposd_class"]


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
        found = narrow.candidates(syndrome, seed)
        assert narrow.widen(syndrome, found) == found
        wider = wide.widen(syndrome, found)
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
    widened = larger = 0
    for syndrome in syndromes:
        flips, info = default.decode(syndrome)
        same_flips, same = options.decode(syndrome)
        assert not same["widened"] and not same["tempered"]
        assert np.array_equal(flips, same_flips)
        assert same["candidates"] == info["candidates"]
        _, narrow = stock.decode(syndrome)
        hard_flips, hard = forced.decode(syndrome)
        assert hard["widened"] == narrow["tempered"]
        assert hard["candidates"] >= narrow["candidates"]
        if hard["tempered"]:
            # The tempered pass at chi 64 is exact on this small model.
            assert hard["lost"] < 1e-8
            _, reference = decode_dem(problem, syndrome.astype(int), chi_max=4096)
            larger += int(np.array_equal(hard_flips, reference))
        widened += hard["widened"]
    assert widened > 5 and larger >= widened - 2


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
    bposd = BpOsdDecoder(
        check, error_channel=list(problem.probs), **DEFAULT_BPOSD_OPTIONS
    )
    class_sum_failures = bposd_failures = fixes = breaks = 0
    for syndrome, expected in zip(syndromes, truth):
        flips, info = decoder.decode(syndrome)
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
