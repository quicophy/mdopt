"""End-to-end checks on the decoders in ``examples.decoding.decoding``.

The reference posterior is computed by brute force: the decoder enforces even
parity on every check, so the surviving configurations f are exactly those with
``H_x f_x = 0`` and ``H_z f_z = 0``. Each is weighted by the bias channel on the
residual ``d = f XOR e``, and ``marginal`` contracts the physical legs with the
all-ones trace vector, so a logical class carries the *sum of amplitudes* -- its
weight is that sum squared, not the sum of the squares.
"""

import itertools
import logging

import numpy as np
import pytest
import qecstruct as qec

from mdopt.mps.utils import inner_product

from examples.decoding.decoding import (
    css_code_stabilisers,
    decode_css,
    decode_custom,
    depolarising_bias,
    generate_pauli_error_string,
    max_amplitude_bound,
    max_product_readout,
    _score_tie,
    multiply_pauli_strings,
)

ORDER = ["I", "X", "Z", "Y"]
BITS = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}


def _dense(mat, num_qubits):
    out = []
    for row in mat.rows():
        vec = np.zeros(num_qubits, dtype=int)
        for col in row:
            vec[col] = 1
        out.append(vec)
    return np.array(out, dtype=int).reshape(-1, num_qubits)


def _nullspace_gf2(mat, num_qubits):
    """Basis of {v : mat v = 0} over GF(2)."""
    if mat.size == 0:
        return list(np.eye(num_qubits, dtype=int))
    work, pivots, row = mat.copy() % 2, [], 0
    for col in range(num_qubits):
        piv = next((i for i in range(row, work.shape[0]) if work[i, col]), None)
        if piv is None:
            continue
        work[[row, piv]] = work[[piv, row]]
        for i in range(work.shape[0]):
            if i != row and work[i, col]:
                work[i] = (work[i] + work[row]) % 2
        pivots.append(col)
        row += 1
        if row == work.shape[0]:
            break
    basis = []
    for free in (c for c in range(num_qubits) if c not in pivots):
        vec = np.zeros(num_qubits, dtype=int)
        vec[free] = 1
        for i, col in enumerate(pivots):
            vec[col] = work[i, free]
        basis.append(vec % 2)
    return basis


def _span(basis, num_qubits):
    if not basis:
        return [np.zeros(num_qubits, dtype=int)]
    out = []
    for bits in itertools.product((0, 1), repeat=len(basis)):
        vec = np.zeros(num_qubits, dtype=int)
        for bit, gen in zip(bits, basis):
            if bit:
                vec = (vec + gen) % 2
        out.append(vec)
    return out


def exact_posterior(code, error, bias_type, prob):
    """Posterior over the four logical classes, by explicit enumeration."""
    num_qubits = len(code)
    err_x = np.array([BITS[c][0] for c in error], dtype=int)
    err_z = np.array([BITS[c][1] for c in error], dtype=int)
    checks_x = _dense(code.x_stabs_binary(), num_qubits)
    checks_z = _dense(code.z_stabs_binary(), num_qubits)
    log_x = _dense(code.x_logicals_binary(), num_qubits)
    log_z = _dense(code.z_logicals_binary(), num_qubits)

    if bias_type == "Depolarising":

        def weight(d_x, d_z):
            return (1 - prob) if (d_x == 0 and d_z == 0) else prob / 3

    else:

        def weight(d_x, d_z):
            return ((1 - prob) if d_x == 0 else prob) * (
                (1 - prob) if d_z == 0 else prob
            )

    amps = dict.fromkeys(ORDER, 0.0)
    names = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    for f_x in _span(_nullspace_gf2(checks_x, num_qubits), num_qubits):
        d_x = (f_x + err_x) % 2
        l_x = int(np.dot(f_x, log_x[0]) % 2)
        for f_z in _span(_nullspace_gf2(checks_z, num_qubits), num_qubits):
            d_z = (f_z + err_z) % 2
            l_z = int(np.dot(f_z, log_z[0]) % 2)
            amp = 1.0
            for j in range(num_qubits):
                amp *= np.sqrt(weight(d_x[j], d_z[j]))
            amps[names[(l_x, l_z)]] += amp

    probs = {k: v * v for k, v in amps.items()}
    total = sum(probs.values())
    return np.array([probs[k] / total for k in ORDER])


def _logicals_as_pauli(code):
    num_qubits = len(code)
    log_x = _dense(code.x_logicals_binary(), num_qubits)
    log_z = _dense(code.z_logicals_binary(), num_qubits)
    as_str = lambda row, p: "".join(p if row[j] else "I" for j in range(num_qubits))
    return [as_str(log_x[0], "X")], [as_str(log_z[0], "Z")]


def _normalised(dense_out):
    probs = np.asarray(dense_out, dtype=float) ** 2
    return probs / probs.sum()


CODES = {"steane": qec.steane_code, "shor": qec.shor_code}
ERRORS = {
    "steane": ["XIIIIII", "ZIIIIII", "YIIIIII", "XZIIIII", "XXXIIII"],
    "shor": ["XIIIIIIII", "ZIIIIIIII", "YIIIIIIII", "XZIIIIIII", "XYZIIIIII"],
}


@pytest.mark.parametrize("name", sorted(CODES))
@pytest.mark.parametrize("bias", ["Bitflip", "Depolarising"])
def test_decode_css_matches_exact_posterior(name, bias):
    """decode_css reproduces the brute-force posterior exactly."""
    code = CODES[name]()
    for error in ERRORS[name]:
        got = _normalised(
            decode_css(
                code,
                error,
                chi_max=int(1e5),
                bias_type=bias,
                bias_prob=0.1,
                renormalise=True,
                silent=True,
            )[0]
        )
        assert np.allclose(got, exact_posterior(code, error, bias, 0.1), atol=1e-6)


@pytest.mark.parametrize("name", sorted(CODES))
@pytest.mark.parametrize("bias", ["Bitflip", "Depolarising"])
def test_decode_custom_matches_exact_posterior(name, bias):
    """decode_custom agrees with the same reference.

    A stabiliser's syndrome is a symplectic product, so its Z part must be
    checked against the X components of the error and vice versa. Getting that
    backwards is invisible for a self-dual code such as Steane and wrong for
    Shor, so both are exercised here.
    """
    code = CODES[name]()
    stabs = sum(css_code_stabilisers(code), [])
    log_x, log_z = _logicals_as_pauli(code)
    for error in ERRORS[name]:
        got = _normalised(
            decode_custom(
                stabs,
                log_x,
                log_z,
                error,
                chi_max=int(1e5),
                bias_type=bias,
                bias_prob=0.1,
                renormalise=True,
                silent=True,
            )[0]
        )
        assert np.allclose(got, exact_posterior(code, error, bias, 0.1), atol=1e-6)


def test_depolarising_bias_reproduces_the_channel():
    """The two-site bias MPO must be the depolarising channel itself."""
    prob = 0.12
    operator = depolarising_bias(prob)
    for a, b in itertools.product((0, 1), repeat=2):
        assert operator[a, b, a, b] == pytest.approx(np.sqrt(1 - prob))
    for a, b, c, d in itertools.product((0, 1), repeat=4):
        if (a, b) != (c, d):
            assert operator[a, b, c, d] == pytest.approx(np.sqrt(prob / 3))
    assert np.allclose((operator.reshape(4, 4) ** 2).sum(axis=1), 1.0)


def test_five_qubit_code_corrects_every_single_qubit_error():
    """The [[5,1,3]] perfect code corrects all weight-1 Paulis."""
    stabs = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    for site, pauli in itertools.product(range(5), "XYZ"):
        error = "".join(pauli if i == site else "I" for i in range(5))
        dense_out, success = decode_custom(
            stabs,
            ["XXXXX"],
            ["ZZZZZ"],
            error,
            chi_max=int(1e4),
            bias_type="Depolarising",
            bias_prob=0.05,
            renormalise=True,
            silent=True,
        )
        assert success == 1
        assert int(np.argmax(np.asarray(dense_out, dtype=float))) == 0


def test_distance_three_code_corrects_every_erasure_below_distance():
    """Any pattern of at most d-1 = 2 erasures must be correctable."""
    code = qec.hypergraph_product(qec.repetition_code(3), qec.repetition_code(3))
    num_qubits = len(code)
    for weight in (1, 2):
        for combo in itertools.combinations(range(num_qubits), weight):
            error = "".join("E" if i in combo else "I" for i in range(num_qubits))
            _, success = decode_css(
                code,
                error,
                chi_max=128,
                bias_type="Bitflip",
                bias_prob=0.05,
                renormalise=True,
                silent=True,
            )
            assert success == 1, f"erasure pattern {combo} should be correctable"


def test_dense_and_dmrg_readout_agree_when_the_map_class_is_unique():
    """The two readout paths must reach the same verdict, ties included.

    Dense contraction and Dephasing DMRG select the MAP class by different means,
    so they must agree. Degenerate posteriors are the interesting case: DMRG
    returns whichever tied class its sweep lands on, so it compares that class's
    amplitude against the identity's rather than testing for equality -- without
    that, a tie would score as a failure here and a success under dense readout.

    ``dense_readout_max_sites=0`` forces the DMRG branch, which is otherwise
    unreachable for a code small enough to cross-check.
    """
    code = qec.steane_code()
    num_qubits = len(code)
    rng = np.random.default_rng(11)
    compared = 0
    degenerate_seen = []

    for _ in range(25):
        error = generate_pauli_error_string(
            num_qubits, 0.15, error_model="Depolarising", rng=rng
        )
        if error == "I" * num_qubits:
            continue

        dense_out, dense_success = decode_css(
            code,
            error,
            chi_max=int(1e4),
            bias_type="Depolarising",
            bias_prob=0.15,
            renormalise=True,
            silent=True,
        )
        amplitudes = np.asarray(dense_out, dtype=float)
        amplitudes = amplitudes / amplitudes.max()
        degenerate = np.count_nonzero(amplitudes > 1 - 1e-9) > 1
        if degenerate:
            degenerate_seen.append(error)

        _, overlap = decode_css(
            code,
            error,
            chi_max=int(1e4),
            bias_type="Depolarising",
            bias_prob=0.15,
            renormalise=True,
            silent=True,
            num_runs=2,
            dense_readout_max_sites=0,
        )
        assert (
            int(round(float(overlap))) == dense_success
        ), f"readouts disagree on {error} (degenerate: {degenerate})"
        compared += 1

    assert compared >= 5, "too few shots to be a meaningful check"
    assert degenerate_seen, "no degenerate posterior arose, so ties went untested"


@pytest.mark.parametrize("bias", ["Bitflip", "Depolarising"])
def test_qubit_reordering_leaves_the_posterior_invariant(bias):
    """RCM reordering is a relabelling, so exact results must not move.

    The optimisation permutes qubits along the chain to shorten the MPO spans.
    That is meant to buy speed and a lower bond dimension, never a different
    answer, so at exact contraction the two orderings have to agree to machine
    precision.
    """
    code = qec.steane_code()
    num_qubits = len(code)
    rng = np.random.default_rng(4)
    compared = 0

    for _ in range(8):
        error = generate_pauli_error_string(
            num_qubits, 0.15, error_model="Depolarising", rng=rng
        )
        if error == "I" * num_qubits:
            continue
        posteriors = []
        for strategy in ("Natural", "Optimised"):
            dense_out, _ = decode_css(
                code,
                error,
                chi_max=int(1e5),
                bias_type=bias,
                bias_prob=0.15,
                renormalise=True,
                silent=True,
                qubit_order_strategy=strategy,
            )
            probs = np.asarray(dense_out, dtype=float) ** 2
            posteriors.append(probs / probs.sum())
        assert np.allclose(
            posteriors[0], posteriors[1], atol=1e-9
        ), f"reordering changed the posterior for {error}"
        compared += 1

    assert compared >= 3


def test_dmrg_readout_finds_the_true_maximum_on_decoder_posteriors():
    """DMRG must return a genuine maximiser of the logical MPS.

    Checking only whether the identity was returned is too weak: DMRG could stop
    on some other suboptimal class and go unnoticed. Here the logical MPS is
    densified so the true maximum is known exactly, and the amplitude DMRG
    reached is compared against it.

    Decoder posteriors are strongly peaked, which is the regime Dephasing DMRG
    handles reliably. It is a local optimiser, so on a nearly flat posterior --
    near threshold, or at large k -- it can settle below the true maximum; that
    is what ``num_restarts`` mitigates.
    """
    code = qec.steane_code()
    num_qubits = len(code)
    rng = np.random.default_rng(21)
    checked = 0

    for _ in range(12):
        error = generate_pauli_error_string(
            num_qubits, 0.15, error_model="Depolarising", rng=rng
        )
        if error == "I" * num_qubits:
            continue

        engine, _ = decode_css(
            code,
            error,
            chi_max=int(1e4),
            bias_type="Depolarising",
            bias_prob=0.15,
            renormalise=True,
            silent=True,
            num_runs=2,
            dense_readout_max_sites=0,
        )
        logical_mps = engine.mps_target
        amplitudes = np.abs(np.asarray(logical_mps.dense(flatten=True), dtype=float))
        found = abs(inner_product(engine.mps, logical_mps))

        assert found == pytest.approx(
            amplitudes.max(), rel=1e-6
        ), f"DMRG stopped below the true maximum for {error}"
        checked += 1

    assert checked >= 5


def test_dmrg_readout_returns_a_computational_basis_state():
    """The readout must be a bitstring, not a superposition.

    Dephasing DMRG restricts the search to computational basis states, so the
    answer has to be a product state with unit bond dimensions -- anything else
    means coherence leaked through the sweep and the "class" it names is not
    well defined.
    """
    code = qec.steane_code()
    engine, _ = decode_css(
        code,
        "XZIIIII",
        chi_max=int(1e4),
        bias_type="Depolarising",
        bias_prob=0.15,
        renormalise=True,
        silent=True,
        num_runs=2,
        dense_readout_max_sites=0,
    )
    answer = engine.mps
    assert answer.bond_dimensions == [1] * answer.num_bonds
    for tensor in answer.tensors:
        weights = np.abs(np.asarray(tensor, dtype=float).reshape(-1))
        weights = weights / weights.max()
        assert np.count_nonzero(weights > 1e-9) == 1


def test_max_amplitude_bound_is_sound_and_tight_when_converged():
    """The max-product bound must never fall below the true maximum.

    Propagating absolute values can only over-estimate, so the bound is always
    valid. When the run is converged the logical MPS has no negative amplitudes
    -- nothing in the pipeline can produce one -- so there is no cancellation
    and the bound is exact.
    """
    code = qec.steane_code()
    rng = np.random.default_rng(13)
    checked = 0

    for _ in range(10):
        error = generate_pauli_error_string(
            len(code), 0.15, error_model="Depolarising", rng=rng
        )
        if error == "I" * len(code):
            continue

        engine, _ = decode_css(
            code,
            error,
            chi_max=int(1e4),
            bias_type="Depolarising",
            bias_prob=0.15,
            renormalise=True,
            silent=True,
            num_runs=1,
            dense_readout_max_sites=0,
        )
        logical_mps = engine.mps_target
        amplitudes = np.asarray(logical_mps.dense(flatten=True), dtype=float)

        assert amplitudes.min() >= -1e-12, "a converged run should stay non-negative"
        bound = max_amplitude_bound(logical_mps)
        assert bound >= np.abs(amplitudes).max() - 1e-12, "bound is not an upper bound"
        assert bound == pytest.approx(np.abs(amplitudes).max(), rel=1e-9)
        checked += 1

    assert checked >= 4


def test_truncation_shows_up_as_a_negative_logical_amplitude(caplog):
    """A negative amplitude is a free chi-convergence signal.

    Nothing in an exact run can make one, so it can only be a low-rank artefact.
    Squeezing chi_max should surface it and a converged chi_max should not.
    """
    code = qec.steane_code()
    rng = np.random.default_rng(13)
    errors = []
    while len(errors) < 6:
        error = generate_pauli_error_string(
            len(code), 0.15, error_model="Depolarising", rng=rng
        )
        if error != "I" * len(code):
            errors.append(error)

    def warnings_for(chi_max):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            for error in errors:
                decode_css(
                    code,
                    error,
                    chi_max=chi_max,
                    bias_type="Depolarising",
                    bias_prob=0.15,
                    renormalise=True,
                    silent=False,
                )
        return [r for r in caplog.records if "Negative logical amplitude" in r.message]

    assert warnings_for(4), "an aggressively truncated run should be flagged"
    assert not warnings_for(64), "a converged run should not be flagged"


def test_max_product_readout_is_optimal_and_certified():
    """Beam search should settle the readout without needing DMRG.

    The max-product pass proposes a basis state and the bound caps what any
    basis state could reach. When they meet, the proposal is provably the
    maximum -- no variational sweep, no local minima, no restarts. This asserts
    both that the proposal really is the maximum (checked against the densified
    logical MPS) and that the bracket closes so the answer is certified.
    """
    code = qec.steane_code()
    rng = np.random.default_rng(7)
    checked = 0

    for _ in range(12):
        error = generate_pauli_error_string(
            len(code), 0.15, error_model="Depolarising", rng=rng
        )
        if error == "I" * len(code):
            continue

        engine, _ = decode_css(
            code,
            error,
            chi_max=int(1e4),
            bias_type="Depolarising",
            bias_prob=0.15,
            renormalise=True,
            silent=True,
            num_runs=1,
            dense_readout_max_sites=0,
        )
        logical_mps = engine.mps_target
        true_max = np.abs(
            np.asarray(logical_mps.dense(flatten=True), dtype=float)
        ).max()

        _, amplitude = max_product_readout(logical_mps)
        bound = max_amplitude_bound(logical_mps)

        assert amplitude == pytest.approx(true_max, rel=1e-9)
        assert amplitude <= bound * (1 + 1e-9), "a witness cannot exceed the bound"
        assert amplitude >= bound * (
            1 - 1e-9
        ), "bracket should close on a converged run"
        checked += 1

    assert checked >= 5


def test_readout_skips_dmrg_when_the_bracket_closes(caplog):
    """A certified beam-search result should make the DMRG sweep unnecessary."""
    caplog.clear()
    with caplog.at_level(logging.INFO):
        decode_css(
            qec.steane_code(),
            "XZIIIII",
            chi_max=int(1e4),
            bias_type="Depolarising",
            bias_prob=0.15,
            renormalise=True,
            silent=False,
            dense_readout_max_sites=0,
        )
    messages = [r.message for r in caplog.records]
    assert any("settled by beam search" in m for m in messages)
    assert not any("falling back to Dephasing DMRG" in m for m in messages)


@pytest.mark.parametrize(
    "policy,degeneracy,expected",
    [
        ("optimistic", 1, 1.0),
        ("optimistic", 4, 1.0),
        ("fractional", 1, 1.0),
        ("fractional", 4, 0.25),
        ("pessimistic", 1, 1.0),
        ("pessimistic", 4, 0.0),
    ],
)
def test_tie_policies_score_a_degenerate_map_set(policy, degeneracy, expected):
    """A tie has to be scored by some convention; make each one explicit."""
    assert _score_tie(True, degeneracy, policy) == pytest.approx(expected)
    assert _score_tie(False, degeneracy, policy) == 0.0


def test_unknown_tie_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown tie_policy"):
        _score_tie(True, 2, "whatever")


def test_tie_policy_is_irrelevant_under_bitflip_noise():
    """Bit-flip noise freezes the Z sector, so no exact degeneracies arise.

    All three conventions must therefore agree, which is what makes the existing
    bit-flip datasets independent of this choice.
    """
    code = qec.steane_code()
    rng = np.random.default_rng(5)
    errors = []
    while len(errors) < 12:
        error = generate_pauli_error_string(
            len(code), 0.08, error_model="Bitflip", rng=rng
        )
        if error != "I" * len(code):
            errors.append(error)

    scores = {}
    for policy in ("optimistic", "fractional", "pessimistic"):
        scores[policy] = [
            float(
                decode_css(
                    code,
                    error,
                    chi_max=256,
                    bias_type="Bitflip",
                    bias_prob=0.08,
                    renormalise=True,
                    silent=True,
                    tie_policy=policy,
                )[1]
            )
            for error in errors
        ]
    assert scores["optimistic"] == scores["fractional"] == scores["pessimistic"]


def test_multiply_by_stabiliser_uses_the_supplied_generator():
    """The stabiliser draw must come from the passed generator, not global state.

    Reaching for ``np.random`` made this path depend on process-wide state, so a
    run that fell back to it could not be reproduced from its seed.
    """
    code = qec.steane_code()
    error = "X" + "I" * (len(code) - 1)

    def run(seed):
        return decode_css(
            code,
            error,
            chi_max=256,
            bias_type="Depolarising",
            bias_prob=0.1,
            renormalise=True,
            silent=True,
            multiply_by_stabiliser=True,
            rng=np.random.default_rng(seed),
        )[0]

    assert np.allclose(np.asarray(run(7), float), np.asarray(run(7), float))


def test_posterior_is_invariant_under_multiplication_by_a_stabiliser():
    """Multiplying the error by a stabiliser must not move the posterior.

    A stabiliser maps the error to a different representative of the same coset,
    so every logical class keeps its weight. This is what makes
    ``multiply_by_stabiliser`` a safe retry when a decode fails numerically, and
    it is a sharp check on the constraint machinery: get the syndrome wiring
    wrong and the invariance breaks immediately.
    """
    code = qec.steane_code()
    stabilisers = sum(css_code_stabilisers(code), [])
    error = "XZ" + "I" * (len(code) - 2)

    def posterior(err):
        return np.asarray(
            decode_css(
                code,
                err,
                chi_max=1000,
                bias_type="Depolarising",
                bias_prob=0.1,
                renormalise=True,
                silent=True,
            )[0],
            dtype=float,
        )

    base = posterior(error)
    for stabiliser in stabilisers:
        assert np.allclose(
            posterior(multiply_pauli_strings(error, stabiliser)), base, atol=1e-9
        )


def test_depolarising_errors_are_reproducible_from_their_seed():
    """Sampling must honour the supplied generator for every noise model."""
    for model in ("Bitflip", "Depolarising", "Phaseflip"):
        runs = [
            [
                generate_pauli_error_string(
                    8, 0.5, error_model=model, rng=np.random.default_rng(5)
                )
                for _ in range(3)
            ]
            for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2], f"{model} sampling is not reproducible"


@pytest.mark.parametrize("bias", ["Bitflip", "Depolarising"])
def test_decode_custom_matches_exact_posterior_with_two_logical_qubits(bias):
    """The k > 1 path against brute-force enumeration.

    Every other decoder test uses a k = 1 code, so multi-logical marginalisation
    and readout were never checked against an exact reference. The [[4,2,2]]
    code has 2k = 4 logical sites, giving 16 classes that all fit in a brute
    force. Logical-site amplitudes come back in reversed bit order because the
    logical MPS is reversed before readout.
    """
    num_qubits = 4
    stabilisers = ["XXXX", "ZZZZ"]
    logicals_x = ["XXII", "XIXI"]
    logicals_z = ["ZIZI", "ZZII"]
    prob = 0.1

    def to_binary(pauli):
        x = np.array([BITS[c][0] for c in pauli], dtype=int)
        z = np.array([BITS[c][1] for c in pauli], dtype=int)
        return x, z

    # The logicals must form conjugate pairs for the enumeration to mean anything.
    for i, l_x in enumerate(logicals_x):
        a_x, a_z = to_binary(l_x)
        for j, l_z in enumerate(logicals_z):
            b_x, b_z = to_binary(l_z)
            assert (a_x @ b_z + a_z @ b_x) % 2 == (1 if i == j else 0)

    checks_x = np.array([to_binary("XXXX")[0]])
    checks_z = np.array([to_binary("ZZZZ")[1]])
    log_x = np.array([to_binary(l)[0] for l in logicals_x])
    log_z = np.array([to_binary(l)[1] for l in logicals_z])

    def kernel(matrix):
        return [
            np.array(v)
            for v in itertools.product([0, 1], repeat=num_qubits)
            if np.all((matrix @ np.array(v)) % 2 == 0)
        ]

    span_x, span_z = kernel(checks_x), kernel(checks_z)
    labels = ["".join(b) for b in itertools.product("01", repeat=4)]

    if bias == "Depolarising":
        weight = lambda dx, dz: (1 - prob) if (dx == 0 and dz == 0) else prob / 3
    else:
        weight = lambda dx, dz: ((1 - prob) if dx == 0 else prob) * (
            (1 - prob) if dz == 0 else prob
        )

    def reference(error):
        err_x, err_z = to_binary(error)
        amps = dict.fromkeys(labels, 0.0)
        for f_x in span_x:
            d_x = (f_x + err_x) % 2
            l_x = (log_x @ f_x) % 2
            for f_z in span_z:
                d_z = (f_z + err_z) % 2
                l_z = (log_z @ f_z) % 2
                prob_term = 1.0
                for j in range(num_qubits):
                    prob_term *= weight(d_x[j], d_z[j])
                amps["".join(map(str, list(l_x) + list(l_z)))] += np.sqrt(prob_term)
        probs = np.array([amps[k] ** 2 for k in labels])
        return probs / probs.sum()

    reversed_order = [int(format(i, "04b")[::-1], 2) for i in range(16)]

    for error in ("XIII", "ZIII", "YIII", "XZII", "XXII", "ZZII", "XYZI", "YYII"):
        got = _normalised(
            decode_custom(
                stabilisers,
                logicals_x,
                logicals_z,
                error,
                chi_max=int(1e5),
                bias_type=bias,
                bias_prob=prob,
                renormalise=True,
                silent=True,
            )[0]
        )
        assert got.shape == (16,)
        assert np.allclose(got[reversed_order], reference(error), atol=1e-9)


def test_trivial_error_takes_the_fast_path():
    """The no-error shortcut is a deliberate speed optimisation, not a posterior.

    Low-p Monte Carlo is dominated by shots with no error, so both decoders
    return immediately without contracting anything. The vector it hands back is
    a k = 1-shaped stub; callers consume only the success flag. Materialising a
    real 2**(2k)-entry posterior here would allocate 128 MB and cost ~10 ms per
    shot on a k = 12 BB code, on exactly the hot path this exists to skip.

    The verdict is nonetheless correct: for a trivial error the identity class is
    the MAP answer at every p below 1/2, which
    ``test_trivial_error_is_decoded_to_the_identity_class`` checks against the
    exact posterior.
    """
    posterior, success = decode_custom(
        ["XXXX", "ZZZZ"],
        ["XXII", "XIXI"],
        ["ZIZI", "ZZII"],
        "IIII",
        chi_max=int(1e5),
        bias_type="Depolarising",
        bias_prob=0.1,
        renormalise=True,
        silent=True,
    )
    assert success == 1
    assert int(np.argmax(posterior)) == 0

    code = qec.steane_code()
    posterior_css, success_css = decode_css(
        code,
        "I" * len(code),
        chi_max=int(1e5),
        bias_type="Depolarising",
        bias_prob=0.1,
        renormalise=True,
        silent=True,
    )
    assert success_css == 1
    assert int(np.argmax(posterior_css)) == 0


@pytest.mark.parametrize("name", sorted(CODES))
def test_trivial_error_is_decoded_to_the_identity_class(name):
    """What the fast path asserts without computing: identity is the MAP class.

    If this ever failed, the shortcut would be silently scoring real failures as
    successes, so it is the property the optimisation actually rests on.
    """
    code = CODES[name]()
    zero_error = "I" * len(code)
    for prob in (0.05, 0.1, 0.2, 0.3, 0.4, 0.49):
        for bias in ("Bitflip", "Depolarising"):
            posterior = exact_posterior(code, zero_error, bias, prob)
            assert int(np.argmax(posterior)) == 0
            assert posterior[0] > posterior[1:].max()
