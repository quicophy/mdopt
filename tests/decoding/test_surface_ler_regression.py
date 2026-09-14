"""Logical-error-rate regression on the distance-5 surface code.

A fixed seed makes the error sample deterministic, so the number of decoding
failures is a property of the code, not a Monte Carlo estimate, and "the
logical error rate must not increase" is an exact check. The bond dimension
is deliberately small: at ``chi_max=16`` the decoder is not converged, so
the verdicts depend on how the truncation is carried out, and a change to
the SVD, the orthogonality-centre moves or the contractor that alters the
truncation moves them. The pinned values were produced by ``main`` and by
the optimised branch alike (c5d2773 / 0ba5f0b, 2026-09-11).
"""

import numpy as np
import qecstruct as qec

from mdopt.decoding.decoding import decode_css, generate_pauli_error_string

SEED = 2026
SHOTS = 40
CHI_MAX = 16


def _surface_code(lattice_size):
    rep = qec.repetition_code(lattice_size)
    return qec.hypergraph_product(rep, rep)


def _verdicts(code, error_rate):
    rng = np.random.default_rng(SEED)
    verdicts = []
    for _ in range(SHOTS):
        error = generate_pauli_error_string(
            len(code), error_rate, error_model="Depolarising", rng=rng
        )
        _, success = decode_css(
            code,
            error,
            chi_max=CHI_MAX,
            bias_type="Depolarising",
            bias_prob=error_rate,
            renormalise=True,
            silent=True,
            contraction_strategy="Optimised",
            qubit_order_strategy="Natural",
        )
        verdicts.append(int(success))
    return "".join(map(str, verdicts))


def test_surface_5_low_error_rate_decodes_every_shot():
    """At p=0.05 (well below threshold) none of the 40 shots fails."""
    verdicts = _verdicts(_surface_code(5), 0.05)
    failures = verdicts.count("0")
    assert failures == 0, f"{failures} of {SHOTS} shots failed: {verdicts}"


def test_surface_5_truncated_ler_does_not_increase():
    """At p=0.08 and chi_max=16 the pinned sample has 4 failures.

    The count must not grow. The pattern is pinned too: a different pattern
    with the same or a lower count means the truncation changed, which is
    worth a review even when it looks like an improvement; update the pinned
    string deliberately after that review.
    """
    pinned = "1111111111111111011011101111111111011111"
    verdicts = _verdicts(_surface_code(5), 0.08)
    failures = verdicts.count("0")
    assert failures <= pinned.count(
        "0"
    ), f"{failures} failures, pinned {pinned.count('0')}: {verdicts}"
    assert (
        verdicts == pinned
    ), f"verdict pattern changed:\n got    {verdicts}\n pinned {pinned}"
