"""Slow decoding regression tests in the regime where truncation decides.

These run the decoders at production-size bond dimensions on instances whose
verdict is known to be sensitive to the numerical details of the truncation
(a wrong singular-vector basis, a corrupted factorisation, a changed
tie-break) even when every cheap unit test and every fingerprint in
``benchmarks/`` passes. They take tens of minutes each and are skipped
unless ``MDOPT_RUN_SLOW=1`` is set; run them before merging any change to
the SVD, the orthogonality-centre moves or the contractor.
"""

import os

import numpy as np
import pytest

from mdopt.decoding.decoding import create_bb_code, decode_css

pytestmark = pytest.mark.skipif(
    os.environ.get("MDOPT_RUN_SLOW") != "1",
    reason="slow decoding regression test; set MDOPT_RUN_SLOW=1 to run",
)


def test_bb_72_12_6_natural_order_single_error_converges_in_chi():
    """A single Z error on the [[72,12,6]] code decodes to the identity at
    chi_max=400 in the natural qubit order, and the verdict is the same at
    chi_max=128.

    This instance exposed the QR pre-reduction of ``svd`` (now off by
    default): with it, the decode returned a flat or wrongly peaked
    posterior at chi_max=400 on NumPy/Accelerate builds -- 24 of 24 shots
    -- while every unit test and benchmark fingerprint still passed. Each
    chi_max=400 decode takes 30-60 minutes on a laptop.
    """
    code = create_bb_code(6, 6, "x**3 + y + y**2", "y**3 + x + x**2")
    error = "I" * 58 + "Z" + "I" * 13
    assert len(error) == len(code)
    peaks = {}
    for chi_max in (128, 400):
        posterior, success = decode_css(
            code,
            error,
            chi_max=chi_max,
            bias_type="Depolarising",
            bias_prob=0.01,
            renormalise=True,
            silent=True,
            contraction_strategy="Optimised",
            qubit_order_strategy="Natural",
        )
        posterior = np.asarray(posterior, dtype=float).ravel()
        assert success == 1.0, f"chi_max={chi_max}: the identity class lost"
        assert int(np.argmax(posterior)) == 0
        peaks[chi_max] = float(posterior.max())
        # A converged posterior for a single low-weight error is a delta.
        assert peaks[chi_max] > 0.99, f"chi_max={chi_max}: peak {peaks[chi_max]}"
