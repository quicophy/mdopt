"""Tests for the ``mdopt.optimiser.dmrg`` module."""

import pytest
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import ArpackError

from mdopt.examples.ising.ising import IsingExact, IsingMPO
from mdopt.mps.utils import create_simple_product_state
from mdopt.optimiser.dmrg import DMRG as dmrg
from mdopt.optimiser.dmrg import EffectiveOperator, _nonzero_start_vector


def test_optimiser_effective_operator():
    """Test for the ``__init__`` method of the ``EffectiveOperator`` class."""

    left_environment = np.random.uniform(low=0, high=1, size=(2, 3, 2))
    mpo_tensor_left = np.random.uniform(low=0, high=1, size=(3, 3, 2, 2))
    mpo_tensor_right = np.random.uniform(low=0, high=1, size=(3, 3, 2, 2))
    right_environment = np.random.uniform(low=0, high=1, size=(2, 3, 2))

    EffectiveOperator(
        left_environment=left_environment,
        mpo_tensor_left=mpo_tensor_left,
        mpo_tensor_right=mpo_tensor_right,
        right_environment=right_environment,
    )

    with pytest.raises(ValueError):
        EffectiveOperator(
            left_environment=np.expand_dims(left_environment, 0),
            mpo_tensor_left=mpo_tensor_left,
            mpo_tensor_right=mpo_tensor_right,
            right_environment=right_environment,
        )
    with pytest.raises(ValueError):
        EffectiveOperator(
            left_environment=left_environment,
            mpo_tensor_left=np.expand_dims(mpo_tensor_left, 0),
            mpo_tensor_right=mpo_tensor_right,
            right_environment=right_environment,
        )
    with pytest.raises(ValueError):
        EffectiveOperator(
            left_environment=left_environment,
            mpo_tensor_left=mpo_tensor_left,
            mpo_tensor_right=np.expand_dims(mpo_tensor_right, 0),
            right_environment=right_environment,
        )
    with pytest.raises(ValueError):
        EffectiveOperator(
            left_environment=left_environment,
            mpo_tensor_left=mpo_tensor_left,
            mpo_tensor_right=mpo_tensor_right,
            right_environment=np.expand_dims(right_environment, 0),
        )


def test_optimiser_ground_states():
    """
    Test how DMRG finds the ground state of a 1D Ising model.
    Check that physical observables are correct and the MPS ground state
    corresponds to the one given by virtue of exact diagonalisation.
    """

    for _ in range(5):
        num_sites = 8
        num_runs = 5
        transverse_magnetic_field = np.random.uniform(0.1, 1)

        ising_exact = IsingExact(num_sites, transverse_magnetic_field)
        ising_mpo = IsingMPO(num_sites, transverse_magnetic_field)
        ham_mpo = ising_mpo.hamiltonian_mpo()
        ham_exact = ising_exact.hamiltonian_dense()

        mps_start = create_simple_product_state(num_sites, which="0", form="Explicit")

        engine = dmrg(mps_start, ham_mpo)
        engine.run(num_runs)
        ground_state_mps = engine.mps
        ground_state_exact = eigsh(ham_exact, k=6)[1][:, 0]

        assert np.allclose(
            abs(ground_state_mps.dense()),
            abs(ground_state_exact),
            atol=1e-6,
        )
        assert np.allclose(
            np.array(
                [
                    ising_exact.x_magnetisation(i, ground_state_exact)
                    for i in range(num_sites)
                ]
            ),
            np.array(
                [
                    ising_mpo.x_magnetisation(i, ground_state_mps)
                    for i in range(num_sites)
                ]
            ),
            atol=1e-3,
        )
        assert np.allclose(
            np.array(
                [
                    ising_exact.z_magnetisation(i, ground_state_exact)
                    for i in range(num_sites)
                ]
            ),
            np.array(
                [
                    ising_mpo.z_magnetisation(i, ground_state_mps)
                    for i in range(num_sites)
                ]
            ),
            atol=1e-3,
        )


def test_nonzero_start_vector_replaces_a_zero_guess():
    """ARPACK refuses a zero v0; the fallback must be a valid unit vector.

    An underflowed two-site tensor produced "ARPACK error -9: Starting vector is
    zero" 4.6 hours into a full-scale classical_ldpc run at chi_max=128, so this
    is not limited to the very small bond dimensions where it was first seen.
    """
    dimension = 8

    zero = np.zeros(dimension, dtype=float)
    replaced = _nonzero_start_vector(zero, dimension)
    assert np.isclose(np.linalg.norm(replaced), 1.0)

    # A usable guess is handed back untouched.
    good = np.arange(1.0, dimension + 1.0)
    assert _nonzero_start_vector(good, dimension) is good

    # Non-finite guesses are replaced too: ARPACK cannot start from NaN either.
    nan = np.full(dimension, np.nan)
    assert np.all(np.isfinite(_nonzero_start_vector(nan, dimension)))


def test_eigsh_rejects_a_zero_start_vector():
    """Pin the failure mode the guard exists for.

    If scipy ever stops raising on a zero v0 this test goes green for the wrong
    reason, so it asserts the error explicitly rather than trusting the guard.
    """
    matrix = np.diag(np.arange(1.0, 9.0))
    dimension = matrix.shape[0]
    zero = np.zeros(dimension)

    with pytest.raises(ArpackError):
        eigsh(matrix, k=1, which="SA", v0=zero, return_eigenvectors=True)

    # The same call succeeds once the guess is routed through the guard.
    _, vectors = eigsh(
        matrix,
        k=1,
        which="SA",
        v0=_nonzero_start_vector(zero, dimension),
        return_eigenvectors=True,
    )
    assert vectors.shape == (dimension, 1)
