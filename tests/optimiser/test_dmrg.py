"""Tests for the ``mdopt.optimiser.dmrg`` module."""

import pytest
import numpy as np
from scipy.sparse.linalg import eigsh

from examples.ising.ising import IsingExact, IsingMPO
from mdopt.mps.utils import create_simple_product_state
from mdopt.optimiser.dmrg import DMRG as dmrg
from mdopt.optimiser.dmrg import EffectiveOperator


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
