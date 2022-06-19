"""
Tests for the DMRG optimiser.
"""

import numpy as np
from scipy.sparse.linalg import eigsh

from experiments.ising import IsingExact, IsingMPO
from mpopt.mps.explicit import create_simple_product_state
from mpopt.optimiser.dmrg import DMRG as dmrg


def test_ground_states():
    """
    Test how DMRG finds the ground state of a 1D Ising model.
    Check that physical observables are correct and the MPS ground state
    corresponds to the one from exact diagonalisation.
    """

    for _ in range(5):

        num_sites = 8
        transverse_magnetic_field = np.random.uniform(0.1, 1)

        ising_exact = IsingExact(num_sites, transverse_magnetic_field)
        ising_mpo = IsingMPO(num_sites, transverse_magnetic_field)
        ham_mpo = ising_mpo.hamiltonian_mpo()
        ham_exact = ising_exact.hamiltonian_dense()

        mps_start = create_simple_product_state(num_sites, which="0")

        engine = dmrg(mps_start, ham_mpo, chi_max=64, cut=1e-14, mode="SA")
        engine.run(10)
        ground_state_mps = engine.mps
        ground_state_exact = eigsh(ham_exact, k=6)[1][:, 0]

        assert np.allclose(
            abs(ground_state_mps.to_dense()),
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
