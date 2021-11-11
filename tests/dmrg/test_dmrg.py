"""
    Tests for the DMRG optimizer.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from experiments.ising import IsingExact, IsingMPO
from mpopt.mps.explicit import create_product_state
from mpopt.optimizer.dmrg import DMRG as dmrg


def test_ground_states():
    """
    Test how DMRG finds the ground state of a 1D Ising model.
    Check that physical observables are correct.
    """

    for _ in range(10):

        number_of_sites = np.random.randint(3, 11)
        transverse_magnetic_field = np.random.uniform(0.1, 1)

        ising_exact = IsingExact(number_of_sites, transverse_magnetic_field)
        ising_mpo = IsingMPO(number_of_sites, transverse_magnetic_field)
        ham_mpo = ising_mpo.hamiltonian_mpo()
        ham_exact = ising_exact.hamiltonian_dense()

        mps_start = create_product_state(number_of_sites, which="0")

        engine = dmrg(mps_start.copy(), ham_mpo, chi_max=64, cut=1e-14, mode="SA")
        engine.run(10)
        ground_state_mps = engine.mps
        ground_state_exact = eigsh(ham_exact, k=6)[1][:, 0]

        assert np.isclose(
            abs(ground_state_mps.to_dense()), abs(ground_state_exact)
        ).all()
        assert np.isclose(
            sum(
                np.array(
                    [
                        ising_exact.x_magnetization(i, ground_state_exact)
                        for i in range(number_of_sites)
                    ]
                )
                - np.array(
                    [
                        ising_mpo.x_magnetization(i, ground_state_mps)
                        for i in range(number_of_sites)
                    ]
                )
            ),
            0,
            atol=1e-7,
        )
        assert np.isclose(
            sum(
                np.array(
                    [
                        ising_exact.z_magnetization(i, ground_state_exact)
                        for i in range(number_of_sites)
                    ]
                )
                - np.array(
                    [
                        ising_mpo.z_magnetization(i, ground_state_mps)
                        for i in range(number_of_sites)
                    ]
                )
            ),
            0,
            atol=1e-7,
        )
