"""
Tests for the DephasingDMRG optimiser.
"""

import numpy as np

from mpopt.mps.explicit import (
    create_simple_product_state,
    create_custom_product_state,
    mps_from_dense,
)
from mpopt.mps.canonical import inner_product
from mpopt.optimiser.dephasing_dmrg import DephasingDMRG as deph_dmrg
from mpopt.optimiser.dmrg import DMRG as dmrg
from tests.mps.test_explicit import _create_psi


def test_main_component():
    """
    Test the dephasing DMRG method with the main component problem.
    We solve the problem using exact diagonalisation, DMRG and dephasing DMRG.
    Next, we compare the solutions which should be exactly the same.
    """

    for _ in range(5):

        num_sites = 8
        num_runs = 5

        # Creating a random pure complex state and its MPS version.
        psi = _create_psi(num_sites)

        # Bumping up the main component amplitude and renormalising the state.
        index_to_bump = np.random.randint(0, 2**num_sites)
        psi[index_to_bump] = 10
        psi /= np.linalg.norm(psi)

        # Creating the exact MPS version of the state.
        mps = mps_from_dense(psi)

        # Creating the matrix product density operator.
        mpdo = mps.density_mpo()

        # Finding the main component (a computational basis state having the largest overlap)
        # of the density matrix in the dense form.
        overlaps_exact = []
        for i in range(2**num_sites):
            state_string = np.binary_repr(i, width=num_sites)
            overlaps_exact.append(
                np.absolute(create_custom_product_state(state_string).to_dense() @ psi)
                ** 2
            )
        main_component_exact = np.argmax(overlaps_exact)

        # Finding the main component of the MPDO using DMRG
        mps_start = create_simple_product_state(num_sites, which="+")
        engine = dmrg(
            mps_start, mpdo, chi_max=1024, cut=1e-14, mode="LA", copy=True, silent=True
        )
        engine.run(num_runs)
        max_excited_mps_from_dmrg = engine.mps

        overlaps_dmrg = []
        for i in range(2**num_sites):
            state_string = np.binary_repr(i, width=num_sites)
            overlaps_dmrg.append(
                np.absolute(
                    inner_product(
                        max_excited_mps_from_dmrg.to_right_canonical(),
                        create_custom_product_state(state_string).to_right_canonical(),
                    )
                )
                ** 2
            )
        main_component_dmrg = np.argmax(overlaps_dmrg)

        # Finding the main component of the MPDO using dephasing DMRG
        mps_start = create_simple_product_state(num_sites, which="+")
        dephasing_engine = deph_dmrg(
            mps_start,
            mps.to_right_canonical(),
            chi_max=1024,
            cut=1e-14,
            mode="LA",
            copy=True,
            silent=True,
        )
        dephasing_engine.run(num_runs)
        main_component_mps = dephasing_engine.mps

        overlaps_dephased = []
        for i in range(2**num_sites):
            state_string = np.binary_repr(i, width=num_sites)
            overlaps_dephased.append(
                np.absolute(
                    inner_product(
                        main_component_mps.to_right_canonical(),
                        create_custom_product_state(state_string).to_right_canonical(),
                    )
                )
                ** 2
            )
        main_component_dephased = np.argmax(overlaps_dephased)

        # All the three answers must be the same.
        assert np.logical_and(
            main_component_exact == main_component_dmrg,
            main_component_exact == main_component_dephased,
        )
