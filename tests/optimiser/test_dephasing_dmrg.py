"""Tests for the ``mdopt.optimiser.dephasing_dmrg`` module."""

import pytest
import numpy as np

from mdopt.optimiser.dephasing_dmrg import DephasingDMRG as deph_dmrg
from mdopt.optimiser.dmrg import DMRG as dmrg
from mdopt.optimiser.dephasing_dmrg import EffectiveDensityOperator
from mdopt.mps.utils import (
    create_state_vector,
    create_simple_product_state,
    create_custom_product_state,
    mps_from_dense,
    inner_product,
)


def test_optimiser_effective_density_operator():
    """Test for the ``__init__`` method of the ``EffectiveDensityOperator`` class."""

    left_environment = np.random.uniform(low=0, high=1, size=(2, 2, 2, 2))
    mps_target_1 = np.random.uniform(low=0, high=1, size=(2, 2, 2))
    mps_target_2 = np.random.uniform(low=0, high=1, size=(2, 2, 2))
    right_environment = np.random.uniform(low=0, high=1, size=(2, 2, 2, 2))

    EffectiveDensityOperator(
        left_environment=left_environment,
        mps_target_1=mps_target_1,
        mps_target_2=mps_target_2,
        right_environment=right_environment,
    )

    with pytest.raises(ValueError):
        EffectiveDensityOperator(
            left_environment=np.expand_dims(left_environment, 0),
            mps_target_1=mps_target_1,
            mps_target_2=mps_target_2,
            right_environment=right_environment,
        )
    with pytest.raises(ValueError):
        EffectiveDensityOperator(
            left_environment=left_environment,
            mps_target_1=np.expand_dims(mps_target_1, 0),
            mps_target_2=mps_target_2,
            right_environment=right_environment,
        )
    with pytest.raises(ValueError):
        EffectiveDensityOperator(
            left_environment=left_environment,
            mps_target_1=mps_target_1,
            mps_target_2=np.expand_dims(mps_target_2, 0),
            right_environment=right_environment,
        )
    with pytest.raises(ValueError):
        EffectiveDensityOperator(
            left_environment=left_environment,
            mps_target_1=mps_target_1,
            mps_target_2=mps_target_2,
            right_environment=np.expand_dims(right_environment, 0),
        )


def test_optimiser_main_component():
    """
    Test the dephasing DMRG optimiser with the main component problem.
    We solve the problem using exact diagonalisation, DMRG and dephasing DMRG.
    Next, we compare the solutions which should be exactly the same.
    """

    for _ in range(5):

        num_sites = 8
        num_runs = 1

        # Creating a random pure complex state and its MPS version.
        psi = create_state_vector(num_sites)

        # Bumping up the main component amplitude and renormalising the state.
        index_to_bump = np.random.randint(0, 2**num_sites)
        psi[index_to_bump] = 10
        psi /= np.linalg.norm(psi)

        # Creating the exact MPS version of the state.
        mps = mps_from_dense(psi, form="Right-canonical")

        # Creating the matrix density product operator.
        mdpo = mps.density_mpo()

        # Finding the main component (a computational basis state having the largest overlap)
        # of the density matrix in the dense form.
        overlaps_exact = []
        for i in range(2**num_sites):
            state_string = np.binary_repr(i, width=num_sites)
            overlaps_exact.append(
                np.absolute(create_custom_product_state(state_string).dense() @ psi)
                ** 2
            )
        main_component_exact = np.argmax(overlaps_exact)

        # Finding the main component of the MDPO using DMRG.
        mps_start = create_simple_product_state(num_sites, which="+")
        engine = dmrg(
            mps_start, mdpo, chi_max=1e4, cut=1e-12, mode="LA", copy=True, silent=True
        )
        engine.run(num_runs)
        max_excited_mps_from_dmrg = engine.mps

        overlaps_dmrg = []
        for i in range(2**num_sites):
            state_string = np.binary_repr(i, width=num_sites)
            overlaps_dmrg.append(
                np.absolute(
                    inner_product(
                        max_excited_mps_from_dmrg,
                        create_custom_product_state(state_string),
                    )
                )
                ** 2
            )
        main_component_dmrg = np.argmax(overlaps_dmrg)

        # Finding the main component of the MDPO using dephasing DMRG.
        mps_start = create_simple_product_state(num_sites, which="+")
        dephasing_engine = deph_dmrg(
            mps_start,
            mps.right_canonical(),
            chi_max=1e4,
            cut=1e-12,
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
                        main_component_mps,
                        create_custom_product_state(state_string),
                    )
                )
                ** 2
            )
        main_component_dephased = np.argmax(overlaps_dephased)

        # Check the answer from the dephasing DMRG is a product state.
        mps_product_answer = dephasing_engine.mps
        assert mps_product_answer.bond_dimensions == [
            1 for _ in range(mps_product_answer.num_bonds)
        ]

        # Check that all the three answers are the same.
        assert np.logical_and(
            main_component_exact == main_component_dmrg,
            main_component_exact == main_component_dephased,
        )
