"""Tests for the :module:`contractor` module."""

import pytest
import numpy as np
from opt_einsum import contract
from scipy.stats import unitary_group

from mdopt.contractor.contractor import (
    apply_one_site_operator,
    apply_two_site_unitary,
    mps_mpo_contract,
)
from mdopt.mps.utils import is_canonical, mps_from_dense, create_state_vector
from mdopt.utils.utils import create_random_mpo, mpo_to_matrix


def test_contractor_apply_one_site_operator():
    r"""Test for the :func:`apply_one_site_operator` function."""

    num_sites = np.random.randint(4, 9)

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    identity = np.eye(2)
    paulis = [pauli_x, pauli_y, pauli_z]

    for _ in range(10):

        psi = create_state_vector(num_sites)

        mps = mps_from_dense(psi, form="Explicit")
        mps_right = mps.right_canonical()
        mps_new = mps_right.copy()

        site = int(np.random.randint(num_sites))

        operator_index = int(np.random.randint(3))
        unitary_tensor = paulis[operator_index]

        unitary_exact = unitary_tensor
        for _ in range(site):
            unitary_exact = np.kron(identity, unitary_exact)
        for _ in range(num_sites - site - 1):
            unitary_exact = np.kron(unitary_exact, identity)
        unitary_exact = unitary_exact.transpose()

        mps_new.tensors[site] = apply_one_site_operator(
            tensor=mps_right.tensors[site],
            operator=unitary_tensor,
        )

        with pytest.raises(ValueError):
            apply_one_site_operator(
                tensor=np.expand_dims(mps_right.tensors[site], axis=0),
                operator=unitary_tensor,
            )
        with pytest.raises(ValueError):
            apply_one_site_operator(
                tensor=mps_right.tensors[site],
                operator=np.expand_dims(unitary_tensor, axis=0),
            )

        assert is_canonical(mps_new)

        assert np.isclose(
            abs(
                np.dot(
                    np.conjugate(
                        contract("ij, j", unitary_exact, psi, optimize=[(0, 1)])
                    ),
                    mps_new.dense(),
                )
            )
            - 1,
            0,
        )

        assert np.isclose(
            contract("ij, j", unitary_exact, psi, optimize=[(0, 1)]), mps_new.dense()
        ).all()


def test_contractor_apply_two_site_unitary():
    r"""Test for the :func:`apply_two_site_unitary` function."""

    identity = np.eye(2)
    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)

        mps = mps_from_dense(psi)
        mps_right = mps.right_canonical()
        mps_new = mps_right.copy()

        site = int(np.random.randint(num_sites - 1))

        unitary_exact = unitary_group.rvs(4)
        unitary_tensor = unitary_exact.reshape((2, 2, 2, 2))

        for _ in range(site):
            unitary_exact = np.kron(identity, unitary_exact)
        for _ in range(num_sites - site - 2):
            unitary_exact = np.kron(unitary_exact, identity)
        unitary_exact = unitary_exact.transpose()

        mps_new.tensors[site], mps_new.tensors[site + 1] = apply_two_site_unitary(
            lambda_0=mps.singular_values[site],
            b_1=mps_right.tensors[site],
            b_2=mps_right.tensors[site + 1],
            unitary=unitary_tensor,
        )

        with pytest.raises(ValueError):
            apply_two_site_unitary(
                lambda_0=mps.singular_values[site],
                b_1=np.expand_dims(mps_right.tensors[site], axis=0),
                b_2=mps_right.tensors[site + 1],
                unitary=unitary_tensor,
            )
        with pytest.raises(ValueError):
            apply_two_site_unitary(
                lambda_0=mps.singular_values[site],
                b_1=mps_right.tensors[site],
                b_2=np.expand_dims(mps_right.tensors[site + 1], axis=0),
                unitary=unitary_tensor,
            )
        with pytest.raises(ValueError):
            apply_two_site_unitary(
                lambda_0=mps.singular_values[site],
                b_1=mps_right.tensors[site],
                b_2=mps_right.tensors[site + 1],
                unitary=np.expand_dims(unitary_tensor, axis=0),
            )

        assert is_canonical(mps_new)

        assert np.isclose(
            abs(
                np.dot(
                    np.conjugate(
                        contract("ij, j", unitary_exact, psi, optimize=[(0, 1)])
                    ),
                    mps_new.dense(),
                )
            )
            - 1,
            0,
        )

        assert np.isclose(
            contract("ij, j", unitary_exact, psi, optimize=[(0, 1)]), mps_new.dense()
        ).all()


def test_contractor_mps_mpo_contract():
    r"""Test for the :func:`mps_mpo_contract` function."""

    num_sites = np.random.randint(4, 9)
    phys_dim = 2
    identity = np.eye(2).reshape((1, 1, 2, 2))

    for _ in range(10):

        psi_init = create_state_vector(num_sites)
        start_site = np.random.randint(0, num_sites - 1)
        mps_init = mps_from_dense(
            psi_init, phys_dim=phys_dim, form="Mixed-canonical", orth_centre=start_site
        )

        mpo_length = np.random.randint(2, num_sites - start_site + 1)
        dims_unique = np.random.randint(3, 10, size=mpo_length - 1)
        mpo = create_random_mpo(
            mpo_length,
            dims_unique,
            phys_dim=phys_dim,
            which=np.random.choice(["uniform", "normal", "randint"], size=1),
        )

        identities_l = [identity for _ in range(start_site)]
        identities_r = [identity for _ in range(num_sites - mpo_length - start_site)]
        full_mpo = identities_l + mpo + identities_r

        mps_fin = mps_mpo_contract(mps_init, mpo, start_site, renormalise=False)
        mps_fin_1 = mps_mpo_contract(mps_init, mpo, start_site, renormalise=True)
        orthogonality_centre = mps_fin_1.tensors[int(start_site + mpo_length - 1)]

        mpo_dense = mpo_to_matrix(full_mpo, interlace=False, group=True)
        psi_fin = mpo_dense @ psi_init

        with pytest.raises(ValueError):
            mps_mpo_contract(mps_init, mpo, 100)
        with pytest.raises(ValueError):
            mpo[0] = np.zeros(
                (
                    2,
                    2,
                    2,
                    2,
                )
            )
            mps_mpo_contract(
                mps_init,
                mpo,
                start_site,
            )

        assert is_canonical(mps_fin)
        assert np.isclose(abs(np.linalg.norm(mps_fin.dense() - psi_fin)), 0, atol=1e-7)
        assert np.isclose(np.linalg.norm(orthogonality_centre), 1)
