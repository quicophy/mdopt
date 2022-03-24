"""
    Tests for the MPS-MPO contractor.
"""

import numpy as np
from opt_einsum import contract
from scipy.stats import unitary_group
from mpopt.utils.utils import mpo_to_matrix, create_random_mpo
from mpopt.mps.explicit import mps_from_dense
from mpopt.mps.canonical import is_canonical, to_dense
from mpopt.contractor.contractor import (
    mps_mpo_contract,
    apply_two_site_unitary,
    apply_one_site_unitary,
)
from tests.mps.test_explicit import _create_psi


def test_mps_mpo_contract():
    """
    Test the implementation of the `mps_mpo_contract` function.
    """

    num_sites = np.random.randint(4, 9)
    phys_dim = 2
    identity = np.eye(2).reshape((1, 1, 2, 2))

    for _ in range(100):

        psi_init = _create_psi(num_sites)
        start_site = np.random.randint(0, num_sites - 1)
        mps_init = mps_from_dense(psi_init, phys_dim=phys_dim).to_mixed_canonical(
            start_site
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
        orthogonality_centre = mps_fin_1[int(start_site + mpo_length - 1)]

        mpo_dense = mpo_to_matrix(full_mpo, interlace=False, group=True)
        psi_fin = mpo_dense @ psi_init

        assert is_canonical(mps_fin)
        assert np.isclose(
            abs(np.linalg.norm(to_dense(mps_fin) - psi_fin)), 0, atol=1e-7
        )
        assert np.isclose(np.linalg.norm(orthogonality_centre), 1)


def test_apply_two_site_unitary():
    """
    Test the implementation of the `apply_two_site_unitary` function.
    """

    identity = np.eye(2)
    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)

        mps = mps_from_dense(psi)
        mps_right = mps.to_right_canonical()
        mps_new = mps_right.copy()

        site = int(np.random.randint(mps_length - 1))

        unitary_exact = unitary_group.rvs(4)
        unitary_tensor = unitary_exact.reshape((2, 2, 2, 2))

        for _ in range(site):
            unitary_exact = np.kron(identity, unitary_exact)
        for _ in range(mps_length - site - 2):
            unitary_exact = np.kron(unitary_exact, identity)
        unitary_exact = unitary_exact.transpose()

        mps_new[site], mps_new[site + 1] = apply_two_site_unitary(
            lambda_0=mps.singular_values[site],
            b_1=mps_right[site],
            b_2=mps_right[site + 1],
            unitary=unitary_tensor,
        )

        assert is_canonical(mps_new)

        assert np.isclose(
            abs(
                np.dot(
                    np.conj(contract("ij, j", unitary_exact, psi, optimize=[(0, 1)])),
                    to_dense(mps_new),
                )
            )
            - 1,
            0,
        )

        assert np.isclose(
            contract("ij, j", unitary_exact, psi, optimize=[(0, 1)]), to_dense(mps_new)
        ).all()


def test_apply_one_site_unitary():
    """
    Test the implementation of the `apply_one_site_unitary` function.
    """

    mps_length = np.random.randint(4, 9)

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    identity = np.eye(2)
    paulis = [pauli_x, pauli_y, pauli_z]

    for _ in range(100):

        psi = _create_psi(mps_length)

        mps = mps_from_dense(psi)
        mps_right = mps.to_right_canonical()
        mps_new = mps_right.copy()

        site = int(np.random.randint(mps_length))

        operator_index = int(np.random.randint(3))
        unitary_tensor = paulis[operator_index]

        unitary_exact = unitary_tensor
        for _ in range(site):
            unitary_exact = np.kron(identity, unitary_exact)
        for _ in range(mps_length - site - 1):
            unitary_exact = np.kron(unitary_exact, identity)
        unitary_exact = unitary_exact.transpose()

        mps_new[site] = apply_one_site_unitary(
            t_1=mps_right[site],
            unitary=unitary_tensor,
        )

        assert is_canonical(mps_new)

        assert np.isclose(
            abs(
                np.dot(
                    np.conj(contract("ij, j", unitary_exact, psi, optimize=[(0, 1)])),
                    to_dense(mps_new),
                )
            )
            - 1,
            0,
        )

        assert np.isclose(
            contract("ij, j", unitary_exact, psi, optimize=[(0, 1)]), to_dense(mps_new)
        ).all()
