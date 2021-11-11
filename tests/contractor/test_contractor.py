"""
    Tests for the MPS-MPO contractor.
"""

import numpy as np
from opt_einsum import contract
from mpopt.mps.explicit import mps_from_dense, is_canonical, to_dense
from mpopt.contractor.contractor import apply_one_site_unitary, apply_two_site_unitary
from tests.mps.test_explicit import _create_psi


def test_apply_two_site_unitary():
    """
    Test the implementation of the apply_two_site_unitary function.
    """

    mps_length = np.random.randint(4, 9)

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_y = np.array([[0.0, -1j], [1j, 0.0]])
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    identity = np.identity(2)
    paulis = [pauli_x, pauli_y, pauli_z]

    for _ in range(100):

        psi = _create_psi(mps_length)

        mps = mps_from_dense(psi)
        mps_right = mps.to_right_canonical()
        mps_new = mps_right.copy()

        site = int(np.random.randint(mps_length - 1))

        operator_index = np.random.randint(3, size=(2,))

        unitary_tensor = contract(
            "ij, kl -> ikjl", paulis[operator_index[0]], paulis[operator_index[1]]
        )

        unitary_exact = unitary_tensor.reshape((4, 4))
        for _ in range(site):
            unitary_exact = np.kron(identity, unitary_exact)
        for _ in range(mps_length - site - 2):
            unitary_exact = np.kron(unitary_exact, identity)
        unitary_exact = unitary_exact.transpose()

        mps_new[site], mps_new[site + 1] = apply_two_site_unitary(
            lambda_0=mps.schmidt_values[site],
            b_1=mps_right[site],
            b_2=mps_right[site + 1],
            unitary=unitary_tensor,
        )

        assert is_canonical(mps_new)

        assert np.isclose(
            abs(
                np.dot(
                    np.conj(contract("ij, j", unitary_exact, psi)), to_dense(mps_new)
                )
            )
            - 1,
            0,
        )

        assert np.isclose(
            contract("ij, j", unitary_exact, psi), to_dense(mps_new)
        ).all()


def test_apply_one_site_unitary():
    """
    Test the implementation of the apply_one_site_unitary function.
    """

    mps_length = np.random.randint(4, 9)

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_y = np.array([[0.0, -1j], [1j, 0.0]])
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    identity = np.identity(2)
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
            b_1=mps_right[site],
            unitary=unitary_tensor,
        )

        assert is_canonical(mps_new)

        assert np.isclose(
            abs(
                np.dot(
                    np.conj(contract("ij, j", unitary_exact, psi)), to_dense(mps_new)
                )
            )
            - 1,
            0,
        )

        assert np.isclose(
            contract("ij, j", unitary_exact, psi), to_dense(mps_new)
        ).all()
