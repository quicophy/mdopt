"""
    Tests for the explicit MPS construction.
"""

from functools import reduce

import numpy as np
from opt_einsum import contract

from mpopt.mps.canonical import (
    find_orth_centre,
    inner_product,
    is_canonical,
    to_explicit,
)
from mpopt.mps.explicit import (
    create_custom_product_state,
    create_simple_product_state,
    mps_from_dense,
)


def _create_psi(length):
    """
    A helper function which creates a random quantum state in the form of a state vector.
    """

    psi = np.random.uniform(size=(2**length)) + 1j * np.random.uniform(
        size=(2**length)
    )
    psi /= np.linalg.norm(psi)

    return psi


def test_from_dense():
    """
    Test of the implementation of the `mps_from_dense` function.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)

        mps = mps_from_dense(psi)
        psi_from_mps = mps.to_dense().reshape((2**mps_length))

        overlap = abs(np.conjugate(psi_from_mps) @ psi) ** 2

        assert np.isclose(overlap, 1)


def test_single_site_left_iso():
    """
    Test of the implementation of the `single_site_left_iso` method.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        for site in range(mps_length):
            isometry = mps.single_site_left_iso(site)

            to_be_identity = contract(
                "ijk, ijl -> kl", isometry, np.conjugate(isometry), optimize=[(0, 1)]
            )

            assert np.isclose(
                np.linalg.norm(to_be_identity - np.identity(to_be_identity.shape[0])), 0
            )


def test_to_left_canonical():
    """
    Test of the implementation of the `to_left_canonical` method.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        mps_left = mps.to_left_canonical()

        assert is_canonical(mps_left)
        assert np.isclose(abs(inner_product(mps_left, mps_left)), 1)
        assert len(find_orth_centre(mps_left)) == 1

        for i in range(mps_length):
            assert mps.tensors[i].shape == mps_left[i].shape

        for i, _ in enumerate(mps_left):

            to_be_identity_left = contract(
                "ijk, ijl -> kl",
                mps_left[i],
                np.conjugate(mps_left[i]),
                optimize=[(0, 1)],
            )

            identity_left = np.identity(to_be_identity_left.shape[0], dtype=np.float64)

            assert np.isclose(np.linalg.norm(to_be_identity_left - identity_left), 0)


def test_single_site_right_iso():
    """
    Test of the implementation of the `single_site_right_iso` method.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        for site in range(mps_length):

            isometry = mps.single_site_right_iso(site)

            to_be_identity = contract(
                "ijk, ljk -> il", isometry, np.conjugate(isometry), optimize=[(0, 1)]
            )

            assert np.isclose(
                np.linalg.norm(to_be_identity - np.identity(to_be_identity.shape[0])), 0
            )


def test_to_right_canonical():
    """
    Test of the implementation of the `to_right_canonical` method.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        mps_right = mps.to_right_canonical()

        assert is_canonical(mps_right)
        assert np.isclose(abs(inner_product(mps_right, mps_right)), 1)
        assert len(find_orth_centre(mps_right)) == 1

        for i in range(mps_length):
            assert mps.tensors[i].shape == mps_right[i].shape

        for i, _ in enumerate(mps_right):

            to_be_identity_right = contract(
                "ijk, ljk -> il",
                mps_right[i],
                np.conjugate(mps_right[i]),
                optimize=[(0, 1)],
            )

            identity_right = np.identity(
                to_be_identity_right.shape[0], dtype=np.float64
            )

            assert np.isclose(np.linalg.norm(to_be_identity_right - identity_right), 0)


def test_to_mixed_canonical():
    """
    Test of the implementation of the `to_mixed_canonical` method.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        orth_centre_index = np.random.randint(mps_length)
        mps_mixed = mps.to_mixed_canonical(orth_centre_index)

        for i in range(mps_length):
            assert mps.tensors[i].shape == mps_mixed[i].shape
        assert is_canonical(mps_mixed)
        assert np.isclose(abs(inner_product(mps_mixed, mps_mixed)), 1)
        assert find_orth_centre(mps_mixed) == [orth_centre_index]


def test_entanglement_entropy():
    """
    Test of the implementation of the `entanglement_entropy` method.
    """

    mps_length = 4

    psi_two_body_dimer = 1 / np.sqrt(2) * np.array([0, -1, 1, 0], dtype=np.float64)
    psi_many_body_dimer = reduce(np.kron, [psi_two_body_dimer] * mps_length)

    mps_dimer = mps_from_dense(psi_many_body_dimer)

    entropy_list = np.array(mps_dimer.entanglement_entropy())

    correct_entropy_list = np.array([0, np.log(2), 0, np.log(2), 0, np.log(2), 0])

    zeros = entropy_list - correct_entropy_list

    assert np.allclose(np.linalg.norm(zeros), 0)


def test_create_custom_product_state():
    """
    Another test of the implementation of the `create_custom_product_state` function.
    """

    mps_1 = create_custom_product_state("0011++").to_right_canonical()

    mps_2 = [
        np.array([[[1.0], [0.0]]]),
        np.array([[[1.0], [0.0]]]),
        np.array([[[0.0], [1.0]]]),
        np.array([[[0.0], [1.0]]]),
        np.array([[[0.70710678], [0.70710678]]]),
        np.array([[[0.70710678], [0.70710678]]]),
    ]

    assert np.isclose(mps_1, mps_2).all()


def test_create_simple_product_state():
    """
    Another test of the implementation of the `create_simple_product_state` function.
    """

    mps_1 = create_simple_product_state(4, "0").to_right_canonical()
    mps_2 = create_simple_product_state(4, "1").to_right_canonical()
    mps_3 = create_simple_product_state(4, "+").to_right_canonical()

    mps_4 = [
        np.array([[[1.0], [0.0]]]),
        np.array([[[1.0], [0.0]]]),
        np.array([[[1.0], [0.0]]]),
        np.array([[[1.0], [0.0]]]),
    ]
    mps_5 = [
        np.array([[[0.0], [1.0]]]),
        np.array([[[0.0], [1.0]]]),
        np.array([[[0.0], [1.0]]]),
        np.array([[[0.0], [1.0]]]),
    ]
    mps_6 = [
        np.array([[[0.70710678], [0.70710678]]]),
        np.array([[[0.70710678], [0.70710678]]]),
        np.array([[[0.70710678], [0.70710678]]]),
        np.array([[[0.70710678], [0.70710678]]]),
    ]

    assert np.isclose(mps_1, mps_4).all()
    assert np.isclose(mps_2, mps_5).all()
    assert np.isclose(mps_3, mps_6).all()


def test_density_mpo():
    """
    Test of the implementation of the `density_mpo` method.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        density_mpo = mps.density_mpo()

        # Juggle the dimensions around to apply the `reduce` function later,
        # which is used to create a density mpo to compare the method against.
        for i in range(mps_length):
            density_mpo[i] = density_mpo[i].transpose((0, 3, 2, 1))

        density_matrix_mpo = reduce(
            lambda a, b: np.tensordot(a, b, (-1, 0)), density_mpo
        )

        # Get rid of ghost dimensions of the MPO.
        density_matrix_mpo = density_matrix_mpo.squeeze()
        # Reshaping to the right order of indices.
        correct_order = list(range(0, 2 * mps_length, 2)) + list(
            range(1, 2 * mps_length, 2)
        )
        density_matrix_mpo = density_matrix_mpo.transpose(correct_order)
        # Reshaping to the matrix form.
        density_matrix_mpo = density_matrix_mpo.reshape(
            (2**mps_length, 2**mps_length)
        )

        # Original density matrix.
        density_matrix = np.tensordot(psi, np.conjugate(psi), 0)

        assert np.isclose(np.trace(density_matrix), 1)
        assert np.isclose(
            np.linalg.norm(density_matrix - np.conjugate(density_matrix).T), 0
        )

        assert np.isclose(np.trace(density_matrix_mpo), 1)
        assert np.isclose(
            np.linalg.norm(density_matrix_mpo - np.conjugate(density_matrix_mpo).T),
            0,
        )


def test_to_explicit():
    """
    Test of the implementation of the `to_explicit` function.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        mps_left = mps.to_left_canonical()
        mps_right = mps.to_right_canonical()
        orth_centre_index = np.random.randint(mps_length)
        mps_mixed = mps.to_mixed_canonical(orth_centre_index)

        assert is_canonical(mps_left)
        assert is_canonical(mps_right)
        assert is_canonical(mps_mixed)

        explicit_from_right = to_explicit(mps_right)
        explicit_from_left = to_explicit(mps_left)
        explicit_from_mixed = to_explicit(mps_mixed)

        assert np.isclose(
            abs(inner_product(mps_right, explicit_from_right.to_right_canonical())), 1
        )

        assert np.isclose(
            abs(inner_product(mps_right, explicit_from_left.to_right_canonical())), 1
        )

        assert np.isclose(
            abs(inner_product(mps_left, explicit_from_right.to_left_canonical())), 1
        )

        assert np.isclose(
            abs(inner_product(mps_left, explicit_from_left.to_left_canonical())), 1
        )

        assert np.isclose(
            abs(
                inner_product(
                    mps_mixed, explicit_from_mixed.to_mixed_canonical(orth_centre_index)
                )
            ),
            1,
        )

        assert np.isclose(
            abs(
                inner_product(
                    mps_mixed, explicit_from_right.to_mixed_canonical(orth_centre_index)
                )
            ),
            1,
        )

        assert np.isclose(
            abs(
                inner_product(
                    mps_mixed, explicit_from_left.to_mixed_canonical(orth_centre_index)
                )
            ),
            1,
        )
