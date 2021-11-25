"""
    Tests for the explicit MPS construction.
"""

from functools import reduce
from itertools import combinations
from opt_einsum import contract
import numpy as np
from mpopt.mps.explicit import (
    mps_from_dense,
    is_canonical,
    find_orth_centre,
    move_orth_centre,
    split_two_site_tensor,
    to_explicit_form,
    inner_product,
    svd,
    interlace_tensors,
)

# TODO
# DMRG being too slow
# Apply unitary/non-unitary
# separate tests to different files
# check out the sanity tests they have in TenPy
# check out typing for fixing types of variables in functions
# mpos + mpo-mps contraction
# decoder in the experiments folder
# dtypes: complex64/32? float64/32?
# check all the functions do not act inplace
# copy attribute for the class
# reverse as an attribute for the class / separate function


def _create_psi(length):
    """
    A helper function which creates a random quantum state in the form of a state vector.
    """

    psi = np.random.uniform(size=(2 ** length)) + 1j * np.random.uniform(
        size=(2 ** length)
    )
    psi /= np.linalg.norm(psi)

    return psi


def test_svd():
    """
    Test the implementation of the svd function.
    """

    for _ in range(100):

        dim = np.random.randint(low=2, high=100, size=2)
        m = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)

        u, s, v_h = svd(m)

        m_trimmed = contract("ij, j, jk -> ik", u, s, v_h)

        u, s, v_h = svd(m_trimmed)

        m_trimmed_new = contract("ij, j, jk -> ik", u, s, v_h)

        assert np.isclose(np.linalg.norm(m_trimmed - m_trimmed_new), 0)


def test_svd_1():
    """
    Another test of the svd function.
    """

    for _ in range(100):

        dim = np.random.randint(low=50, high=100, size=2)
        m = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)
        num_sing_values = np.random.randint(1, 10)

        _, s, _ = svd(m, cut=1e-16, max_number=num_sing_values, normalise=True)

        assert len(s) == num_sing_values


def test_interlace_tensors():
    """
    Test the implementation of the interlace_tensors function.
    """

    for _ in range(100):

        dims_1 = np.random.randint(2, 11, size=3)
        dims_2 = np.random.randint(2, 11, size=3)

        tensor_1 = np.random.uniform(
            size=(dims_1[0], dims_1[1], dims_1[2])
        ) + 1j * np.random.uniform(size=(dims_1[0], dims_1[1], dims_1[2]))
        tensor_2 = np.random.uniform(
            size=(dims_2[0], dims_2[1], dims_2[2])
        ) + 1j * np.random.uniform(size=(dims_2[0], dims_2[1], dims_2[2]))

        product_1 = interlace_tensors(
            tensor_1, tensor_2, conjugate_second=True, merge_virtuals=True
        )
        product_2 = interlace_tensors(
            tensor_1, tensor_2, conjugate_second=True, merge_virtuals=False
        )
        product_3 = interlace_tensors(
            tensor_1, tensor_2, conjugate_second=False, merge_virtuals=True
        )
        product_4 = interlace_tensors(
            tensor_1, tensor_2, conjugate_second=False, merge_virtuals=False
        )

        product_5 = np.tensordot(tensor_1, np.conjugate(tensor_2), 0)
        product_5 = product_5.transpose(0, 3, 1, 4, 2, 5)
        product_5 = product_5.reshape(
            (dims_1[0] * dims_2[0], dims_1[1], dims_2[1], dims_1[2] * dims_2[2])
        )

        product_6 = np.tensordot(tensor_1, np.conjugate(tensor_2), 0)
        product_6 = product_6.transpose(0, 3, 1, 4, 2, 5)
        product_6 = product_6.reshape(
            (
                product_6.shape[0] * product_6.shape[1],
                product_6.shape[2] * product_6.shape[3],
                product_6.shape[4] * product_6.shape[5],
            )
        )

        product_7 = np.tensordot(tensor_1, tensor_2, 0)
        product_7 = product_7.transpose(0, 3, 1, 4, 2, 5)
        product_7 = product_7.reshape(
            (dims_1[0] * dims_2[0], dims_1[1], dims_2[1], dims_1[2] * dims_2[2])
        )

        product_8 = np.tensordot(tensor_1, tensor_2, 0)
        product_8 = product_8.transpose(0, 3, 1, 4, 2, 5)
        product_8 = product_8.reshape(
            (
                product_8.shape[0] * product_8.shape[1],
                product_8.shape[2] * product_8.shape[3],
                product_8.shape[4] * product_8.shape[5],
            )
        )

        assert np.isclose(np.linalg.norm(product_1 - product_5), 0)
        assert np.isclose(np.linalg.norm(product_2 - product_6), 0)
        assert np.isclose(np.linalg.norm(product_3 - product_7), 0)
        assert np.isclose(np.linalg.norm(product_4 - product_8), 0)


def test_from_dense():
    """
    Test the implementation of the mps_from_dense function.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)

        mps = mps_from_dense(psi)
        psi_from_mps = mps.to_dense().reshape((2 ** mps_length))

        overlap = abs(np.conjugate(psi_from_mps) @ psi) ** 2

        assert np.isclose(overlap, 1)


def test_single_site_left_iso():
    """
    Test the implementation of the single_site_left_iso method.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        for site in range(mps_length):
            isometry = mps.single_site_left_iso(site)

            to_be_identity = contract(
                "ijk, ijl -> kl", isometry, np.conjugate(isometry)
            )

            assert np.isclose(
                np.linalg.norm(to_be_identity - np.identity(to_be_identity.shape[0])), 0
            )


def test_to_left_canonical():
    """
    Test the implementation of the to_left_canonical method.
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
                "ijk, ijl -> kl", mps_left[i], np.conjugate(mps_left[i])
            )

            identity_left = np.identity(to_be_identity_left.shape[0], dtype=np.float64)

            assert np.isclose(np.linalg.norm(to_be_identity_left - identity_left), 0)


def test_single_site_right_iso():
    """
    Test the implementation of the single_site_right_iso method.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        for site in range(mps_length):

            isometry = mps.single_site_right_iso(site)

            to_be_identity = contract(
                "ijk, ljk -> il", isometry, np.conjugate(isometry)
            )

            assert np.isclose(
                np.linalg.norm(to_be_identity - np.identity(to_be_identity.shape[0])), 0
            )


def test_to_right_canonical():
    """
    Test the implementation of the to_right_canonical method.
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
                "ijk, ljk -> il", mps_right[i], np.conjugate(mps_right[i])
            )

            identity_right = np.identity(
                to_be_identity_right.shape[0], dtype=np.float64
            )

            assert np.isclose(np.linalg.norm(to_be_identity_right - identity_right), 0)


def test_to_mixed_canonical():
    """
    Test the implementation of the to_mixed_canonical method.
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
    Test the implementation of the entanglement_entropy method.
    """

    mps_length = 4

    psi_two_body_dimer = 1 / np.sqrt(2) * np.array([0, -1, 1, 0], dtype=np.float64)
    psi_many_body_dimer = reduce(np.kron, [psi_two_body_dimer] * mps_length)

    mps_dimer = mps_from_dense(psi_many_body_dimer)

    entropy_list = np.array(mps_dimer.entanglement_entropy())

    correct_entropy_list = np.array([0, np.log(2), 0, np.log(2), 0, np.log(2), 0])

    zeros = entropy_list - correct_entropy_list

    assert np.allclose(np.linalg.norm(zeros), 0)


def test_density_mpo():
    """
    Test the implementation of the density_mpo method.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        density_mpo = mps.density_mpo()

        for i in range(mps_length):
            density_mpo[i] = density_mpo[i].transpose((0, 3, 1, 2))

        density_matrix_mpo = reduce(
            lambda a, b: np.tensordot(a, b, (-1, 0)), density_mpo
        )

        # get rid of ghost dimensions of the MPO
        density_matrix_mpo = density_matrix_mpo.squeeze()
        # reshaping to the right order of indices
        correct_order = list(range(0, 2 * mps_length, 2)) + list(
            range(1, 2 * mps_length, 2)
        )
        density_matrix_mpo = density_matrix_mpo.transpose(correct_order)
        # reshaping to the matrix form
        density_matrix_mpo = density_matrix_mpo.reshape(
            (2 ** mps_length, 2 ** mps_length)
        )

        # original density matrix
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


def test_split_two_site_tensor():
    """
    Test the implementation of the split_two_site_tensor function.
    """

    for _ in range(100):

        d = 2
        bond_dim = np.random.randint(2, 18, size=2)
        t = np.random.uniform(
            size=(bond_dim[0], d, d, bond_dim[1])
        ) + 1j * np.random.uniform(size=(bond_dim[0], d, d, bond_dim[1]))

        u_l, schmidt_values, v_r = split_two_site_tensor(t)

        should_be_t = contract(
            "ijk, kl, lmn -> ijmn", u_l, np.diag(schmidt_values), v_r
        )

        assert t.shape == should_be_t.shape
        assert np.isclose(np.linalg.norm(t - should_be_t), 0)


def test_find_orth_centre():
    """
    Test the implementation of the find_orth_centre function.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        orth_centre_index = np.random.randint(mps_length)
        mps_mixed = mps.to_mixed_canonical(orth_centre_index)

        assert is_canonical(mps_mixed)
        assert find_orth_centre(mps_mixed) == [orth_centre_index]


def test_move_orth_centre():
    """
    Test the implementation of the move_orth_centre function.
    """

    mps_length = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        orth_centre_index_init = np.random.randint(mps_length)
        mps_mixed_init = mps.to_mixed_canonical(orth_centre_index_init)
        assert np.isclose(abs(inner_product(mps_mixed_init, mps_mixed_init)), 1)
        assert is_canonical(mps_mixed_init)
        assert find_orth_centre(mps_mixed_init) == [orth_centre_index_init]

        orth_centre_index_final = np.random.randint(mps_length)
        mps_mixed_final = move_orth_centre(
            mps_mixed_init, orth_centre_index_init, orth_centre_index_final
        )

        assert np.isclose(abs(inner_product(mps_mixed_final, mps_mixed_final)), 1)
        assert is_canonical(mps_mixed_final)
        assert find_orth_centre(mps_mixed_final) == [orth_centre_index_final]


def test_inner_product():
    """
    Test the implementation of the inner_product function.
    """

    mps_length = 5

    for _ in range(100):

        psi = _create_psi(mps_length)
        mps = mps_from_dense(psi)

        # all possible orthogonality centre indices
        orth_centre_indices = np.arange(mps_length)

        list_of_mps = []

        list_of_mps.append(mps.to_left_canonical())
        list_of_mps.append(mps.to_right_canonical())
        for index in orth_centre_indices:
            list_of_mps.append(mps.to_mixed_canonical(index))

        index_list = np.arange(len(list_of_mps))
        index_pairs = list(combinations(index_list, 2))

        for pair in index_pairs:
            assert np.isclose(
                abs(inner_product(list_of_mps[pair[0]], list_of_mps[pair[1]])), 1
            )


def test_to_explicit_form():
    """
    Test the implementation of the to_explicit_form function.
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

        explicit_from_right = to_explicit_form(mps_right)
        explicit_from_left = to_explicit_form(mps_left)
        explicit_from_mixed = to_explicit_form(mps_mixed)

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
