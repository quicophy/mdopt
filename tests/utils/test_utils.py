"""
Tests for the `utils` module.
"""

import numpy as np
from opt_einsum import contract
from mpopt.utils.utils import (
    mpo_from_matrix,
    mpo_to_matrix,
    create_random_mpo,
    svd,
    kron_tensors,
    split_two_site_tensor,
)


def test_svd():
    """
    Test of the implementation of the `svd` function.
    """

    for _ in range(100):

        dim = np.random.randint(low=2, high=100, size=2)
        m = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)

        u, s, v_h = svd(m)

        m_trimmed = contract("ij, j, jk -> ik", u, s, v_h, optimize=[(0, 1), (0, 1)])

        u, s, v_h = svd(m_trimmed)

        m_trimmed_new = contract(
            "ij, j, jk -> ik", u, s, v_h, optimize=[(0, 1), (0, 1)]
        )

        assert np.isclose(np.linalg.norm(m_trimmed - m_trimmed_new), 0)


def test_svd_1():
    """
    Another test of the `svd` function.
    """

    for _ in range(100):

        dim = np.random.randint(low=50, high=100, size=2)
        m = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)
        num_sing_values = np.random.randint(1, 10)

        _, s, _ = svd(m, cut=1e-16, chi_max=num_sing_values, renormalise=True)

        assert len(s) == num_sing_values


def test_split_two_site_tensor():
    """
    Test of the implementation of the `split_two_site_tensor` function.
    """

    for _ in range(100):

        d = 2
        bond_dim = np.random.randint(2, 18, size=2)
        t = np.random.uniform(
            size=(bond_dim[0], d, d, bond_dim[1])
        ) + 1j * np.random.uniform(size=(bond_dim[0], d, d, bond_dim[1]))

        u_l, singular_values, v_r = split_two_site_tensor(t)

        should_be_t = contract(
            "ijk, kl, lmn -> ijmn",
            u_l,
            np.diag(singular_values),
            v_r,
            optimize=[(0, 1), (0, 1)],
        )

        assert t.shape == should_be_t.shape
        assert np.isclose(np.linalg.norm(t - should_be_t), 0)


def test_kron_tensors():
    """
    Test of the implementation of the `kron_tensors` function.
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

        product_1 = kron_tensors(
            tensor_1, tensor_2, conjugate_second=True, merge_physicals=True
        )
        product_2 = kron_tensors(
            tensor_1, tensor_2, conjugate_second=True, merge_physicals=False
        )
        product_3 = kron_tensors(
            tensor_1, tensor_2, conjugate_second=False, merge_physicals=True
        )
        product_4 = kron_tensors(
            tensor_1, tensor_2, conjugate_second=False, merge_physicals=False
        )

        product_5 = np.tensordot(tensor_1, np.conjugate(tensor_2), 0)
        product_5 = product_5.transpose(0, 3, 1, 4, 2, 5)
        product_5 = product_5.reshape(
            (dims_1[0] * dims_2[0], dims_1[1] * dims_2[1], dims_1[2] * dims_2[2])
        )

        product_6 = np.tensordot(tensor_1, np.conjugate(tensor_2), 0)
        product_6 = product_6.transpose(0, 3, 1, 4, 2, 5)
        product_6 = product_6.reshape(
            (
                product_6.shape[0] * product_6.shape[1],
                product_6.shape[2],
                product_6.shape[3],
                product_6.shape[4] * product_6.shape[5],
            )
        )

        product_7 = np.tensordot(tensor_1, tensor_2, 0)
        product_7 = product_7.transpose(0, 3, 1, 4, 2, 5)
        product_7 = product_7.reshape(
            (dims_1[0] * dims_2[0], dims_1[1] * dims_2[1], dims_1[2] * dims_2[2])
        )

        product_8 = np.tensordot(tensor_1, tensor_2, 0)
        product_8 = product_8.transpose(0, 3, 1, 4, 2, 5)
        product_8 = product_8.reshape(
            (
                product_8.shape[0] * product_8.shape[1],
                product_8.shape[2],
                product_8.shape[3],
                product_8.shape[4] * product_8.shape[5],
            )
        )

        assert np.isclose(np.linalg.norm(product_1 - product_5), 0)
        assert np.isclose(np.linalg.norm(product_2 - product_6), 0)
        assert np.isclose(np.linalg.norm(product_3 - product_7), 0)
        assert np.isclose(np.linalg.norm(product_4 - product_8), 0)


def test_mpo_from_matrix():
    """
    Test of the implementation of the `mpo_from_matrix` function.
    """

    for _ in range(100):

        num_sites = np.random.randint(4, 6)
        phys_dim = np.random.randint(2, 4)
        matrix_shape = (phys_dim**num_sites, phys_dim**num_sites)
        matrix = np.random.uniform(size=matrix_shape) + 1j * np.random.uniform(
            size=matrix_shape
        )

        mpo = mpo_from_matrix(
            matrix, interlaced=True, num_sites=num_sites, phys_dim=phys_dim
        )

        matrix_from_mpo = mpo_to_matrix(mpo, interlace=True, group=True)

        assert np.isclose(abs(np.linalg.norm(matrix - matrix_from_mpo)), 0)


def test_mpo_to_matrix():
    """
    Test of the implementation of the `mpo_to_matrix` function.
    """

    for _ in range(100):

        num_sites = np.random.randint(4, 6)
        phys_dim = np.random.randint(2, 4)

        dims_unique = np.random.randint(3, 10, size=num_sites - 1)
        mpo = create_random_mpo(
            num_sites,
            dims_unique,
            phys_dim=phys_dim,
            which=np.random.choice(["uniform", "normal", "randint"], size=1),
        )

        matrix_0 = mpo_to_matrix(mpo, interlace=False, group=False)
        matrix_1 = mpo_to_matrix(mpo, interlace=False, group=True)
        matrix_2 = mpo_to_matrix(mpo, interlace=True, group=False)
        matrix_3 = mpo_to_matrix(mpo, interlace=True, group=True)

        mpo_0 = mpo_from_matrix(
            matrix_0, interlaced=False, num_sites=num_sites, phys_dim=phys_dim
        )
        mpo_1 = mpo_from_matrix(
            matrix_1, interlaced=False, num_sites=num_sites, phys_dim=phys_dim
        )
        mpo_2 = mpo_from_matrix(
            matrix_2, interlaced=True, num_sites=num_sites, phys_dim=phys_dim
        )
        mpo_3 = mpo_from_matrix(
            matrix_3, interlaced=True, num_sites=num_sites, phys_dim=phys_dim
        )

        matrix_00 = mpo_to_matrix(mpo_0, interlace=False, group=False)
        matrix_01 = mpo_to_matrix(mpo_1, interlace=False, group=True)
        matrix_02 = mpo_to_matrix(mpo_2, interlace=True, group=False)
        matrix_03 = mpo_to_matrix(mpo_3, interlace=True, group=True)

        assert np.isclose(abs(np.linalg.norm(matrix_0 - matrix_00)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_1 - matrix_01)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_2 - matrix_02)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_3 - matrix_03)), 0, atol=1e-6)


def test_mpo_to_matrix_1():
    """
    Another test of the implementation of the `mpo_to_matrix` function.
    Here, we Test of the order of indices, so we fix the number of sites.
    """

    for _ in range(100):

        num_sites = 4
        mpo = create_random_mpo(
            num_sites,
            np.random.randint(2, 6, size=num_sites - 1),
            phys_dim=2,
            which=np.random.choice(["uniform", "normal", "randint"], size=1),
        )

        matrix_0 = mpo_to_matrix(mpo, interlace=True, group=True)
        matrix_1 = mpo_to_matrix(mpo, interlace=True, group=False)
        matrix_2 = mpo_to_matrix(mpo, interlace=False, group=True)
        matrix_3 = mpo_to_matrix(mpo, interlace=False, group=False)

        matrix_00 = contract(
            "abij, bckl, cdmn, deop -> ijklmnop",
            mpo[0],
            mpo[1],
            mpo[2],
            mpo[3],
            optimize=[(0, 1), (0, 1), (0, 1)],
        ).reshape((16, 16))
        matrix_01 = contract(
            "abij, bckl, cdmn, deop -> ijklmnop",
            mpo[0],
            mpo[1],
            mpo[2],
            mpo[3],
            optimize=[(0, 1), (0, 1), (0, 1)],
        )
        matrix_02 = contract(
            "abij, bckl, cdmn, deop -> jlnpikmo",
            mpo[0],
            mpo[1],
            mpo[2],
            mpo[3],
            optimize=[(0, 1), (0, 1), (0, 1)],
        ).reshape((16, 16))
        matrix_03 = contract(
            "abij, bckl, cdmn, deop -> jlnpikmo",
            mpo[0],
            mpo[1],
            mpo[2],
            mpo[3],
            optimize=[(0, 1), (0, 1), (0, 1)],
        )

        assert np.isclose(abs(np.linalg.norm(matrix_0 - matrix_00)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_1 - matrix_01)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_2 - matrix_02)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_3 - matrix_03)), 0, atol=1e-6)
