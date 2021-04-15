"""
    Tests for the explicit MPS construction.
"""

import numpy as np
from mpopt.utils import dagger, trimmed_svd, tensor_product_with_dagger
from mpopt.mps.explicit import mps_from_dense
from experiments.decoder import ferro_mps, antiferro_mps

# TODO:
# 1) try to implement tests for all of the functions there are in the classes mps and dmrg
# 2) the canonical form test
# 3) check out the sanity tests they have in TenPy
# 4) check out typing, mypy
# 5) notebook to experiments


def test_trimmed_svd():
    for _ in range(100):
        dim_1 = np.random.randint(low=2, high=100)
        dim_2 = np.random.randint(low=2, high=100)
        m = np.random.rand(dim_1, dim_2)

        u, s, v_h = trimmed_svd(
            m,
            cut=1e-12,
            max_num=1e6,
            normalise=True,
            init_norm=True,
            norm_ord=2,
            limit_max=False,
            err_th=1e-12,
        )

        m_trimmed = np.einsum("ij, j, jk -> ik", u, s, v_h)

        u, s, v_h = trimmed_svd(
            m_trimmed,
            cut=1e-12,
            max_num=int(1e6),
            normalise=True,
            init_norm=True,
            norm_ord=2,
            limit_max=False,
            err_th=1e-12,
        )

        m_trimmed_new = np.einsum("ij, j, jk -> ik", u, s, v_h)

        difference_norm = np.linalg.norm(m_trimmed - m_trimmed_new, ord="fro")

        assert np.isclose(difference_norm, 0.0)


def test_from_dense():

    n = 8

    for _ in range(100):

        psi = np.random.rand(2 ** n)
        index = np.random.randint(low=0, high=2 ** n - 1)
        psi[index] = 1.0
        psi /= np.linalg.norm(psi)

        mps = mps_from_dense(psi, dim=2, limit_max=False, max_num=100)
        psi_from_mps = mps.to_dense().reshape(2 ** n)

        overlap = abs(np.dot(psi, psi_from_mps))

        assert np.isclose(overlap, 1.0, atol=1e-7)

    return


def test_ferro_mps():

    L = 8
    d = 2

    mps = ferro_mps(L, d)
    psi = mps.to_dense().reshape(d ** L)

    psi_true = np.zeros(d ** L)
    psi_true[0] = 1.0

    overlap = abs(psi @ psi_true)

    assert np.isclose(overlap, 1.0)

    return


def test_antiferro_mps():

    L = 8
    d = 2

    mps = antiferro_mps(L, d)
    psi = mps.to_dense().reshape(d ** L)

    psi_true = np.zeros(d ** L)
    psi_true[-1] = 1.0

    overlap = abs(psi @ psi_true)

    assert np.isclose(overlap, 1.0)

    return


def test_tensor_product_with_dagger():

    for _ in range(100):
        # TODO complex!
        dim_0 = np.random.randint(2, 11)
        dim_1 = np.random.randint(2, 11)
        dim_2 = np.random.randint(2, 11)
        tensor = np.random.rand(dim_0, dim_1, dim_2)

        product = tensor_product_with_dagger(tensor)

        product_to_compare = np.tensordot(
           tensor, dagger(tensor), axes=0
        )
        product_to_compare = product_to_compare.transpose(0,3,1,4,2,5)
        product_to_compare = np.reshape(
            product_to_compare, (dim_0 ** 2, dim_1, dim_1, dim_2 ** 2)
        )

        assert product.shape == product_to_compare.shape
        assert np.isclose(product, product_to_compare).all()
