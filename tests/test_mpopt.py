"""
    This module contains tests.
    Written by Alex Berezutskii inspired by TenPy in 2020-2021.
"""

import scipy
import pytest
import numpy as np
from mpopt.utils import trimmed_SVD, MPS_from_dense, FM_MPS, AFM_MPS


# TO DO:
# 1) try to implement tests for all of the functions there are in the classes MPS and DMRG
# 2) the canonical form test 
# 3) check out the sanity tests they have in TenPy


def test_trimmed_SVD():

    for _ in range(100):

        dim_1 = np.random.randint(low=2, high=100)
        dim_2 = np.random.randint(low=2, high=100)
        M = np.random.rand(dim_1, dim_2)

        U, S, Vh = trimmed_SVD(
            M,
            cut=1e-12,
            max_num=int(1e6),
            normalize=True,
            init_norm=True,
            norm_ord=2,
            limit_max=False,
            err_th=1e-12,
        )

        M_trimmed = np.einsum("ij, j, jk -> ik", U, S, Vh)

        U, S, Vh = trimmed_SVD(
            M_trimmed,
            cut=1e-12,
            max_num=int(1e6),
            normalize=True,
            init_norm=True,
            norm_ord=2,
            limit_max=False,
            err_th=1e-12,
        )

        M_trimmed_new = np.einsum("ij, j, jk -> ik", U, S, Vh)

        norm = np.linalg.norm(M_trimmed - M_trimmed_new, ord="fro")

        assert np.isclose(norm, 0.0)

    return


def test_from_dense():

    n = 8

    for _ in range(100):

        psi = np.random.rand(2 ** n)
        index = np.random.randint(low=0, high=2 ** n - 1)
        psi[index] = 1.0
        psi /= np.linalg.norm(psi)

        mps = MPS_from_dense(psi, d=2, limit_max=False, max_num=100)
        psi_from_mps = mps.to_dense().reshape(2 ** n)

        overlap = abs(np.dot(psi, psi_from_mps))

        assert np.isclose(overlap, 1.0, atol=1e-7)

    return


def test_mps_fm():

    L = 8
    d = 2

    mps = FM_MPS(L, d)
    psi = mps.to_dense().reshape(d ** L)

    psi_true = np.zeros(d ** L)
    psi_true[0] = 1.0

    overlap = abs(psi @ psi_true)

    assert np.isclose(overlap, 1.0)

    return


def test_mps_afm():

    L = 8
    d = 2

    mps = AFM_MPS(L, d)
    psi = mps.to_dense().reshape(d ** L)

    psi_true = np.zeros(d ** L)
    psi_true[-1] = 1.0

    overlap = abs(psi @ psi_true)

    assert np.isclose(overlap, 1.0)

    return
