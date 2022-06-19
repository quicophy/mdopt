"""
    Tests for the canonical MPS construction.
"""

from itertools import combinations

import numpy as np

from mpopt.mps.canonical import (
    find_orth_centre,
    inner_product,
    is_canonical,
    move_orth_centre,
    to_density_mpo,
)
from mpopt.mps.explicit import mps_from_dense
from tests.mps.test_explicit import _create_psi


def test_find_orth_centre():
    """
    Test of the implementation of the `find_orth_centre` function.
    """

    num_sites = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(num_sites)
        mps = mps_from_dense(psi)

        orth_centre_index = np.random.randint(num_sites)
        mps_mixed = mps.to_mixed_canonical(orth_centre_index)

        assert is_canonical(mps_mixed)
        assert find_orth_centre(mps_mixed) == [orth_centre_index]


def test_find_orth_centre_1():
    """
    Another test of the implementation of the `find_orth_centre` function.
    """

    num_sites = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(num_sites)
        mps = mps_from_dense(psi)

        mps_left = mps.to_left_canonical()
        mps_right = mps.to_right_canonical()

        assert is_canonical(mps_left)
        assert is_canonical(mps_right)
        assert find_orth_centre(mps_left, return_flags=True)[1] == [True] * num_sites
        assert find_orth_centre(mps_right, return_flags=True)[2] == [True] * num_sites


def test_move_orth_centre():
    """
    Test of the implementation of the `move_orth_centre` function.
    """

    num_sites = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(num_sites)
        mps = mps_from_dense(psi)

        orth_centre_index_init = np.random.randint(num_sites)
        mps_mixed_init = mps.to_mixed_canonical(orth_centre_index_init)

        assert np.isclose(abs(inner_product(mps_mixed_init, mps_mixed_init)), 1)
        assert is_canonical(mps_mixed_init)
        assert find_orth_centre(mps_mixed_init) == [orth_centre_index_init]

        orth_centre_index_final = np.random.randint(num_sites)
        mps_mixed_final = move_orth_centre(
            mps_mixed_init, orth_centre_index_init, orth_centre_index_final
        )

        assert np.isclose(abs(inner_product(mps_mixed_final, mps_mixed_final)), 1)
        assert is_canonical(mps_mixed_final)
        assert find_orth_centre(mps_mixed_final) == [orth_centre_index_final]


def test_inner_product():
    """
    Test of the implementation of the `inner_product` function.
    """

    num_sites = 5

    for _ in range(100):

        psi = _create_psi(num_sites)
        mps = mps_from_dense(psi)

        # all possible orthogonality centre indices
        orth_centre_indices = np.arange(num_sites)

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


def test_to_density_mpo():
    """
    Test of the implementation of the `to_density_mpo` function.
    """

    num_sites = np.random.randint(4, 9)

    for _ in range(100):

        psi = _create_psi(num_sites)
        mps = mps_from_dense(psi)

        density_mpo_from_expl = mps.density_mpo()
        density_mpo_from_can = to_density_mpo(mps.to_right_canonical())

        for i in range(num_sites):
            assert np.isclose(
                np.linalg.norm(density_mpo_from_expl[i] - density_mpo_from_can[i]), 0
            )
