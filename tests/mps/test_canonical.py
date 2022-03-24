"""
    Tests for the canonical MPS construction.
"""

from itertools import combinations
import numpy as np
from tests.mps.test_explicit import _create_psi
from mpopt.mps.canonical import (
    is_canonical,
    inner_product,
    find_orth_centre,
    move_orth_centre,
)
from mpopt.mps.explicit import mps_from_dense


def test_find_orth_centre():
    """
    Test the implementation of the `find_orth_centre` function.
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
    Test the implementation of the `move_orth_centre` function.
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
    Test the implementation of the `inner_product` function.
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
