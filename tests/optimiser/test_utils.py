"""Tests for the ``mdopt.optimiser.utils`` module."""

import pytest
import numpy as np

from mdopt.optimiser.utils import (
    SWAP,
    XOR_BULK,
    XOR_LEFT,
    XOR_RIGHT,
    COPY_LEFT,
    IDENTITY,
    ConstraintString,
)


def test_optimiser_utils_tensors():
    """Test for the combinatorial optimisation operations."""

    identity = np.eye(2).reshape((1, 1, 2, 2))

    copy_left = np.zeros(shape=(1, 2, 2, 2))
    copy_left[0, :, 0, :] = np.eye(2)
    copy_left[0, :, 1, :] = np.eye(2)

    swap = np.zeros(shape=(2, 2, 2, 2))
    swap[0, 0] = np.eye(2)
    swap[1, 1] = np.eye(2)

    xor_bulk = np.zeros(shape=(2, 2, 2, 2))
    xor_bulk[0, 0] = np.array([[1, 0], [0, 0]])
    xor_bulk[0, 1] = np.array([[0, 0], [0, 1]])
    xor_bulk[1, 0] = np.array([[0, 0], [0, 1]])
    xor_bulk[1, 1] = np.array([[1, 0], [0, 0]])

    xor_left = np.zeros(shape=(1, 2, 2, 2))
    xor_left[0] = xor_bulk[0]

    xor_right = np.zeros(shape=(2, 1, 2, 2))
    xor_right[:, 0] = xor_bulk[:, 0]

    assert (identity == IDENTITY).all()
    assert (copy_left == COPY_LEFT).all()
    assert (swap == SWAP).all()
    assert (xor_bulk == XOR_BULK).all()
    assert (xor_left == XOR_LEFT).all()
    assert (xor_right == XOR_RIGHT).all()


def test_optimiser_utils_constraint_string():
    """Tests for the ``ConstraintString`` class."""

    tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]
    for _ in range(10):
        num_sites = 20
        xor_left_site = np.random.randint(low=0, high=10)
        xor_right_site = np.random.randint(low=xor_left_site + 1, high=num_sites)
        sites = list(range(xor_left_site + 1, xor_right_site))
        xor_bulk_sites = list(
            np.sort(np.random.choice(sites, size=int(len(sites) / 2)))
        )
        xor_bulk_sites = list(set(xor_bulk_sites))
        swap_sites = [site for site in sites if site not in xor_bulk_sites]

        constraint_sites = [
            [xor_left_site],
            xor_bulk_sites,
            swap_sites,
            [xor_right_site],
        ]

        string = ConstraintString(constraints=tensors, sites=constraint_sites)

        mpo = [None for _ in range(num_sites)]
        mpo[xor_left_site] = XOR_LEFT
        mpo[xor_right_site] = XOR_RIGHT
        for site in xor_bulk_sites:
            mpo[site] = XOR_BULK
        for site in swap_sites:
            mpo[site] = SWAP
        mpo = [tensor for tensor in mpo if tensor is not None]

        for tensor_string, tensor_test in zip(string.mpo(), mpo):
            assert (tensor_string == tensor_test).all()
        for i in range(4):
            assert np.isclose(string[i][0], constraint_sites[i]).all()
            assert np.isclose(string[i][1], tensors[i]).all()
        assert string.span() == xor_right_site - xor_left_site + 1

        with pytest.raises(ValueError):
            ConstraintString(constraints=[], sites=constraint_sites)

        with pytest.raises(ValueError):
            ConstraintString(constraints=tensors, sites=[])

        with pytest.raises(ValueError):
            ConstraintString(
                constraints=[XOR_LEFT, XOR_BULK, XOR_RIGHT], sites=constraint_sites
            )

        with pytest.raises(ValueError):
            constraint_sites = [
                [0],
                [1, 2, 3],
                [3, 4, 5],
                [6],
            ]
            ConstraintString(constraints=tensors, sites=constraint_sites)

        with pytest.raises(ValueError):
            constraint_sites = [
                [0],
                [1],
                [3, 4, 5],
                [6],
            ]
            ConstraintString(constraints=tensors, sites=constraint_sites)
