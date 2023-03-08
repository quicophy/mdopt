"""Tests for the ``mdopt.optimiser.utils`` module."""

import pytest
import numpy as np

from mdopt.optimiser.utils import (
    IDENTITY,
    COPY_RIGHT,
    SWAP,
    XOR_BULK,
    XOR_LEFT,
    XOR_RIGHT,
    ConstraintString,
)


def test_optimiser_utils_tensors():
    """Test for the combinatorial optimisation operations."""

    id = np.eye(2).reshape((1, 1, 2, 2))

    copy_right = np.zeros(shape=(2, 1, 2, 2))
    copy_right[0, 0, :, :] = np.array([[1, 0], [0, 0]])
    copy_right[1, 0, :, :] = np.array([[0, 0], [0, 1]])

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

    assert (id == IDENTITY).all()
    assert (copy_right == COPY_RIGHT).all()
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

        constraints_sites = [
            [xor_left_site],
            xor_bulk_sites,
            swap_sites,
            [xor_right_site],
        ]
        string = ConstraintString(constraints=tensors, sites=constraints_sites)

        with pytest.raises(ValueError):
            ConstraintString(constraints=[], sites=constraints_sites)

        with pytest.raises(ValueError):
            ConstraintString(constraints=tensors, sites=[])

        with pytest.raises(ValueError):
            ConstraintString(
                constraints=[XOR_LEFT, XOR_BULK, XOR_RIGHT], sites=constraints_sites
            )

        with pytest.raises(ValueError):
            constraints_sites = [
                [0],
                [1, 2, 3],
                [3, 4, 5],
                [6],
            ]
            ConstraintString(constraints=tensors, sites=constraints_sites)

        with pytest.raises(ValueError):
            constraints_sites = [
                [0],
                [1],
                [3, 4, 5],
                [6],
            ]
            ConstraintString(constraints=tensors, sites=constraints_sites)
