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


def _contract_with_trivial_boundaries(tensor):
    """
    Helper function: contract a tensor with shape (vL, vR, pU, pD)
    over its virtual indices using trivial boundary vectors (all ones).
    """
    vL, vR, pU, pD = tensor.shape
    left_bound = np.ones(vL)
    right_bound = np.ones(vR)
    op = np.zeros((pU, pD))
    for i in range(vL):
        for j in range(vR):
            op += left_bound[i] * tensor[i, j, :, :] * right_bound[j]
    return op


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

    # Testing using boundary contracting
    op = _contract_with_trivial_boundaries(IDENTITY)
    expected = np.eye(2)  # Since IDENTITY has shape (1,1,2,2)
    assert np.allclose(op, expected)

    op = _contract_with_trivial_boundaries(COPY_LEFT)
    # According to the definition:
    # For j = 0: COPY_LEFT[0,0,:,:] = np.eye(2)[0,:] = [[1, 0],[1, 0]] (constant in pU),
    # and for j = 1: COPY_LEFT[0,1,:,:] = np.eye(2)[1,:] = [[0, 1],[0, 1]].
    # Their sum gives:
    expected = np.array([[1, 1], [1, 1]])
    assert np.allclose(op, expected)

    op = _contract_with_trivial_boundaries(XOR_BULK)
    # Compute the expected operator by summing explicitly.
    expected = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    expected[k, l] += (i ^ j ^ k ^ 1) * (1 if k == l else 0)
    assert np.allclose(op, expected)

    op = _contract_with_trivial_boundaries(XOR_LEFT)
    # Here, since XOR_LEFT has shape (1,2,2,2), we sum over the right virtual index:
    expected = np.sum(XOR_LEFT[0, :, :, :], axis=0)
    assert np.allclose(op, expected)

    op = _contract_with_trivial_boundaries(XOR_RIGHT)
    expected = np.sum(XOR_RIGHT[:, 0, :, :], axis=0)
    assert np.allclose(op, expected)
    print("XOR_RIGHT effective operator test passed.")

    op = _contract_with_trivial_boundaries(SWAP)
    # SWAP is defined so that only [0,0,:,:] and [1,1,:,:] are nonzero.
    # Each equals np.eye(2). Therefore, op = np.eye(2) + np.eye(2) = 2I.
    expected = np.eye(2) * 2
    assert np.allclose(op, expected)


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
