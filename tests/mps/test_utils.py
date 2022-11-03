"""Tests for the ``mdopt.mps.utils`` module."""

from itertools import combinations
import pytest
import numpy as np

from mdopt.mps.utils import (
    create_state_vector,
    find_orth_centre,
    is_canonical,
    inner_product,
    mps_from_dense,
    create_custom_product_state,
    create_simple_product_state,
)


def test_mps_utils_create_state_vector():
    """Test for the ``create_state_vector`` function"""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        assert psi.shape[0] == 2**num_sites
        assert isinstance(psi, np.ndarray)
        assert psi.dtype == np.dtype(np.complex128)
        assert np.isclose(np.linalg.norm(psi) - 1, 0)


def test_mps_utils_find_orth_centre():
    """Test for the ``find_orth_centre`` function."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        orth_centre = np.random.randint(num_sites)
        mps_explicit = mps_from_dense(psi, form="Explicit")
        mps_mixed = mps_from_dense(psi, form="Mixed-canonical", orth_centre=orth_centre)
        mps_left = mps_from_dense(psi, form="Left-canonical")
        mps_right = mps_from_dense(psi, form="Right-canonical")
        mps_product = create_simple_product_state(
            num_sites=num_sites, which="0", form="Right-canonical"
        )

        with pytest.raises(ValueError):
            find_orth_centre(mps_explicit)
        assert np.isclose(find_orth_centre(mps_mixed), [orth_centre])
        assert np.isclose(
            find_orth_centre(mps_left, return_orth_flags=True)[1], [True] * num_sites
        ).all()
        assert np.isclose(
            find_orth_centre(mps_right, return_orth_flags=True)[2], [True] * num_sites
        ).all()
        assert np.isclose(find_orth_centre(mps_product), [0])


def test_mps_utils_is_canonical():
    """Test for the ``is_canonical`` function."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        orth_centre = np.random.randint(num_sites)
        mps_explicit = mps_from_dense(psi, form="Explicit")
        mps_mixed = mps_from_dense(psi, form="Mixed-canonical", orth_centre=orth_centre)
        mps_left = mps_from_dense(psi, form="Left-canonical")
        mps_right = mps_from_dense(psi, form="Right-canonical")

        with pytest.raises(ValueError):
            is_canonical(mps_explicit)
        assert is_canonical(mps_mixed)
        assert is_canonical(mps_left)
        assert is_canonical(mps_right)


def test_mps_utils_inner_product():
    """Test for the ``inner_product`` function."""

    num_sites = 5

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        # All possible orthogonality centre indices.
        orth_centre_indices = np.arange(num_sites)

        list_of_mps = []

        list_of_mps.append(mps.left_canonical())
        list_of_mps.append(mps.right_canonical())
        for index in orth_centre_indices:
            list_of_mps.append(mps.mixed_canonical(index))

        index_list = np.arange(len(list_of_mps))
        index_pairs = list(combinations(index_list, 2))

        for pair in index_pairs:
            assert np.isclose(
                abs(inner_product(list_of_mps[pair[0]], list_of_mps[pair[1]])), 1
            )

        with pytest.raises(ValueError):
            inner_product(mps, create_simple_product_state(num_sites=6))


def test_mps_utils_mps_from_dense():
    """Test for the ``mps_from_dense`` function."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps_explicit = mps_from_dense(psi, form="Explicit")
        psi_from_mps_explicit = mps_explicit.dense().reshape((2**num_sites))
        overlap = abs(np.conjugate(psi_from_mps_explicit) @ psi) ** 2
        assert np.isclose(overlap, 1)

        psi = create_state_vector(num_sites)
        mps_right = mps_from_dense(psi, form="Right-canonical")
        psi_from_mps_right = mps_right.dense().reshape((2**num_sites))
        overlap = abs(np.conjugate(psi_from_mps_right) @ psi) ** 2
        assert np.isclose(overlap, 1)

        psi = create_state_vector(num_sites)
        mps_left = mps_from_dense(psi, form="Left-canonical")
        psi_from_mps_left = mps_left.dense().reshape((2**num_sites))
        overlap = abs(np.conjugate(psi_from_mps_left) @ psi) ** 2
        assert np.isclose(overlap, 1)

        psi = create_state_vector(num_sites)
        orth_centre_index = np.random.randint(num_sites)
        mps_mixed = mps_from_dense(
            psi, form="Mixed-canonical", orth_centre=orth_centre_index
        )
        psi_from_mps_mixed = mps_mixed.dense().reshape((2**num_sites))
        overlap = abs(np.conjugate(psi_from_mps_mixed) @ psi) ** 2
        assert np.isclose(overlap, 1)

        with pytest.raises(ValueError):
            mps_from_dense(np.ones(shape=(23,)))


def test_mps_utils_create_simple_product_state():
    """Test for the ``create_simple_product_state`` function."""

    mps_1 = create_simple_product_state(4, "0").right_canonical()
    mps_2 = create_simple_product_state(4, "1").right_canonical()
    mps_3 = create_simple_product_state(4, "+").right_canonical()

    mps_1_tensors = [
        np.array([[[1.0], [0.0]]]),
        np.array([[[1.0], [0.0]]]),
        np.array([[[1.0], [0.0]]]),
        np.array([[[1.0], [0.0]]]),
    ]
    mps_2_tensors = [
        np.array([[[0.0], [1.0]]]),
        np.array([[[0.0], [1.0]]]),
        np.array([[[0.0], [1.0]]]),
        np.array([[[0.0], [1.0]]]),
    ]
    mps_3_tensors = [
        np.array([[[0.70710678], [0.70710678]]]),
        np.array([[[0.70710678], [0.70710678]]]),
        np.array([[[0.70710678], [0.70710678]]]),
        np.array([[[0.70710678], [0.70710678]]]),
    ]

    assert np.isclose(mps_1.tensors, mps_1_tensors).all()
    assert np.isclose(mps_2.tensors, mps_2_tensors).all()
    assert np.isclose(mps_3.tensors, mps_3_tensors).all()
    with pytest.raises(ValueError):
        create_simple_product_state(4, "0", form="Mixed-canonical")


def test_mps_utils_create_custom_product_state():
    """Test for the ``create_custom_product_state`` function."""

    mps_1 = create_custom_product_state("0011++").right_canonical()

    with pytest.raises(ValueError):
        create_custom_product_state("0213++")

    mps_1_tensors = [
        np.array([[[1.0], [0.0]]]),
        np.array([[[1.0], [0.0]]]),
        np.array([[[0.0], [1.0]]]),
        np.array([[[0.0], [1.0]]]),
        np.array([[[0.70710678], [0.70710678]]]),
        np.array([[[0.70710678], [0.70710678]]]),
    ]

    assert np.isclose(mps_1.tensors, mps_1_tensors).all()
    with pytest.raises(ValueError):
        create_custom_product_state("0011++", form="Mixed-canonical")
