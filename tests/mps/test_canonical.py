"""Tests for the ``mdopt.mps.canonical.CanonicalMPS`` class."""

from functools import reduce
from typing import Iterable
import numpy as np
import pytest
from opt_einsum import contract

from mdopt.mps.utils import (
    create_state_vector,
    mps_from_dense,
    inner_product,
    is_canonical,
    find_orth_centre,
)
from mdopt.mps.canonical import CanonicalMPS


def test_canonical_init():
    """Tests for the ``__init__`` and ``__len__`` methods of the ``CanonicalMPS`` class."""

    for _ in range(10):

        num_sites = np.random.randint(4, 9)
        tensors = [
            np.random.uniform(low=0, high=1, size=(1, 2, 1))
            + 1j * np.random.uniform(low=0, high=1, size=(1, 2, 1))
            for _ in range(num_sites)
        ]
        orth_centre = np.random.randint(low=0, high=num_sites)
        tolerance = 1e-10
        chi_max = 1e3
        mps = CanonicalMPS(
            tensors=tensors,
            orth_centre=orth_centre,
            tolerance=tolerance,
            chi_max=chi_max,
        )

        for mps_tensor, tensor in zip(mps.tensors, tensors):
            assert np.isclose(mps_tensor, tensor).all()
        assert np.isclose(mps.num_sites, num_sites).all()
        assert np.isclose(mps.num_bonds, num_sites - 1).all()
        assert np.isclose(mps.bond_dimensions, [1 for _ in range(mps.num_bonds)]).all()
        assert np.isclose(mps.phys_dimensions, [2 for _ in range(mps.num_sites)]).all()
        assert mps.orth_centre == orth_centre
        assert mps.dtype == np.dtype(np.complex128)
        assert np.isclose(mps.tolerance, tolerance).all()
        assert np.isclose(mps.chi_max, chi_max).all()
        assert np.isclose(len(mps), mps.num_sites).all()

        with pytest.raises(ValueError):
            CanonicalMPS(tensors=tensors, orth_centre=num_sites + 1)

        with pytest.raises(ValueError):
            CanonicalMPS(tensors=tensors + np.zeros((2, 2, 2, 2)))


def test_canonical_copy():
    """Test for the ``copy`` method of the ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")
        mps_copied = mps.copy()

        for tensor, tensor_copy in zip(mps.tensors, mps_copied.tensors):
            assert np.isclose(tensor, tensor_copy).all()
        assert np.isclose(mps.num_sites, mps_copied.num_sites).all()
        assert np.isclose(mps.num_bonds, mps_copied.num_bonds).all()
        assert np.isclose(mps.bond_dimensions, mps_copied.bond_dimensions).all()
        assert np.isclose(mps.phys_dimensions, mps_copied.phys_dimensions).all()
        assert mps.orth_centre == mps_copied.orth_centre
        assert mps.dtype == mps_copied.dtype
        assert np.isclose(mps.tolerance, mps_copied.tolerance).all()
        assert np.isclose(mps.chi_max, mps_copied.chi_max).all()
        assert np.isclose(len(mps), len(mps_copied)).all()


def test_canonical_reverse():
    """Test for the ``reverse`` method of the ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Mixed-canonical", orth_centre=num_sites - 2)
        mps_reversed = mps.reverse()

        for tensor, reversed_tensor in zip(reversed(mps.tensors), mps_reversed.tensors):
            assert np.isclose(tensor, np.transpose(reversed_tensor)).all()
        assert np.isclose(mps_reversed.num_sites, mps.num_sites).all()
        assert np.isclose(mps_reversed.num_bonds, mps.num_bonds).all()
        assert np.isclose(mps_reversed.bond_dimensions, mps.bond_dimensions).all()
        assert np.isclose(mps_reversed.phys_dimensions, mps.phys_dimensions).all()
        assert mps_reversed.orth_centre == mps.num_sites - 1 - mps.orth_centre
        assert mps_reversed.dtype == mps.dtype
        assert np.isclose(mps_reversed.tolerance, mps.tolerance).all()
        assert np.isclose(mps_reversed.chi_max, mps.chi_max).all()
        assert np.isclose(len(mps_reversed), len(mps)).all()


def test_canonical_conjugate():
    """Test for the ``conjugate`` method of the ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")
        mps_conjugated = mps.conjugate()

        for tensor, conjugated_tensor in zip(mps.tensors, mps_conjugated.tensors):
            assert np.isclose(tensor, np.conjugate(conjugated_tensor)).all()
        assert np.isclose(mps.num_sites, mps_conjugated.num_sites).all()
        assert np.isclose(mps.num_bonds, mps_conjugated.num_bonds).all()
        assert np.isclose(mps.bond_dimensions, mps_conjugated.bond_dimensions).all()
        assert np.isclose(mps.phys_dimensions, mps_conjugated.phys_dimensions).all()
        assert mps.orth_centre == mps_conjugated.orth_centre
        assert mps.dtype == mps_conjugated.dtype
        assert np.isclose(mps.tolerance, mps_conjugated.tolerance).all()
        assert np.isclose(mps.chi_max, mps_conjugated.chi_max).all()
        assert np.isclose(len(mps), len(mps_conjugated)).all()


def test_canonical_single_site_tensor():
    """Test for the ``single_site_tensor`` method of the ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")
        site = np.random.randint(0, num_sites)

        with pytest.raises(ValueError):
            mps.single_site_tensor(-100)

        assert np.isclose(mps.single_site_tensor(site), mps.tensors[site]).all()


def test_canonical_single_site_tensor_iter():
    """Test for the ``single_site_tensor_iter`` method of the ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")

        assert isinstance(mps.single_site_tensor_iter(), Iterable)


def test_canonical_two_site_tensor_next():
    """Test for the ``two_site_tensor_next`` method of the ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")
        site = np.random.randint(0, num_sites - 1)

        with pytest.raises(ValueError):
            mps.single_site_tensor(-2)

        assert np.isclose(
            mps.two_site_tensor_next(site),
            contract(
                "ijk, klm -> ijlm",
                mps.single_site_tensor(site),
                mps.single_site_tensor(site + 1),
            ),
        ).all()


def test_canonical_two_site_tensor_prev():
    """Test for the ``two_site_tensor_prev`` method of the ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")
        site = np.random.randint(1, num_sites)

        with pytest.raises(ValueError):
            mps.single_site_tensor(-site)

        assert np.isclose(
            mps.two_site_tensor_prev(site),
            contract(
                "ijk, klm -> ijlm",
                mps.single_site_tensor(site - 1),
                mps.single_site_tensor(site),
            ),
        ).all()


def test_canonical_two_site_tensor_next_iter():
    """Test for the ``two_site_tensor_next_iter`` method of ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")

        assert isinstance(mps.two_site_tensor_next_iter(), Iterable)


def test_canonical_two_site_tensor_prev_iter():
    """Test for the ``two_site_tensor_prev_iter`` method of ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")

        assert isinstance(mps.two_site_tensor_prev_iter(), Iterable)


def test_canonical_dense():
    """Test for the ``dense`` method of the ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")

        assert np.isclose(psi, mps.dense(flatten=True)).all()


def test_canonical_density_mpo():
    """Test for the ``density_mpo`` method of the ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps_exp = mps_from_dense(psi, form="Explicit")
        mps_can = mps_from_dense(psi, form="Right-canonical")

        density_mpo_from_exp = mps_exp.density_mpo()
        density_mpo_from_can = mps_can.density_mpo()

        for i in range(num_sites):
            assert np.isclose(
                np.linalg.norm(density_mpo_from_exp[i] - density_mpo_from_can[i]), 0
            )


def test_canonical_entanglement_entropy():
    """Test for the ``density_mpo`` method of the ``CanonicalMPS`` class."""

    num_sites = 4

    psi_two_body_dimer = 1 / np.sqrt(2) * np.array([0, -1, 1, 0], dtype=np.float32)
    psi_many_body_dimer = reduce(np.kron, [psi_two_body_dimer] * num_sites)

    mps_dimer = mps_from_dense(psi_many_body_dimer, tolerance=1e-6)

    entropy_list = mps_dimer.entanglement_entropy()

    correct_entropy_list = np.array([0, np.log(2), 0, np.log(2), 0, np.log(2), 0])

    zeros = entropy_list - correct_entropy_list

    assert np.allclose(np.linalg.norm(zeros), 0, atol=1e-6)


def test_canonical_move_orth_centre():
    """Test for the ``move_orth_centre`` method of the ``CanonicalMPS`` class."""

    # num_sites = np.random.randint(4, 9)
    num_sites = 6

    for _ in range(10):

        psi = create_state_vector(num_sites)

        orth_centre_init = 3  # np.random.randint(num_sites)
        mps_mixed_init = mps_from_dense(
            psi, form="Mixed-canonical", orth_centre=orth_centre_init
        )
        orth_centre_final = 0  # np.random.randint(num_sites)
        mps_mixed_final = mps_mixed_init.move_orth_centre(orth_centre_final)

        with pytest.raises(ValueError):
            mps_mixed_init.move_orth_centre(-3)

        assert is_canonical(mps_mixed_init)
        assert is_canonical(mps_mixed_final)
        assert find_orth_centre(mps_mixed_init) == [orth_centre_init]
        assert find_orth_centre(mps_mixed_final) == [orth_centre_final]


def test_canonical_move_orth_centre_to_border():
    """Test for the ``move_orth_centre_to_border`` method of ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps_mixed_init_1 = mps_from_dense(psi, form="Mixed-canonical", orth_centre=1)
        mps_mixed_init_2 = mps_from_dense(
            psi, form="Mixed-canonical", orth_centre=num_sites - 2
        )

        mps, position = mps_mixed_init_1.move_orth_centre_to_border()
        for tensor_1, tensor_2 in zip(
            mps.tensors, mps_mixed_init_1.move_orth_centre(0).tensors
        ):
            assert np.isclose(tensor_1, tensor_2).all()
        assert position == "first"

        mps, position = mps_mixed_init_2.move_orth_centre_to_border()
        for tensor_1, tensor_2 in zip(
            mps.tensors, mps_mixed_init_2.move_orth_centre(num_sites - 1).tensors
        ):
            assert np.isclose(tensor_1, tensor_2).all()
        assert position == "last"


def test_canonical_explicit():
    """Test for the ``explicit`` method of the ``CanonicalMPS`` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps_left = mps_from_dense(psi, form="Left-canonical")
        mps_right = mps_from_dense(psi, form="Right-canonical")
        orth_centre = np.random.randint(num_sites)
        mps_mixed = mps_from_dense(psi, form="Mixed-canonical", orth_centre=orth_centre)

        assert is_canonical(mps_left)
        assert is_canonical(mps_right)
        assert is_canonical(mps_mixed)

        explicit_from_right = mps_right.explicit()
        explicit_from_left = mps_left.explicit()
        explicit_from_mixed = mps_mixed.explicit()

        assert np.isclose(
            abs(inner_product(mps_right, explicit_from_right.right_canonical())), 1
        )

        assert np.isclose(
            abs(inner_product(mps_right, explicit_from_left.right_canonical())), 1
        )

        assert np.isclose(
            abs(inner_product(mps_left, explicit_from_right.left_canonical())), 1
        )

        assert np.isclose(
            abs(inner_product(mps_left, explicit_from_left.left_canonical())), 1
        )

        assert np.isclose(
            abs(
                inner_product(
                    mps_mixed, explicit_from_mixed.mixed_canonical(orth_centre)
                )
            ),
            1,
        )

        assert np.isclose(
            abs(
                inner_product(
                    mps_mixed, explicit_from_right.mixed_canonical(orth_centre)
                )
            ),
            1,
        )

        assert np.isclose(
            abs(
                inner_product(
                    mps_mixed, explicit_from_left.mixed_canonical(orth_centre)
                )
            ),
            1,
        )
