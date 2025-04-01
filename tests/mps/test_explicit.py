"""Tests for the ``mdopt.mps.explicit.ExplicitMPS`` class."""

from functools import reduce
from typing import Iterable
import numpy as np
import pytest
from opt_einsum import contract

from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.explicit import ExplicitMPS
from mdopt.utils.utils import mpo_from_matrix
from mdopt.contractor.contractor import apply_one_site_operator, mps_mpo_contract
from mdopt.mps.utils import (
    create_state_vector,
    is_canonical,
    mps_from_dense,
    create_simple_product_state,
    inner_product,
    find_orth_centre,
)


def test_explicit_init():
    """Tests for the ``__init__`` and ``__len__`` methods of the :class:`ExplicitMPS` class."""

    for _ in range(10):
        num_sites = np.random.randint(4, 9)

        product_mps = create_simple_product_state(
            num_sites=num_sites,
            which="0",
            phys_dim=2,
            form="Explicit",
        )
        product_tensor = np.array([1.0, 0.0]).reshape((1, 2, 1))
        product_tensors = [product_tensor for _ in range(num_sites)]

        assert np.isclose(product_mps.tensors, product_tensors).all()
        assert np.isclose(
            product_mps.singular_values, [[1.0] for _ in range(num_sites + 1)]
        ).all()
        assert np.isclose(
            product_mps.bond_dimensions, [1 for _ in range(product_mps.num_bonds)]
        ).all()
        assert np.isclose(
            product_mps.phys_dimensions, [2 for _ in range(product_mps.num_sites)]
        ).all()

        psi = create_state_vector(num_sites)
        tolerance = 1e-12
        chi_max = 1e4
        mps = mps_from_dense(psi, form="Explicit")

        assert np.isclose(mps.num_sites, num_sites).all()
        assert np.isclose(mps.num_bonds, num_sites - 1).all()
        assert np.isclose(mps.num_singval_mat, len(mps.singular_values)).all()
        assert mps.dtype == np.dtype(np.complex128)
        assert np.isclose(mps.tolerance, tolerance).all()
        assert np.isclose(mps.chi_max, chi_max).all()
        assert np.isclose(len(mps), mps.num_sites).all()
        assert isinstance(iter(mps), zip)

        with pytest.raises(ValueError):
            ExplicitMPS(
                tensors=product_tensors,
                singular_values=product_mps.singular_values + [1.0],
            )

        with pytest.raises(ValueError):
            mps.tensors[0] = np.expand_dims(mps.tensors[0], 0)
            ExplicitMPS(
                tensors=mps.tensors,
                singular_values=mps.singular_values,
            )

        with pytest.raises(ValueError):
            ExplicitMPS(
                tensors=mps.tensors, singular_values=mps.singular_values + [0.0]
            )

        with pytest.raises(ValueError):
            two_tensors = [
                np.array([1.0, 0.0]).reshape((1, 2, 1)),
                np.array([1.0, 0.0]).reshape((1, 2, 1)),
            ]
            sing_val = [5.0]
            ExplicitMPS(tensors=two_tensors, singular_values=sing_val)


def test_explicit_copy():
    """Test for the ``copy`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")
        mps_copied = mps.copy()

        for tensor, tensor_copy in zip(mps.tensors, mps_copied.tensors):
            assert np.isclose(tensor, tensor_copy).all()
        assert np.isclose(mps.num_sites, mps_copied.num_sites).all()
        assert np.isclose(mps.num_bonds, mps_copied.num_bonds).all()
        assert np.isclose(mps.bond_dimensions, mps_copied.bond_dimensions).all()
        assert np.isclose(mps.phys_dimensions, mps_copied.phys_dimensions).all()
        for sing_vals, sing_vals_copied in zip(
            mps.singular_values, mps_copied.singular_values
        ):
            assert np.isclose(sing_vals, sing_vals_copied).all()
        assert np.isclose(mps.num_singval_mat, mps_copied.num_singval_mat).all()
        assert mps.dtype == mps_copied.dtype
        assert np.isclose(mps.tolerance, mps_copied.tolerance).all()
        assert np.isclose(mps.chi_max, mps_copied.chi_max).all()
        assert np.isclose(len(mps), len(mps_copied)).all()


def test_explicit_reverse():
    """Test for the ``reverse`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")
        mps_reversed = mps.reverse()

        for tensor, reversed_tensor in zip(reversed(mps.tensors), mps_reversed.tensors):
            assert np.isclose(tensor, np.transpose(reversed_tensor)).all()
        assert np.isclose(mps_reversed.num_sites, mps.num_sites).all()
        assert np.isclose(mps_reversed.num_bonds, mps.num_bonds).all()
        assert np.isclose(mps_reversed.bond_dimensions, mps.bond_dimensions).all()
        assert np.isclose(mps_reversed.phys_dimensions, mps.phys_dimensions).all()
        for sing_vals, sing_vals_reversed in zip(
            mps_reversed.singular_values, reversed(mps.singular_values)
        ):
            assert np.isclose(sing_vals, sing_vals_reversed).all()
        assert mps_reversed.dtype == mps.dtype
        assert np.isclose(mps_reversed.tolerance, mps.tolerance).all()
        assert np.isclose(mps_reversed.chi_max, mps.chi_max).all()
        assert np.isclose(len(mps_reversed), len(mps)).all()


def test_explicit_conjugate():
    """Test for the ``conjugate`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")
        mps_conjugated = mps.conjugate()

        for tensor, conjugated_tensor in zip(mps.tensors, mps_conjugated.tensors):
            assert np.isclose(tensor, np.conjugate(conjugated_tensor)).all()
        assert np.isclose(mps.num_sites, mps_conjugated.num_sites).all()
        assert np.isclose(mps.num_bonds, mps_conjugated.num_bonds).all()
        assert np.isclose(mps.bond_dimensions, mps_conjugated.bond_dimensions).all()
        assert np.isclose(mps.phys_dimensions, mps_conjugated.phys_dimensions).all()

        for singular_values, conjugated_singular_values in zip(
            mps.singular_values, mps_conjugated.singular_values
        ):
            assert np.isclose(
                singular_values, np.conjugate(conjugated_singular_values)
            ).all()
        assert np.isclose(mps.num_singval_mat, mps_conjugated.num_singval_mat).all()
        assert mps.dtype == mps_conjugated.dtype
        assert np.isclose(mps.tolerance, mps_conjugated.tolerance).all()
        assert np.isclose(mps.chi_max, mps_conjugated.chi_max).all()
        assert np.isclose(len(mps), len(mps_conjugated)).all()


def test_explicit_one_site_left_iso():
    """Test for the ``one_site_left_iso`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        for site in range(num_sites):
            isometry = mps.one_site_left_iso(site)

            to_be_identity = contract(
                "ijk, ijl -> kl", isometry, np.conjugate(isometry), optimize=[(0, 1)]
            )

            assert np.isclose(
                np.linalg.norm(to_be_identity - np.identity(to_be_identity.shape[0])), 0
            )

        with pytest.raises(ValueError):
            mps.one_site_left_iso(num_sites + 100)


def test_explicit_one_site_right_iso():
    """Test for the ``one_site_right_iso`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        for site in range(num_sites):
            isometry = mps.one_site_right_iso(site)

            to_be_identity = contract(
                "ijk, ljk -> il", isometry, np.conjugate(isometry), optimize=[(0, 1)]
            )

            assert np.isclose(
                np.linalg.norm(to_be_identity - np.identity(to_be_identity.shape[0])), 0
            )

        with pytest.raises(ValueError):
            mps.one_site_right_iso(num_sites + 100)


def test_explicit_one_left_iso_iter():
    """Test for the ``one_site_left_iso_iter`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        assert isinstance(mps.one_site_left_iso_iter(), Iterable)


def test_explicit_one_right_iso_iter():
    """Test for the ``one_site_right_iso_iter`` method of :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        assert isinstance(mps.one_site_right_iso_iter(), Iterable)


def test_explicit_two_site_left_iso():
    """Test for the ``two_site_left_iso`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        with pytest.raises(ValueError):
            mps.two_site_left_iso(num_sites)

        for site in range(num_sites - 1):
            two_site_left_iso = mps.two_site_left_iso(site)

            to_be_identity = contract(
                "ijkl, ijkm -> lm",
                two_site_left_iso,
                np.conjugate(two_site_left_iso),
                optimize=[(0, 1)],
            )

            assert np.isclose(
                np.linalg.norm(to_be_identity - np.identity(to_be_identity.shape[-1])),
                0,
            )


def test_explicit_two_site_right_iso():
    """Test for the ``two_site_right_iso`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        with pytest.raises(ValueError):
            mps.two_site_right_iso(num_sites)

        for site in range(num_sites - 1):
            two_site_right_iso = mps.two_site_right_iso(site)

            to_be_identity = contract(
                "ijkl, mjkl -> im",
                two_site_right_iso,
                np.conjugate(two_site_right_iso),
                optimize=[(0, 1)],
            )

            assert np.isclose(
                np.linalg.norm(to_be_identity - np.identity(to_be_identity.shape[-1])),
                0,
            )


def test_explicit_two_site_iter():
    """
    Tests for the ``two_site_right_iso_iter`` and ``two_site_left_iso_iter``
    methods of the :class:`ExplicitMPS` class.
    """

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        assert isinstance(mps.two_site_right_iso_iter(), Iterable)
        assert isinstance(mps.two_site_left_iso_iter(), Iterable)


def test_explicit_dense():
    """Test for the ``dense`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")
        shape = [2] * num_sites

        assert np.isclose(psi, mps.dense(flatten=True, renormalise=False)).all()
        assert np.isclose(
            np.linalg.norm(mps.dense(flatten=True, renormalise=True, norm=1), ord=1), 1
        )
        assert np.isclose(
            psi.reshape(shape), mps.dense(flatten=False, renormalise=False)
        ).all()


def test_explicit_density_mpo():
    """Test for the ``density_mpo`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        density_mpo = mps.density_mpo()

        # Juggle the dimensions around to apply the ``reduce`` function later
        # which is used to create a density mpo to compare the method against.
        for i in range(num_sites):
            density_mpo[i] = density_mpo[i].transpose((0, 3, 2, 1))

        density_matrix_mpo = reduce(
            lambda a, b: np.tensordot(a, b, (-1, 0)), density_mpo
        )

        # Get rid of ghost dimensions of the MPO.
        density_matrix_mpo = density_matrix_mpo.squeeze()

        # Reshaping to the right order of indices.
        correct_order = list(range(0, 2 * num_sites, 2)) + list(
            range(1, 2 * num_sites, 2)
        )

        density_matrix_mpo = density_matrix_mpo.transpose(correct_order)
        # Reshaping to the matrix form.
        density_matrix_mpo = density_matrix_mpo.reshape((2**num_sites, 2**num_sites))

        # Original density matrix.
        density_matrix = np.tensordot(psi, np.conjugate(psi), 0)

        assert np.isclose(np.trace(density_matrix), 1)
        assert np.isclose(
            np.linalg.norm(density_matrix - np.conjugate(density_matrix).T), 0
        )

        assert np.isclose(np.trace(density_matrix_mpo), 1)
        assert np.isclose(
            np.linalg.norm(density_matrix_mpo - np.conjugate(density_matrix_mpo).T),
            0,
        )


def test_explicit_entanglement_entropy():
    """Test for the ``entanglement_entropy`` method of the :class:`ExplicitMPS` class."""

    num_sites = 4

    psi_two_body_dimer = 1 / np.sqrt(2) * np.array([0, -1, 1, 0], dtype=float)
    psi_many_body_dimer = reduce(np.kron, [psi_two_body_dimer] * num_sites)

    mps_dimer = mps_from_dense(psi_many_body_dimer, form="Explicit", tolerance=1e-6)

    entropy_list = np.array(mps_dimer.entanglement_entropy())

    correct_entropy_list = np.array([0, np.log(2), 0, np.log(2), 0, np.log(2), 0])

    zeros = entropy_list - correct_entropy_list

    assert np.allclose(np.linalg.norm(zeros), 0, atol=1e-6)


def test_explicit_right_canonical():
    """Test for the ``right_canonical`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        mps_right = mps.right_canonical()

        assert is_canonical(mps_right)
        assert isinstance(mps_right, CanonicalMPS)
        assert np.isclose(abs(inner_product(mps_right, mps_right)), 1)
        assert len(find_orth_centre(mps_right)) == 1

        for i in range(num_sites):
            assert mps.tensors[i].shape == mps_right.tensors[i].shape

        for i, _ in enumerate(mps_right.tensors):
            to_be_identity_right = contract(
                "ijk, ljk -> il",
                mps_right.tensors[i],
                np.conjugate(mps_right.tensors[i]),
                optimize=[(0, 1)],
            )

            identity_right = np.identity(to_be_identity_right.shape[0], dtype=float)

            assert np.isclose(np.linalg.norm(to_be_identity_right - identity_right), 0)


def test_explicit_left_canonical():
    """Test for the ``left_canonical`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        mps_left = mps.left_canonical()

        assert is_canonical(mps_left)
        assert isinstance(mps_left, CanonicalMPS)
        assert np.isclose(abs(inner_product(mps_left, mps_left)), 1)
        assert len(find_orth_centre(mps_left)) == 1

        for i in range(num_sites):
            assert mps.tensors[i].shape == mps_left.tensors[i].shape

        for i, tensor in enumerate(mps_left.tensors):
            to_be_identity_left = contract(
                "ijk, ijl -> kl",
                tensor,
                np.conjugate(tensor),
                optimize=[(0, 1)],
            )

            identity_left = np.identity(to_be_identity_left.shape[0], dtype=float)

            assert np.isclose(np.linalg.norm(to_be_identity_left - identity_left), 0)


def test_explicit_mixed_canonical():
    """Test for the ``mixed_canonical`` method of the :class:`ExplicitMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        orth_centre_index = np.random.randint(num_sites)
        mps_mixed = mps.mixed_canonical(orth_centre_index)

        for i in range(num_sites):
            assert mps.tensors[i].shape == mps_mixed.tensors[i].shape
        assert is_canonical(mps_mixed)
        assert isinstance(mps_mixed, CanonicalMPS)
        assert np.isclose(abs(inner_product(mps_mixed, mps_mixed)), 1)
        assert find_orth_centre(mps_mixed) == [orth_centre_index]

        with pytest.raises(ValueError):
            mps.mixed_canonical(num_sites + 100)


def test_explicit_norm():
    """
    Test for the ``norm`` method of the :class:`ExplicitMPS` class.
    """

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Explicit")

        assert isinstance(mps.norm(), float)
        assert np.isclose(abs(mps.norm() - abs(inner_product(mps, mps)) ** 2), 0)


def test_explicit_one_site_expectation_value():
    """
    Test for the ``one_site_expectation_value`` method of the :class:`ExplicitMPS` class.
    """

    num_sites = np.random.randint(4, 9)
    phys_dim = np.random.randint(2, 4)

    for _ in range(10):
        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        mps = mps_from_dense(psi, phys_dim=phys_dim, form="Explicit")
        mps_copy = mps.copy()
        operator = np.random.uniform(
            size=(phys_dim, phys_dim)
        ) + 1j * np.random.uniform(size=(phys_dim, phys_dim))
        site = int(np.random.randint(num_sites))

        with pytest.raises(ValueError):
            mps.one_site_expectation_value(100, operator)
        with pytest.raises(ValueError):
            mps.one_site_expectation_value(
                site, np.random.uniform(size=(phys_dim, phys_dim, phys_dim))
            )

        exp_value = mps.one_site_expectation_value(site, operator)

        mps.tensors[site] = apply_one_site_operator(mps.tensors[site], operator)
        exp_value_to_compare = inner_product(mps_copy, mps)

        assert np.isclose(abs(exp_value - exp_value_to_compare) ** 2, 0)


def test_explicit_two_site_expectation_value():
    """
    Test for the ``two_site_expectation_value`` method of the :class:`ExplicitMPS` class.
    """

    num_sites = np.random.randint(4, 9)
    phys_dim = np.random.randint(2, 4)

    for _ in range(10):
        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        operator = np.random.uniform(
            size=(phys_dim, phys_dim, phys_dim, phys_dim)
        ) + 1j * np.random.uniform(size=(phys_dim, phys_dim, phys_dim, phys_dim))
        site = int(np.random.randint(num_sites - 1))
        mps = mps_from_dense(psi, phys_dim=phys_dim, form="Explicit")
        mps_copy = mps.copy()
        operator_mpo = mpo_from_matrix(
            matrix=operator, num_sites=2, phys_dim=phys_dim, interlaced=True
        )

        with pytest.raises(ValueError):
            mps.two_site_expectation_value(100, operator_mpo)
        with pytest.raises(ValueError):
            mps.two_site_expectation_value(
                site, np.random.uniform(size=(phys_dim, phys_dim, phys_dim))
            )

        exp_value = mps.two_site_expectation_value(site, operator)

        mps = mps_mpo_contract(mps=mps, mpo=operator_mpo, start_site=site)
        exp_value_to_compare = inner_product(mps_copy, mps)

        assert np.isclose(abs(exp_value - exp_value_to_compare) ** 2, 0)


def test_explicit_compress_bond():
    """
    Test for the ``compress_bond`` method of the :class:`ExplicitMPS` class.
    """

    num_sites = 10
    phys_dim = 2
    bond_to_compress = 4
    for _ in range(10):
        # Testing the maximum bond dimension control
        chi_max = 7
        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        mps = mps_from_dense(
            state_vector=psi, phys_dim=phys_dim, form="Explicit", tolerance=np.inf
        )
        mps_compressed, truncation_error = mps.compress_bond(
            bond=bond_to_compress,
            chi_max=chi_max,
            renormalise=False,
            return_truncation_error=True,
        )
        assert mps_compressed.bond_dimensions[bond_to_compress] <= chi_max
        assert truncation_error >= 0
        assert inner_product(mps_compressed, mps_compressed) <= 1
        assert np.isclose(mps_compressed.norm() + truncation_error - 1, 0)
        with pytest.raises(ValueError):
            mps.compress_bond(bond=100)

        # Testing the spectrum cut control
        cut = 1e-1
        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        mps = mps_from_dense(
            state_vector=psi, phys_dim=phys_dim, form="Explicit", tolerance=np.inf
        )
        mps_compressed, truncation_error = mps.compress_bond(
            bond=bond_to_compress,
            cut=cut,
            renormalise=False,
            return_truncation_error=True,
        )
        singular_values = mps.singular_values[bond_to_compress + 1]
        for singular_value in singular_values:
            assert singular_value >= cut
        assert truncation_error >= 0
        assert mps_compressed.norm() <= 1
        assert np.isclose(mps_compressed.norm() + truncation_error - 1, 0)
        assert inner_product(mps_compressed, mps_compressed) <= 1
        assert inner_product(mps, mps_compressed) <= 1

        # Testing the renormalisation option
        cut = 1e-1
        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        mps = mps_from_dense(
            state_vector=psi, phys_dim=phys_dim, form="Explicit", tolerance=np.inf
        )
        mps_compressed, truncation_error = mps.compress_bond(
            bond=bond_to_compress,
            cut=cut,
            renormalise=True,
            return_truncation_error=True,
        )
        singular_values = mps.singular_values[bond_to_compress + 1]
        assert np.isclose(np.linalg.norm(singular_values) - 1, 0)
        for singular_value in singular_values:
            assert singular_value >= cut
        assert truncation_error >= 0
        assert np.isclose(mps_compressed.norm() - 1, 0)
        assert np.isclose(inner_product(mps_compressed, mps_compressed) - 1, 0)
        assert np.isclose(inner_product(mps, mps_compressed) - 1, 0)

        # Testing that for no cut the mps stays the same
        cut = 1e-12
        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        mps = mps_from_dense(
            state_vector=psi, phys_dim=phys_dim, form="Explicit", tolerance=np.inf
        )
        mps_compressed, truncation_error = mps.compress_bond(
            bond=bond_to_compress,
            cut=cut,
            renormalise=False,
            return_truncation_error=True,
        )
        singular_value = mps_compressed.singular_values[bond_to_compress + 1]
        for singular_value in singular_values:
            assert singular_value >= cut
        assert np.isclose(truncation_error, 0)
        assert np.isclose(mps_compressed.norm() - 1, 0)
        assert np.isclose(mps_compressed.norm() + truncation_error - 1, 0)
        assert np.isclose(inner_product(mps_compressed, mps_compressed) - 1, 0)
        assert np.isclose(inner_product(mps, mps_compressed) - 1, 0)


def test_explicit_compress():
    """
    Test for the ``compress`` method of the :class:`ExplicitMPS` class.
    """

    for _ in range(10):
        renormalise = False
        num_sites = 10
        phys_dim = 2
        chi_max = 3
        cut = 1e-12
        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        mps = mps_from_dense(
            state_vector=psi, phys_dim=phys_dim, form="Explicit", tolerance=np.inf
        )
        mps_compressed, truncation_errors = mps.compress(
            chi_max=chi_max,
            cut=cut,
            renormalise=renormalise,
            return_truncation_errors=True,
        )
        assert inner_product(mps, mps_compressed) <= 1
        assert mps_compressed.norm() <= 1
        for bond_dim in mps_compressed.bond_dimensions:
            assert bond_dim <= chi_max
        for truncation_error in truncation_errors:
            assert truncation_error >= 0
        assert np.sum(truncation_errors) <= mps.num_singval_mat
