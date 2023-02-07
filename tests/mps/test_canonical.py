"""Tests for the :class:`CanonicalMPS` class."""

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
    create_simple_product_state,
)
from mdopt.contractor.contractor import apply_one_site_operator, mps_mpo_contract
from mdopt.mps.canonical import CanonicalMPS
from mdopt.utils.utils import mpo_from_matrix


def test_canonical_init():
    """Tests for the ``__init__`` and ``__len__`` methods of the :class:`CanonicalMPS` class."""

    for _ in range(10):

        num_sites = np.random.randint(4, 9)
        tensors = [
            np.random.uniform(low=0, high=1, size=(1, 2, 1))
            + 1j * np.random.uniform(low=0, high=1, size=(1, 2, 1))
            for _ in range(num_sites)
        ]
        tensors = [tensor.astype(np.complex128) for tensor in tensors]
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
            mps.tensors[0] = np.expand_dims(mps.tensors[0], 0)
            CanonicalMPS(tensors=mps.tensors)


def test_canonical_copy():
    """Test for the ``copy`` method of the :class:`CanonicalMPS` class."""

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
    """Test for the ``reverse`` method of the :class:`CanonicalMPS` class."""

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
    """Test for the ``conjugate`` method of the :class:`CanonicalMPS` class."""

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


def test_canonical_one_site_tensor():
    """Test for the ``one_site_tensor`` method of the :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")
        site = np.random.randint(0, num_sites)

        with pytest.raises(ValueError):
            mps.one_site_tensor(-100)

        assert np.isclose(mps.one_site_tensor(site), mps.tensors[site]).all()


def test_canonical_one_site_tensor_iter():
    """Test for the ``one_site_tensor_iter`` method of the :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")

        assert isinstance(mps.one_site_tensor_iter(), Iterable)


def test_canonical_two_site_tensor_next():
    """Test for the ``two_site_tensor_next`` method of the :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")
        site = np.random.randint(0, num_sites - 1)

        with pytest.raises(ValueError):
            mps.two_site_tensor_next(-2)

        assert np.isclose(
            mps.two_site_tensor_next(site),
            contract(
                "ijk, klm -> ijlm",
                mps.one_site_tensor(site),
                mps.one_site_tensor(site + 1),
            ),
        ).all()


def test_canonical_two_site_tensor_prev():
    """Test for the ``two_site_tensor_prev`` method of the :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")
        site = np.random.randint(1, num_sites)

        with pytest.raises(ValueError):
            mps.two_site_tensor_prev(-2)

        assert np.isclose(
            mps.two_site_tensor_prev(site),
            contract(
                "ijk, klm -> ijlm",
                mps.one_site_tensor(site - 1),
                mps.one_site_tensor(site),
            ),
        ).all()


def test_canonical_two_site_tensor_next_iter():
    """Test for the ``two_site_tensor_next_iter`` method of :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")

        assert isinstance(mps.two_site_tensor_next_iter(), Iterable)


def test_canonical_two_site_tensor_prev_iter():
    """Test for the ``two_site_tensor_prev_iter`` method of :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")

        assert isinstance(mps.two_site_tensor_prev_iter(), Iterable)


def test_canonical_dense():
    """Test for the ``dense`` method of the :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")

        assert np.isclose(psi, mps.dense(flatten=True)).all()


def test_canonical_density_mpo():
    """Test for the ``density_mpo`` method of the :class:`CanonicalMPS` class."""

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
    """Test for the ``entanglement_entropy`` method of the :class:`CanonicalMPS` class."""

    num_sites = 4

    psi_two_body_dimer = 1 / np.sqrt(2) * np.array([0, -1, 1, 0], dtype=np.float32)
    psi_many_body_dimer = reduce(np.kron, [psi_two_body_dimer] * num_sites)

    mps_dimer = mps_from_dense(
        psi_many_body_dimer, form="Right-canonical", tolerance=1e-6
    )
    mps_dimer.orth_centre = 0

    entropy_list = np.array(mps_dimer.entanglement_entropy(tolerance=1e-1))

    correct_entropy_list = np.array([0, np.log(2), 0, np.log(2), 0, np.log(2), 0])

    zeros = entropy_list - correct_entropy_list

    assert np.allclose(np.linalg.norm(zeros), 0, atol=1e-6)


def test_canonical_move_orth_centre():
    """Test for the ``move_orth_centre`` method of the :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)

        orth_centre_init = np.random.randint(num_sites)
        mps_mixed_init = mps_from_dense(
            psi, form="Mixed-canonical", orth_centre=orth_centre_init
        )
        orth_centre_final = np.random.randint(num_sites)
        mps_mixed_final = mps_mixed_init.move_orth_centre(orth_centre_final)
        mps_product = create_simple_product_state(
            num_sites=num_sites, form="Right-canonical"
        )

        with pytest.raises(ValueError):
            mps_mixed_init.move_orth_centre(-3)

        assert is_canonical(mps_mixed_init)
        assert is_canonical(mps_mixed_final)
        assert find_orth_centre(mps_mixed_init) == [orth_centre_init]
        assert find_orth_centre(mps_mixed_final) == [orth_centre_final]
        assert find_orth_centre(mps_product) == [0]


def test_canonical_move_orth_centre_to_border():
    """Test for the ``move_orth_centre_to_border`` method of :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps_mixed_init_1 = mps_from_dense(psi, form="Mixed-canonical", orth_centre=1)
        mps_mixed_init_2 = mps_from_dense(
            psi, form="Mixed-canonical", orth_centre=num_sites - 2
        )
        mps_product = create_simple_product_state(
            num_sites=num_sites, form="Right-canonical"
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

        mps, position = mps_product.move_orth_centre_to_border()
        for tensor_1, tensor_2 in zip(mps.tensors, mps_product.tensors):
            assert np.isclose(tensor_1, tensor_2).all()
        assert position == "first"


def test_canonical_explicit():
    """
    Tests for the ``explicit``, ``right_canonical``, ``left_canonical`` and
    ``mixed_canonical`` methods of the :class:`CanonicalMPS` class.
    """

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        left = mps_from_dense(psi, form="Left-canonical")
        right = mps_from_dense(psi, form="Right-canonical")
        orth_centre = np.random.randint(num_sites)
        mixed = mps_from_dense(psi, form="Mixed-canonical", orth_centre=orth_centre)

        assert is_canonical(left)
        assert is_canonical(right)
        assert is_canonical(mixed)

        explicit_from_right = right.explicit()
        explicit_from_left = left.explicit()
        explicit_from_mixed = mixed.explicit()

        mixed_from_right = right.mixed_canonical(orth_centre=orth_centre)
        mixed_from_left = left.mixed_canonical(orth_centre=orth_centre)

        assert isinstance(mixed.right_canonical(), CanonicalMPS)
        assert isinstance(mixed.left_canonical(), CanonicalMPS)
        assert isinstance(mixed_from_right, CanonicalMPS)
        assert isinstance(mixed_from_left, CanonicalMPS)

        assert np.isclose(explicit_from_right.tolerance, 1e-12)
        assert np.isclose(explicit_from_left.tolerance, 1e-12)
        assert np.isclose(explicit_from_mixed.tolerance, 1e-12)

        assert np.isclose(
            abs(inner_product(right, explicit_from_right.right_canonical())), 1
        )

        assert np.isclose(
            abs(inner_product(right, explicit_from_left.right_canonical())), 1
        )

        assert np.isclose(
            abs(inner_product(left, explicit_from_right.left_canonical())), 1
        )

        assert np.isclose(
            abs(inner_product(left, explicit_from_left.left_canonical())), 1
        )

        assert np.isclose(
            abs(inner_product(mixed, explicit_from_mixed.mixed_canonical(orth_centre))),
            1,
        )

        assert np.isclose(
            abs(inner_product(mixed, explicit_from_right.mixed_canonical(orth_centre))),
            1,
        )

        assert np.isclose(
            abs(inner_product(mixed, explicit_from_left.mixed_canonical(orth_centre))),
            1,
        )

        assert np.isclose(
            abs(
                inner_product(
                    mixed_from_left, explicit_from_left.mixed_canonical(orth_centre)
                )
            ),
            1,
        )

        assert np.isclose(
            abs(
                inner_product(
                    mixed_from_right, explicit_from_left.mixed_canonical(orth_centre)
                )
            ),
            1,
        )


def test_canonical_norm():
    """
    Test for the ``norm`` method of the :class:`CanonicalMPS` class.
    """

    num_sites = np.random.randint(4, 9)

    for _ in range(10):

        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")

        assert isinstance(mps.norm(), np.float64)
        assert np.isclose(mps.norm() - abs(inner_product(mps, mps)) ** 2, 0)


def test_canonical_one_site_expectation_value():
    """
    Test for the ``one_site_expectation_value`` method of the :class:`CanonicalMPS` class.
    """

    num_sites = np.random.randint(4, 9)
    phys_dim = np.random.randint(2, 4)

    for _ in range(10):

        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        mps = mps_from_dense(psi, phys_dim=phys_dim, form="Right-canonical")
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


def test_canonical_two_site_expectation_value():
    """
    Test for the ``two_site_expectation_value`` method of the :class:`CanonicalMPS` class.
    """

    num_sites = np.random.randint(4, 9)
    phys_dim = np.random.randint(2, 4)

    for _ in range(10):

        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        operator = np.random.uniform(
            size=(phys_dim, phys_dim, phys_dim, phys_dim)
        ) + 1j * np.random.uniform(size=(phys_dim, phys_dim, phys_dim, phys_dim))
        site = int(np.random.randint(num_sites - 1))
        mps = mps_from_dense(
            psi, phys_dim=phys_dim, form="Mixed-canonical", orth_centre=site
        )
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


def test_canonical_marginal():
    """
    Test for the ``marginal`` method of the :class:`CanonicalMPS` class.
    """

    num_sites = np.random.randint(4, 9)
    phys_dim = np.random.randint(2, 4)

    for _ in range(10):

        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        mps = mps_from_dense(psi, phys_dim=phys_dim, form="Right-canonical")

        sites_all = list(range(num_sites))
        sites_to_marginalise = []
        for site in sites_all:
            if np.random.uniform() < 1 / 2:
                sites_to_marginalise.append(site)
        sites_left = [site for site in sites_all if site not in sites_to_marginalise]

        mps_marginalised = mps.marginal(sites_to_marginalise)

        with pytest.raises(ValueError):
            mps.marginal([100, 200])

        if isinstance(mps_marginalised, CanonicalMPS):
            assert mps_marginalised.num_sites == len(sites_left)
            assert is_canonical(mps_marginalised)
        else:
            assert isinstance(mps_marginalised, np.complex128)
