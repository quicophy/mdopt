"""Tests for the :class:`CanonicalMPS` class."""

from functools import reduce
from typing import Iterable
import numpy as np
import pytest
from opt_einsum import contract

from mdopt.mps.utils import (
    create_simple_product_state,
    create_state_vector,
    find_orth_center,
    mps_from_dense,
    inner_product,
    is_canonical,
)
from mdopt.contractor.contractor import apply_one_site_operator, mps_mpo_contract
from mdopt.mps.canonical import CanonicalMPS
from mdopt.utils.utils import mpo_from_matrix, split_two_site_tensor


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
        orth_center = np.random.randint(low=0, high=num_sites)
        tolerance = 1e-10
        chi_max = 1e3
        mps = CanonicalMPS(
            tensors=tensors,
            orth_center=orth_center,
            tolerance=tolerance,
            chi_max=chi_max,
        )

        for mps_tensor, tensor in zip(mps.tensors, tensors):
            assert np.isclose(mps_tensor, tensor).all()
        assert np.isclose(mps.num_sites, num_sites).all()
        assert np.isclose(mps.num_bonds, num_sites - 1).all()
        assert np.isclose(mps.bond_dimensions, [1 for _ in range(mps.num_bonds)]).all()
        assert np.isclose(mps.phys_dimensions, [2 for _ in range(mps.num_sites)]).all()
        assert mps.orth_center == orth_center
        assert mps.dtype == np.dtype(np.complex128)
        assert np.isclose(mps.tolerance, tolerance).all()
        assert np.isclose(mps.chi_max, chi_max).all()
        assert np.isclose(len(mps), mps.num_sites).all()

        with pytest.raises(ValueError):
            CanonicalMPS(tensors=tensors, orth_center=num_sites + 1)

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
        assert mps.orth_center == mps_copied.orth_center
        assert mps.dtype == mps_copied.dtype
        assert np.isclose(mps.tolerance, mps_copied.tolerance).all()
        assert np.isclose(mps.chi_max, mps_copied.chi_max).all()
        assert np.isclose(len(mps), len(mps_copied)).all()


def test_canonical_reverse():
    """Test for the ``reverse`` method of the :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Mixed-canonical", orth_center=num_sites - 2)
        mps_reversed = mps.reverse()

        for tensor, reversed_tensor in zip(reversed(mps.tensors), mps_reversed.tensors):
            assert np.isclose(tensor, np.transpose(reversed_tensor)).all()
        assert np.isclose(mps_reversed.num_sites, mps.num_sites).all()
        assert np.isclose(mps_reversed.num_bonds, mps.num_bonds).all()
        assert np.isclose(mps_reversed.bond_dimensions, mps.bond_dimensions).all()
        assert np.isclose(mps_reversed.phys_dimensions, mps.phys_dimensions).all()
        assert mps_reversed.orth_center == mps.num_sites - 1 - mps.orth_center
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
        assert mps.orth_center == mps_conjugated.orth_center
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
        shape = [2] * num_sites

        assert np.isclose(psi, mps.dense(flatten=True, renormalize=False)).all()
        assert np.isclose(
            np.linalg.norm(mps.dense(flatten=True, renormalize=True, norm=1), ord=1), 1
        )
        assert np.isclose(
            psi.reshape(shape), mps.dense(flatten=False, renormalize=False)
        ).all()


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

    psi_two_body_dimer = 1 / np.sqrt(2) * np.array([0, -1, 1, 0], dtype=float)
    psi_many_body_dimer = reduce(np.kron, [psi_two_body_dimer] * num_sites)

    mps_dimer = mps_from_dense(
        psi_many_body_dimer, form="Right-canonical", tolerance=1e-6
    )
    mps_dimer.orth_center = 0

    entropy_list = np.array(mps_dimer.entanglement_entropy(tolerance=1e-1))

    # Dimers span sites (0,1), (2,3), ... so the cut after site 0 falls INSIDE
    # the first dimer: the sequence starts at log(2), not at 0.
    correct_entropy_list = np.array(
        [np.log(2), 0, np.log(2), 0, np.log(2), 0, np.log(2)]
    )

    zeros = entropy_list - correct_entropy_list

    assert np.allclose(np.linalg.norm(zeros), 0, atol=1e-6)


def test_canonical_move_orth_center():
    """Test for the ``move_orth_center`` method of the :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)

        orth_center_init = np.random.randint(num_sites)
        mps_mixed_init = mps_from_dense(
            psi, form="Mixed-canonical", orth_center=orth_center_init
        )
        orth_center_final = np.random.randint(num_sites)
        mps_mixed_final = mps_mixed_init.move_orth_center(
            final_pos=orth_center_final, return_singular_values=False, renormalize=False
        )
        mps_mixed_final_renorm = mps_mixed_init.move_orth_center(
            final_pos=orth_center_final, return_singular_values=False, renormalize=True
        )
        mps_product = create_simple_product_state(
            num_sites=num_sites, form="Right-canonical"
        )

        with pytest.raises(ValueError):
            mps_mixed_init.move_orth_center(-3)

        assert is_canonical(mps_mixed_init)
        assert is_canonical(mps_mixed_final)
        assert find_orth_center(mps_mixed_init) == [orth_center_init]
        assert find_orth_center(mps_mixed_final) == [orth_center_final]
        assert find_orth_center(mps_mixed_final_renorm) == [orth_center_final]
        assert find_orth_center(mps_product) == [0]
        assert np.isclose(mps_mixed_final_renorm.norm() - 1, 0)


def test_canonical_move_orth_center_to_border():
    """Test for the ``move_orth_center_to_border`` method of :class:`CanonicalMPS` class."""

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps_mixed_init_1 = mps_from_dense(psi, form="Mixed-canonical", orth_center=1)
        mps_mixed_init_2 = mps_from_dense(
            psi, form="Mixed-canonical", orth_center=num_sites - 2
        )
        mps_product = create_simple_product_state(
            num_sites=num_sites, form="Right-canonical"
        )

        mps, position = mps_mixed_init_1.move_orth_center_to_border()
        for tensor_1, tensor_2 in zip(
            mps.tensors, mps_mixed_init_1.move_orth_center(0).tensors
        ):
            assert np.isclose(tensor_1, tensor_2).all()
        assert position == "first"

        mps, position = mps_mixed_init_2.move_orth_center_to_border()
        for tensor_1, tensor_2 in zip(
            mps.tensors, mps_mixed_init_2.move_orth_center(num_sites - 1).tensors
        ):
            assert np.isclose(tensor_1, tensor_2).all()
        assert position == "last"

        with pytest.raises(UnboundLocalError):
            mps, position = mps_product.move_orth_center_to_border()


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
        orth_center = np.random.randint(num_sites)
        mixed = mps_from_dense(psi, form="Mixed-canonical", orth_center=orth_center)

        assert is_canonical(left)
        assert is_canonical(right)
        assert is_canonical(mixed)

        explicit_from_right = right.explicit()
        explicit_from_left = left.explicit()
        explicit_from_mixed = mixed.explicit()

        mixed_from_right = right.mixed_canonical(orth_center=orth_center)
        mixed_from_left = left.mixed_canonical(orth_center=orth_center)

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
            abs(inner_product(mixed, explicit_from_mixed.mixed_canonical(orth_center))),
            1,
        )

        assert np.isclose(
            abs(inner_product(mixed, explicit_from_right.mixed_canonical(orth_center))),
            1,
        )

        assert np.isclose(
            abs(inner_product(mixed, explicit_from_left.mixed_canonical(orth_center))),
            1,
        )

        assert np.isclose(
            abs(
                inner_product(
                    mixed_from_left, explicit_from_left.mixed_canonical(orth_center)
                )
            ),
            1,
        )

        assert np.isclose(
            abs(
                inner_product(
                    mixed_from_right, explicit_from_left.mixed_canonical(orth_center)
                )
            ),
            1,
        )


def test_canonical_explicit_near_zero_singular_values():
    """
    Test that ``explicit()`` does not produce NaN or Inf when the MPS has
    near-zero (but not exactly zero) singular values at some bonds.

    The old implementation used ``np.where(s != 0.0, s, 1.0)`` which only
    guards against exact zeros.  A value like 1e-20 passed through unchanged,
    so ``1/s`` overflowed.  The fixed code uses a machine-epsilon relative
    threshold (LAPACK pinv convention) so those near-zero components are
    suppressed rather than inverted.
    """
    # |00...0> + eps|11...1>: after normalization the interior singular values
    # at every bond are ~eps, exercising the near-zero division path.
    num_sites = 5
    for eps in (1e-20, 1e-15, 1e-10):
        psi = np.zeros(2**num_sites, dtype=complex)
        psi[0] = 1.0
        psi[-1] = eps
        psi /= np.linalg.norm(psi)

        for form in ("Right-canonical", "Left-canonical"):
            mps_canonical = mps_from_dense(psi, form=form)
            mps_explicit = mps_canonical.explicit()

            for i, gamma in enumerate(mps_explicit.tensors):
                assert np.all(np.isfinite(gamma)), (
                    f"Γ[{i}] contains NaN or Inf for eps={eps}, form={form}"
                )
            assert np.all(np.isfinite(mps_explicit.dense())), (
                f"dense() contains NaN or Inf for eps={eps}, form={form}"
            )

    # float32 path
    psi32 = psi.astype(np.complex64)
    mps32 = mps_from_dense(psi32, form="Right-canonical")
    mps32_explicit = mps32.explicit()
    for gamma in mps32_explicit.tensors:
        assert np.all(np.isfinite(gamma)), "float32 Γ tensor contains NaN or Inf"


def test_canonical_norm():
    """
    Test for the ``norm`` method of the :class:`CanonicalMPS` class.
    """

    num_sites = np.random.randint(4, 9)

    for _ in range(10):
        psi = create_state_vector(num_sites)
        mps = mps_from_dense(psi, form="Right-canonical")

        assert isinstance(mps.norm(), float)
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
            psi, phys_dim=phys_dim, form="Mixed-canonical", orth_center=site
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


def test_canonical_compress_bond():
    """
    Test for the ``compress_bond`` method of the :class:`CanonicalMPS` class.
    """

    for _ in range(10):
        # Testing the maximum bond dimension control
        for strategy in ["svd", "qr", "svd_advanced"]:
            num_sites = 10
            phys_dim = 2
            bond_to_compress = 4
            chi_max = 7
            psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
            mps = mps_from_dense(
                state_vector=psi, phys_dim=phys_dim, form="Right-canonical"
            )
            mps_compressed, truncation_error = mps.compress_bond(
                bond=bond_to_compress,
                chi_max=chi_max,
                renormalize=False,
                strategy=strategy,
                return_truncation_error=True,
            )
            assert mps_compressed.bond_dimensions[bond_to_compress] <= chi_max
            assert truncation_error >= 0
            if strategy in ["svd", "svd_advanced"]:
                assert inner_product(mps_compressed, mps_compressed) <= 1
                if strategy == "svd":
                    assert np.isclose(mps_compressed.norm() + truncation_error - 1, 0)
            with pytest.raises(ValueError):
                mps.compress_bond(bond=100)
            with pytest.raises(ValueError):
                mps.compress_bond(bond=0, strategy="strategy")

        # Testing the spectrum cut control
        for strategy in ["svd", "svd_advanced"]:
            num_sites = 10
            phys_dim = 2
            bond_to_compress = 4
            cut = 1e-1
            psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
            mps = mps_from_dense(
                state_vector=psi, phys_dim=phys_dim, form="Right-canonical"
            )
            mps_compressed, truncation_error = mps.compress_bond(
                bond=bond_to_compress,
                cut=cut,
                renormalize=False,
                strategy=strategy,
                return_truncation_error=True,
            )
            tensor_left = mps_compressed.tensors[bond_to_compress]
            tensor_right = mps_compressed.tensors[bond_to_compress + 1]
            two_site_tensor = contract(
                "ijk, klm -> ijlm",
                tensor_left,
                tensor_right,
                optimize=[(0, 1)],
            )
            _, singular_values, _, _ = split_two_site_tensor(
                tensor=two_site_tensor, return_truncation_error=True, strategy="svd"
            )
            for singular_value in singular_values:
                assert singular_value >= cut
            assert truncation_error >= 0
            if strategy == "svd":
                assert mps_compressed.norm() <= 1
                assert np.isclose(mps_compressed.norm() + truncation_error - 1, 0)
            if strategy == "svd_advanced":
                assert inner_product(mps_compressed, mps_compressed) <= 1
                assert inner_product(mps, mps_compressed) <= 1

        # Testing the renormalization option
        for strategy in ["svd", "svd_advanced"]:
            num_sites = 10
            phys_dim = 2
            bond_to_compress = 4
            cut = 1e-1
            psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
            mps = mps_from_dense(
                state_vector=psi, phys_dim=phys_dim, form="Right-canonical"
            )
            mps_compressed, truncation_error = mps.compress_bond(
                bond=bond_to_compress,
                cut=cut,
                renormalize=True,
                strategy=strategy,
                return_truncation_error=True,
            )
            tensor_left = mps_compressed.tensors[bond_to_compress]
            tensor_right = mps_compressed.tensors[bond_to_compress + 1]
            two_site_tensor = contract(
                "ijk, klm -> ijlm",
                tensor_left,
                tensor_right,
                optimize=[(0, 1)],
            )
            _, singular_values, _, _ = split_two_site_tensor(
                tensor=two_site_tensor, return_truncation_error=True, strategy="svd"
            )
            assert np.isclose(np.linalg.norm(singular_values) - 1, 0)
            for singular_value in singular_values:
                assert singular_value >= cut
            assert truncation_error >= 0
            if strategy == "svd":
                assert np.isclose(mps_compressed.norm() - 1, 0)
            if strategy == "svd_advanced":
                assert np.isclose(inner_product(mps_compressed, mps_compressed) - 1, 0)
                assert inner_product(mps, mps_compressed) <= 1

        # Testing that for no cut the mps stays the same
        for strategy in ["svd", "svd_advanced"]:
            num_sites = 10
            phys_dim = 2
            bond_to_compress = 4
            cut = 1e-12
            psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
            mps = mps_from_dense(
                state_vector=psi, phys_dim=phys_dim, form="Right-canonical"
            )
            mps_compressed, truncation_error = mps.compress_bond(
                bond=bond_to_compress,
                cut=cut,
                renormalize=False,
                strategy=strategy,
                return_truncation_error=True,
            )
            tensor_left = mps_compressed.tensors[bond_to_compress]
            tensor_right = mps_compressed.tensors[bond_to_compress + 1]
            two_site_tensor = contract(
                "ijk, klm -> ijlm",
                tensor_left,
                tensor_right,
                optimize=[(0, 1)],
            )
            _, singular_values, _, _ = split_two_site_tensor(
                tensor=two_site_tensor, return_truncation_error=True, strategy="svd"
            )
            for singular_value in singular_values:
                assert singular_value >= cut
            assert np.isclose(truncation_error, 0)
            if strategy == "svd":
                assert np.isclose(mps_compressed.norm() - 1, 0)
                assert np.isclose(mps_compressed.norm() + truncation_error - 1, 0)
            if strategy == "svd_advanced":
                assert np.isclose(inner_product(mps_compressed, mps_compressed) - 1, 0)
                assert np.isclose(inner_product(mps, mps_compressed) - 1, 0)


def test_canonical_compress():
    """
    Test for the ``compress`` method of the :class:`CanonicalMPS` class.
    """

    for _ in range(10):
        strategy = "svd"
        renormalize = False
        num_sites = 10
        phys_dim = 2
        chi_max = 3
        cut = 1e-12
        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        mps = mps_from_dense(
            state_vector=psi, phys_dim=phys_dim, form="Right-canonical"
        )
        mps_compressed, truncation_errors = mps.compress(
            chi_max=chi_max,
            cut=cut,
            renormalize=renormalize,
            strategy=strategy,
            return_truncation_errors=True,
        )
        assert inner_product(mps, mps_compressed) <= 1
        assert mps_compressed.norm() <= 1
        for bond_dim in mps_compressed.bond_dimensions:
            assert bond_dim <= chi_max
        for truncation_error in truncation_errors:
            assert truncation_error >= 0
        assert np.isclose(mps_compressed.norm() + np.sum(truncation_errors) - 1, 0)


def test_canonical_marginal():
    """
    Test for the ``marginal`` method of the :class:`CanonicalMPS` class.
    """

    # Testing marginalization on product states.
    num_sites = np.random.randint(4, 9)
    phys_dim = np.random.randint(2, 4)
    sites_to_marginalize = [0, 1]

    mps_prod = create_simple_product_state(
        num_sites=num_sites, which="0", phys_dim=phys_dim, form="Right-canonical"
    )
    mps_prod_result = create_simple_product_state(
        num_sites=num_sites - len(sites_to_marginalize),
        which="0",
        phys_dim=phys_dim,
        form="Right-canonical",
    )
    mps_marginalized = mps_prod.marginal(sites_to_marginalize)
    mps_prod_result.tensors[0] /= phys_dim
    for tensor_0, tensor_1 in zip(mps_prod.tensors, mps_prod_result.tensors):
        assert np.isclose(tensor_0, tensor_1).all()

    mps_prod = create_simple_product_state(
        num_sites=num_sites, which="+", phys_dim=phys_dim, form="Right-canonical"
    )
    for tensor in mps_prod.tensors:
        tensor *= np.sqrt(phys_dim)
    mps_prod_result = create_simple_product_state(
        num_sites=num_sites - len(sites_to_marginalize),
        which="+",
        phys_dim=phys_dim,
        form="Right-canonical",
    )
    for tensor in mps_prod_result.tensors:
        tensor *= np.sqrt(phys_dim)
    mps_prod.tensors[0] /= phys_dim
    mps_marginalized = mps_prod.marginal(sites_to_marginalize)
    for tensor_0, tensor_1 in zip(mps_prod.tensors, mps_prod_result.tensors):
        assert np.isclose(tensor_0, tensor_1).all()

    # Testing marginalization on random states.
    for _ in range(10):
        psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
        mps = mps_from_dense(psi, phys_dim=phys_dim, form="Right-canonical")
        mps_copy = mps.copy()

        sites_all = list(range(mps.num_sites))
        sites_to_marginalize = [site for site in sites_all if np.random.uniform() < 0.3]
        sites_left = [site for site in sites_all if site not in sites_to_marginalize]

        mps_marginalized_normalized = mps_copy.marginal(
            sites_to_marginalize, renormalize=True
        )

        with pytest.raises(ValueError):
            mps.marginal([100, 200])

        assert (
            mps_marginalized_normalized.num_sites == len(sites_left)
            if len(sites_left) > 0
            else 1
        )
        if len(sites_left) > 0:
            assert np.isclose(mps_marginalized_normalized.norm(), 1)

    # Testing marginalization on a particular random state.
    num_sites = 8
    phys_dim = 2

    sites_all = list(range(num_sites))
    sites_to_marginalize = [0, 1, 4]
    sites_left = [site for site in sites_all if site not in sites_to_marginalize]

    psi = create_state_vector(num_sites=num_sites, phys_dim=phys_dim)
    mps = mps_from_dense(psi, phys_dim=phys_dim, form="Right-canonical")

    mps_start = mps.copy()
    trace_tensor = np.ones(phys_dim) / np.sqrt(phys_dim)

    mps_marginalized = mps.marginal(sites_to_marginalize, renormalize=False)

    mps_start.tensors[0] = np.tensordot(mps_start.tensors[0], trace_tensor, (1, 0))
    mps_start.tensors[1] = np.tensordot(mps_start.tensors[1], trace_tensor, (1, 0))
    mps_start.tensors[4] = np.tensordot(mps_start.tensors[4], trace_tensor, (1, 0))
    mps_start.tensors[2] = contract(
        "ij, jk, klm -> ilm",
        mps_start.tensors[0],
        mps_start.tensors[1],
        mps_start.tensors[2],
    )
    mps_start.tensors[5] = contract(
        "ij, jkl -> ikl", mps_start.tensors[4], mps_start.tensors[5]
    )
    mps_start = CanonicalMPS(
        tensors=[tensor for tensor in mps_start.tensors if tensor.ndim == 3]
    )

    for tensor_0, tensor_1 in zip(mps_start.tensors, mps_marginalized.tensors):
        assert tensor_0.shape == tensor_1.shape
        assert np.isclose(tensor_0, tensor_1).all()

    # Testing another exact case.
    tensor_0 = np.array([[[1, 0], [0, 2]]])  # (1,2,2)
    tensor_1 = np.array([[[1, 0], [0, 2]], [[3, 0], [0, 4]]])  # (2,2,2)
    tensor_2 = np.array([[[1], [0]], [[0], [2]]])  # shape (2,2,1)
    tensors = [tensor_0, tensor_1, tensor_2]
    mps = CanonicalMPS(tensors=tensors)

    trace_tensor = np.ones(mps.phys_dimensions[0]) / np.sqrt(2)
    expected_tensor_1 = np.tensordot(tensors[1], trace_tensor, axes=([1], [0]))
    expected_tensor_1 = np.einsum("ij, jkl", expected_tensor_1, tensor_2)

    mps = mps.marginal([1], renormalize=False)

    assert mps.tensors[1].shape == expected_tensor_1.shape
    assert np.allclose(mps.tensors[1], expected_tensor_1)


def test_marginal_does_not_produce_nans_when_the_center_underflows():
    """Marginalizing a chain whose amplitudes have underflowed must stay finite.

    Long chains of very small amplitudes (low physical error rates, hundreds of
    sites) can drive the orthogonality center to exactly zero. Renormalizing by
    that norm gives 0/0 and silently fills the state with NaNs, which then reach
    the decoder as unusable shots.
    """
    mps = create_simple_product_state(12, which="0", form="Right-canonical")
    # It has to be the LAST tensor: marginal() sets the orthogonality center to
    # the final site and renormalizes by *that* tensor's norm, so zeroing any
    # earlier one leaves the guard untouched and the test vacuous.
    mps.tensors[-1] = mps.tensors[-1] * 0.0

    marginalized = mps.marginal(sites_to_marginalize=list(range(10)), renormalize=True)

    assert np.all(np.isfinite(marginalized.dense(flatten=True)))


def _two_site_reference_move(mps, final_pos, renormalize):
    """The two-site-SVD move spelled out: the reference for the one-site path.

    move_orth_center no longer has a flag that forces the two-site branch
    (renormalized moves and returned spectra take the one-site path too), so
    an old-vs-new comparison has to build the reference explicitly.
    """
    if mps.orth_center == final_pos:
        return mps.copy()
    leftwards = mps.orth_center > final_pos
    work = mps.reverse() if leftwards else mps.copy()
    begin = work.orth_center
    final = (mps.num_sites - 1 - final_pos) if leftwards else final_pos
    for i in range(begin, final):
        u_l, s_bond, v_r, _ = split_two_site_tensor(
            work.two_site_tensor_next(i),
            chi_max=mps.chi_max,
            renormalize=renormalize,
            strategy="svd",
            return_truncation_error=True,
        )
        work.tensors[i] = u_l
        work.tensors[i + 1] = v_r * s_bond[:, None, None]
        work.orth_center = i + 1
    return work.reverse() if leftwards else work


def test_move_orth_center_carries_a_collapsed_bond_through():
    """A bond of dimension 0 (a truncation that emptied the spectrum) must
    move through the center-only path without raising.

    Three example notebooks hit this in CI: a zero-width center has nothing
    to factor, so such a bond must take the two-site SVD branch, which
    carries the degenerate shape through unchanged.
    """
    from mdopt.mps.canonical import CanonicalMPS

    tensors = [
        np.zeros((1, 2, 0)),
        np.zeros((0, 2, 1)),
        np.zeros((1, 2, 1)),
    ]
    mps = CanonicalMPS(tensors, orth_center=0, chi_max=4)
    moved = mps.move_orth_center(2, renormalize=False)
    assert [t.shape for t in moved.tensors][0][2] == 0
    back = moved.move_orth_center(0, renormalize=False)
    assert len(back) == 3


def test_move_orth_center_collapses_a_sub_cut_spectrum_like_the_svd_path():
    """A center whose whole spectrum sits below the 1e-12 cut must collapse
    to a zero-width bond on the center-only path exactly as on the two-site
    SVD path, rather than propagating a sub-cut direction."""
    from mdopt.mps.canonical import CanonicalMPS

    def tiny_center_mps():
        tensors = [
            np.array([[[1.0, 0.0], [0.0, 1.0]]]).reshape(1, 2, 2) * 5e-13,
            np.eye(2).reshape(2, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ]
        return CanonicalMPS(tensors, orth_center=0, chi_max=4)

    via_one_site = tiny_center_mps().move_orth_center(2, renormalize=False)
    via_two_site = _two_site_reference_move(tiny_center_mps(), 2, renormalize=False)
    assert via_one_site.tensors[0].shape[2] == 0
    assert via_one_site.tensors[0].shape[2] == via_two_site.tensors[0].shape[2]


def test_move_orth_center_qr_path_matches_svd_path_on_full_rank_states():
    """The one-site move must reproduce the two-site SVD move exactly.

    Real and complex random states, moves in both directions, with a
    chi_max below the full Schmidt rank so the finite truncation is
    exercised too: dense() and every bond dimension must agree between the
    one-site path and the two-site reference spelled out above.
    """
    from mdopt.mps.utils import mps_from_dense

    for seed, complex_case in ((11, False), (12, True), (13, False), (14, True)):
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(2**8)
        if complex_case:
            vec = vec + 1j * rng.standard_normal(2**8)
        vec = vec / np.linalg.norm(vec)
        for chi_max in (int(1e4), 3):
            base = mps_from_dense(vec, form="Right-canonical")
            base.chi_max = chi_max
            for targets in ((7, 0), (0, 7), (4, 1, 6)):
                fast, slow = base.copy(), base.copy()
                for target in targets:
                    fast = fast.move_orth_center(target, renormalize=False)
                    slow = _two_site_reference_move(slow, target, renormalize=False)
                assert list(fast.bond_dimensions) == list(slow.bond_dimensions), (
                    seed,
                    chi_max,
                    targets,
                )
                assert np.allclose(
                    fast.dense(flatten=True),
                    slow.dense(flatten=True),
                    rtol=0.0,
                    atol=1e-11,
                ), (seed, chi_max, targets)


def test_move_orth_center_matches_svd_path_on_non_canonical_chains():
    """Behind a biased pair the neighbors are not isometries, so a move
    that must truncate at chi_max has to take the two-site SVD; the fast
    path may only handle moves that truncate nothing."""
    from mdopt.mps.utils import mps_from_dense
    from mdopt.decoding.decoding import apply_depolarizing_bias

    rng = np.random.default_rng(5)
    vec = rng.standard_normal(2**8)
    vec = vec / np.linalg.norm(vec)
    for chi_max in (2, 4, int(1e4)):
        base = mps_from_dense(vec, form="Right-canonical", chi_max=chi_max)
        biased = apply_depolarizing_bias(
            base, sites_to_bias=[0, 2, 4, 6], prob_bias_list=0.3
        )
        fast = biased.copy().move_orth_center(0, renormalize=False)
        slow = _two_site_reference_move(biased.copy(), 0, renormalize=False)
        assert list(fast.bond_dimensions) == list(slow.bond_dimensions), chi_max
        assert np.allclose(
            fast.dense(flatten=True), slow.dense(flatten=True), rtol=0.0, atol=1e-10
        ), chi_max


def test_move_orth_center_prunes_on_discarded_amplitude_not_on_the_center_spectrum():
    """Review counterexamples: a sub-cut center direction amplified by a large
    neighbor entry must be KEPT (its amplitude is macroscopic), and a
    direction a null neighbor row makes worthless may go. Both cases must
    agree with the two-site SVD path on the represented state."""
    from mdopt.mps.canonical import CanonicalMPS

    def chain(neighbor):
        center = np.zeros((1, 2, 2))
        center[0, 0, 0], center[0, 1, 1] = 1.0, 5e-13
        last = np.array([1.0, 0.0]).reshape(1, 2, 1)
        return CanonicalMPS([center, neighbor, last], orth_center=0, chi_max=4)

    amplified = np.zeros((2, 2, 1))
    amplified[0, 0, 0], amplified[1, 1, 0] = 1.0, 1e13
    null_row = np.zeros((2, 2, 1))
    null_row[0, 0, 0] = 1.0
    for neighbor in (amplified, null_row):
        fast = chain(neighbor).move_orth_center(2, renormalize=False)
        slow = _two_site_reference_move(chain(neighbor), 2, renormalize=False)
        assert np.allclose(
            fast.dense(flatten=True), slow.dense(flatten=True), rtol=0.0, atol=1e-10
        )
    kept = chain(amplified).move_orth_center(1, renormalize=False)
    assert kept.tensors[0].shape[2] == 2, "the amplified direction carries amplitude 5"


def test_one_site_move_matches_two_site_move_with_renormalization():
    """The center-only move also covers renormalized moves and returned spectra.

    With an isometric neighbor B, theta = C B has theta theta^dag = C C^dag,
    so the spectrum, its renormalization, the cut and the chi_max count are
    those of the center alone. Both DMRG sweeps (renormalize=True) and the
    explicit-form conversion (return_singular_values=True) go through it now;
    the reference below is the two-site formula spelled out.
    """
    from mdopt.utils.utils import split_two_site_tensor

    rng = np.random.default_rng(11)
    for dtype in (float, complex):
        vector = rng.standard_normal(2**7)
        if dtype is complex:
            vector = vector + 1j * rng.standard_normal(2**7)
        vector /= np.linalg.norm(vector)
        mps = mps_from_dense(vector, form="Right-canonical", chi_max=5)
        assert isinstance(mps, CanonicalMPS)
        mps.chi_max = 5

        reference = mps.copy()
        spectra = []
        for site in range(0, 4):
            theta = reference.two_site_tensor_next(site)
            u_l, s_bond, v_r, _ = split_two_site_tensor(
                theta, chi_max=5, renormalize=True, return_truncation_error=True
            )
            spectra.append(s_bond)
            reference.tensors[site] = u_l
            reference.tensors[site + 1] = v_r * s_bond[:, None, None]
            reference.orth_center = site + 1

        moved, singular_values = mps.move_orth_center(
            4, return_singular_values=True, renormalize=True
        )
        assert moved.orth_center == 4
        assert len(singular_values) == len(spectra)
        for got, want in zip(singular_values, spectra):
            assert np.allclose(got, want, rtol=0.0, atol=1e-12)
        assert np.allclose(moved.dense(), reference.dense(), rtol=0.0, atol=1e-12)
        assert moved.bond_dimensions == reference.bond_dimensions


def test_move_orth_center_accepts_integer_and_boolean_tensors():
    """Exact-arithmetic product states are valid input; the isometry gate must
    promote its Gram matrix rather than fail on an in-place float subtraction."""
    from mdopt.mps.canonical import CanonicalMPS

    for dtype in (int, bool):
        tensors = [np.array([1, 0], dtype=dtype).reshape(1, 2, 1) for _ in range(4)]
        mps = CanonicalMPS(tensors, orth_center=0, chi_max=4)
        moved = mps.move_orth_center(3, renormalize=False)
        assert moved.orth_center == 3
        assert np.allclose(moved.dense(flatten=True)[0], 1.0)
        back = moved.move_orth_center(0, renormalize=True)
        assert back.orth_center == 0
        assert list(back.bond_dimensions) == [1, 1, 1]
