"""Tests for the ``mdopt.utils.utils`` module."""

import pytest
import scipy
import numpy as np
from opt_einsum import contract

from mdopt.utils.utils import (
    create_random_mpo,
    kron_tensors,
    mpo_from_matrix,
    mpo_to_matrix,
    split_two_site_tensor,
    svd,
    qr,
)


def test_utils_svd():
    """Test for the ``svd`` function."""

    for _ in range(10):
        dim = np.random.randint(low=10, high=100, size=2)
        m = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)

        u, s, v_h, _ = svd(m)

        m_trimmed = contract("ij, j, jk -> ik", u, s, v_h, optimize=[(0, 1), (0, 1)])

        u, s, v_h, _ = svd(m_trimmed)

        m_trimmed_new = contract(
            "ij, j, jk -> ik", u, s, v_h, optimize=[(0, 1), (0, 1)]
        )

        with pytest.raises(ValueError):
            dim = np.random.randint(low=10, high=100, size=4)
            m = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)
            svd(m)

        assert np.isclose(np.linalg.norm(m_trimmed - m_trimmed_new), 0)

    for _ in range(10):
        dim = np.random.randint(low=50, high=100, size=2)
        m = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)
        num_sing_values = np.random.randint(1, 10)

        _, s, _, _ = svd(m, chi_max=num_sing_values, renormalise=True)
        assert len(s) == num_sing_values

        _, s, _, _ = svd(m, cut=1, renormalise=False)
        for value in s:
            assert np.abs(value) > 1

    # Check return_truncation_error flag
    dim = (30, 40)
    m = np.random.rand(*dim)
    u, s, v_h, trunc_err = svd(m, return_truncation_error=True)
    assert isinstance(trunc_err, float)
    assert trunc_err >= 0

    # Check renormalisation with return_truncation_error
    _, s, _, trunc_err = svd(m, renormalise=True, return_truncation_error=True)
    assert np.isclose(np.linalg.norm(s), 1)

    # Check that truncation works when cut > all singular values
    small_mat = np.random.rand(5, 5) * 1e-10
    u, s, v_h, err = svd(small_mat, cut=1.0, return_truncation_error=True)
    assert len(s) == 0
    assert err > 0  # should be equal to norm of original singular spectrum

    # Force fallback SVD method by passing a pathological matrix
    # This matrix will make `np.linalg.svd` fail due to NaNs
    bad_matrix = np.full((10, 10), np.nan)
    with pytest.raises(RuntimeError):
        svd(bad_matrix)

    # Force another fallback by using a near-singular matrix
    near_singular = np.eye(20) * 1e-15
    u, s, v_h, _ = svd(near_singular)
    assert all(np.abs(s) > 0)


def test_utils_qr():
    """Test for the `qr` function."""

    for _ in range(10):
        dim = np.random.randint(low=10, high=100, size=2)
        mat = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)
        q, r, _ = qr(mat)
        mat_reconstructed = np.dot(q, r)
        assert np.allclose(mat, mat_reconstructed, atol=1e-10)

        dim = np.random.randint(low=10, high=100, size=2)
        mat = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)
        chi_max = np.random.randint(5, 30)
        q, r, _ = qr(mat, chi_max=chi_max)
        assert q.shape[1] <= chi_max
        assert r.shape[0] <= chi_max

        dim = np.random.randint(low=10, high=100, size=2)
        mat = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)
        q, r, truncation_error = qr(mat, return_truncation_error=True)
        assert np.isclose(truncation_error, 0)

        dim = np.random.randint(low=10, high=100, size=2)
        mat = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)
        mat /= np.linalg.norm(mat)
        cut = 1e-1
        q, r, _ = qr(mat, cut=cut)
        _, _, pivots = scipy.linalg.qr(
            mat, pivoting=True, mode="economic", check_finite=False
        )
        permutation_matrix = np.eye(mat.shape[1])[:, pivots]
        r = r @ permutation_matrix
        for element in np.absolute(np.diag(r)):
            assert element > cut

        dim = np.random.randint(low=10, high=100, size=2)
        mat = np.random.uniform(size=dim) + 1j * np.random.uniform(size=dim)
        q, r, _ = qr(mat, renormalise=True)
        assert np.isclose(np.linalg.norm(np.diag(r)), 1, atol=1e-10)

        with pytest.raises(ValueError):
            dim_invalid = np.random.randint(low=1, high=4, size=1)
            mat_invalid = np.random.uniform(size=dim_invalid)
            qr(mat_invalid)


def test_utils_split_two_site_tensor():
    """Test for the ``split_two_site_tensor`` function."""

    for _ in range(100):
        phys_dim = 2
        bond_dim = np.random.randint(2, 18, size=2)
        tensor = np.random.uniform(
            size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1])
        ) + 1j * np.random.uniform(size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1]))

        u_l, singular_values, v_r, truncation_error = split_two_site_tensor(
            tensor=tensor,
            chi_max=1e4,
            cut=1e-12,
            strategy="svd",
            renormalise=False,
            return_truncation_error=True,
        )
        should_be_t = contract(
            "ijk, kl, lmn -> ijmn",
            u_l,
            np.diag(singular_values),
            v_r,
            optimize=[(0, 1), (0, 1)],
        )
        assert np.isclose(truncation_error, 0)
        assert tensor.shape == should_be_t.shape
        assert np.isclose(np.linalg.norm(tensor - should_be_t), 0)

        bond_dim = np.random.randint(2, 18, size=2)
        tensor = np.random.uniform(
            size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1])
        ) + 1j * np.random.uniform(size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1]))
        chi_max = 2
        u_l, singular_values, v_r, truncation_error = split_two_site_tensor(
            tensor=tensor,
            chi_max=chi_max,
            cut=1e-12,
            strategy="svd",
            renormalise=False,
            return_truncation_error=True,
        )
        assert truncation_error > 0
        assert len(singular_values) == chi_max

        bond_dim = np.random.randint(2, 18, size=2)
        tensor = np.random.uniform(
            size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1])
        ) + 1j * np.random.uniform(size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1]))
        tensor /= np.linalg.norm(tensor)
        cut = 0.5
        u_l, singular_values, v_r, truncation_error = split_two_site_tensor(
            tensor=tensor,
            chi_max=1e4,
            cut=cut,
            strategy="svd",
            renormalise=False,
            return_truncation_error=True,
        )
        assert truncation_error >= 0
        for singular_value in singular_values:
            assert singular_value > cut

        bond_dim = np.random.randint(2, 18, size=2)
        tensor = np.random.uniform(
            size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1])
        ) + 1j * np.random.uniform(size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1]))
        tensor /= np.linalg.norm(tensor)
        cut = 0.5
        u_l, singular_values, v_r, truncation_error = split_two_site_tensor(
            tensor=tensor,
            chi_max=1e4,
            cut=cut,
            strategy="svd",
            renormalise=True,
            return_truncation_error=True,
        )
        assert np.isclose(np.linalg.norm(singular_values), 1)
        assert truncation_error >= 0

        bond_dim = np.random.randint(2, 18, size=2)
        tensor = np.random.uniform(
            size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1])
        ) + 1j * np.random.uniform(size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1]))
        q_l, r_r, truncation_error, _ = split_two_site_tensor(
            tensor=tensor,
            chi_max=1e4,
            cut=1e-12,
            strategy="qr",
            renormalise=False,
            return_truncation_error=True,
        )
        should_be_t = contract(
            "ijk, klm -> ijlm",
            q_l,
            r_r,
            optimize=[(0, 1)],
        )
        assert np.isclose(truncation_error, 0)
        assert tensor.shape == should_be_t.shape
        assert np.isclose(np.linalg.norm(tensor - should_be_t), 0)

        bond_dim = np.random.randint(2, 18, size=2)
        tensor = np.random.uniform(
            size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1])
        ) + 1j * np.random.uniform(size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1]))
        chi_max = 3
        q_l, r_r, truncation_error, _ = split_two_site_tensor(
            tensor=tensor,
            chi_max=chi_max,
            cut=1e-12,
            strategy="qr",
            renormalise=False,
            return_truncation_error=True,
        )
        should_be_t = contract(
            "ijk, klm -> ijlm",
            q_l,
            r_r,
            optimize=[(0, 1)],
        )
        assert truncation_error > 0
        assert q_l.shape[2] == chi_max
        assert r_r.shape[0] == chi_max

        bond_dim = np.random.randint(2, 18, size=2)
        tensor = np.random.uniform(
            size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1])
        ) + 1j * np.random.uniform(size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1]))
        q_l, r_r, truncation_error, _ = split_two_site_tensor(
            tensor=tensor,
            chi_max=1e4,
            cut=1e-12,
            strategy="qr",
            renormalise=True,
            return_truncation_error=True,
        )
        should_be_t = contract(
            "ijk, klm -> ijlm",
            q_l,
            r_r,
            optimize=[(0, 1)],
        )
        assert truncation_error > 0
        r_r = r_r.reshape((r_r.shape[0], r_r.shape[1] * r_r.shape[2]))
        assert np.isclose(np.linalg.norm(np.diag(r_r)), 1)

        with pytest.raises(ValueError):
            tensor = np.random.uniform(
                size=(bond_dim[0], phys_dim, phys_dim, phys_dim, bond_dim[1])
            ) + 1j * np.random.uniform(
                size=(bond_dim[0], phys_dim, phys_dim, phys_dim, bond_dim[1])
            )
            split_two_site_tensor(tensor)

        with pytest.raises(ValueError):
            tensor = np.random.uniform(
                size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1])
            ) + 1j * np.random.uniform(
                size=(bond_dim[0], phys_dim, phys_dim, bond_dim[1])
            )
            split_two_site_tensor(tensor, strategy="bla")


def test_utils_kron_tensors():
    """Test for the ``kron_tensors`` function."""

    for _ in range(10):
        dims_1 = np.random.randint(2, 11, size=3)
        dims_2 = np.random.randint(2, 11, size=3)

        tensor_1 = np.random.uniform(
            size=(dims_1[0], dims_1[1], dims_1[2])
        ) + 1j * np.random.uniform(size=(dims_1[0], dims_1[1], dims_1[2]))
        tensor_2 = np.random.uniform(
            size=(dims_2[0], dims_2[1], dims_2[2])
        ) + 1j * np.random.uniform(size=(dims_2[0], dims_2[1], dims_2[2]))

        product_1 = kron_tensors(
            tensor_1, tensor_2, conjugate_second=True, merge_physicals=True
        )
        product_2 = kron_tensors(
            tensor_1, tensor_2, conjugate_second=True, merge_physicals=False
        )
        product_3 = kron_tensors(
            tensor_1, tensor_2, conjugate_second=False, merge_physicals=True
        )
        product_4 = kron_tensors(
            tensor_1, tensor_2, conjugate_second=False, merge_physicals=False
        )

        product_5 = np.tensordot(tensor_1, np.conjugate(tensor_2), 0)
        product_5 = product_5.transpose(0, 3, 1, 4, 2, 5)
        product_5 = product_5.reshape(
            (dims_1[0] * dims_2[0], dims_1[1] * dims_2[1], dims_1[2] * dims_2[2])
        )

        product_6 = np.tensordot(tensor_1, np.conjugate(tensor_2), 0)
        product_6 = product_6.transpose(0, 3, 1, 4, 2, 5)
        product_6 = product_6.reshape(
            (
                product_6.shape[0] * product_6.shape[1],
                product_6.shape[2],
                product_6.shape[3],
                product_6.shape[4] * product_6.shape[5],
            )
        )

        product_7 = np.tensordot(tensor_1, tensor_2, 0)
        product_7 = product_7.transpose(0, 3, 1, 4, 2, 5)
        product_7 = product_7.reshape(
            (dims_1[0] * dims_2[0], dims_1[1] * dims_2[1], dims_1[2] * dims_2[2])
        )

        product_8 = np.tensordot(tensor_1, tensor_2, 0)
        product_8 = product_8.transpose(0, 3, 1, 4, 2, 5)
        product_8 = product_8.reshape(
            (
                product_8.shape[0] * product_8.shape[1],
                product_8.shape[2],
                product_8.shape[3],
                product_8.shape[4] * product_8.shape[5],
            )
        )

        with pytest.raises(ValueError):
            dims_1 = np.random.randint(2, 11, size=4)
            dims_2 = np.random.randint(2, 11, size=3)
            tensor_1 = np.random.uniform(
                size=(dims_1[0], dims_1[1], dims_1[2], dims_1[3])
            ) + 1j * np.random.uniform(size=(dims_1[0], dims_1[1], dims_1[2]))
            tensor_2 = np.random.uniform(
                size=(dims_2[0], dims_2[1], dims_2[2])
            ) + 1j * np.random.uniform(size=(dims_2[0], dims_2[1], dims_2[2]))
            kron_tensors(tensor_1, tensor_2)

            dims_1 = np.random.randint(2, 11, size=3)
            dims_2 = np.random.randint(2, 11, size=4)
            tensor_1 = np.random.uniform(
                size=(dims_1[0], dims_1[1], dims_1[2])
            ) + 1j * np.random.uniform(size=(dims_1[0], dims_1[1], dims_1[2]))
            tensor_2 = np.random.uniform(
                size=(dims_2[0], dims_2[1], dims_2[2], dims_2[3])
            ) + 1j * np.random.uniform(size=(dims_2[0], dims_2[1], dims_2[2]))
            kron_tensors(tensor_1, tensor_2)

        assert np.isclose(np.linalg.norm(product_1 - product_5), 0)
        assert np.isclose(np.linalg.norm(product_2 - product_6), 0)
        assert np.isclose(np.linalg.norm(product_3 - product_7), 0)
        assert np.isclose(np.linalg.norm(product_4 - product_8), 0)


def test_utils_mpo_from_matrix():
    """Test for the ``mpo_from_matrix`` function."""

    for _ in range(10):
        num_sites = np.random.randint(4, 6)
        phys_dim = np.random.randint(2, 4)
        matrix_shape = (phys_dim**num_sites, phys_dim**num_sites)
        matrix = np.random.uniform(size=matrix_shape) + 1j * np.random.uniform(
            size=matrix_shape
        )

        mpo = mpo_from_matrix(
            matrix,
            interlaced=True,
            orthogonalise=False,
            num_sites=num_sites,
            phys_dim=phys_dim,
        )
        mpo_iso = mpo_from_matrix(
            matrix,
            interlaced=True,
            orthogonalise=True,
            num_sites=num_sites,
            phys_dim=phys_dim,
        )

        matrix_from_mpo = mpo_to_matrix(mpo, interlace=True, group=True)
        matrix_from_mpo_iso = mpo_to_matrix(mpo_iso, interlace=True, group=True)

        assert np.isclose(abs(np.linalg.norm(matrix - matrix_from_mpo)), 0)
        assert np.isclose(abs(np.linalg.norm(matrix - matrix_from_mpo_iso)), 0)

        for site in range(num_sites):
            if site == 0:
                mpo_iso[site] /= np.linalg.norm(mpo_iso[site])
            to_be_identity = contract(
                "ijkl, mjkl -> im", mpo_iso[site], np.conjugate(mpo_iso[site])
            )
            identity = np.identity(to_be_identity.shape[0])
            assert np.isclose(np.linalg.norm(identity - to_be_identity), 0)
            assert mpo[site].shape == mpo_iso[site].shape

        with pytest.raises(ValueError):
            matrix_shape = (
                phys_dim**num_sites,
                phys_dim**num_sites,
                phys_dim**num_sites,
            )
            matrix = np.random.uniform(size=matrix_shape) + 1j * np.random.uniform(
                size=matrix_shape
            )
            mpo = mpo_from_matrix(
                matrix, interlaced=True, num_sites=num_sites, phys_dim=phys_dim
            )


def test_utils_mpo_to_matrix():
    """Test for the ``mpo_to_matrix`` function."""

    for _ in range(10):
        num_sites = np.random.randint(4, 6)
        phys_dim = np.random.randint(2, 4)

        dims_unique = np.random.randint(3, 10, size=num_sites - 1)
        mpo = create_random_mpo(
            num_sites,
            dims_unique,
            phys_dim=phys_dim,
            which=np.random.choice(["uniform", "normal", "randint"], size=1),
        )

        matrix_0 = mpo_to_matrix(mpo, interlace=False, group=False)
        matrix_1 = mpo_to_matrix(mpo, interlace=False, group=True)
        matrix_2 = mpo_to_matrix(mpo, interlace=True, group=False)
        matrix_3 = mpo_to_matrix(mpo, interlace=True, group=True)

        mpo_0 = mpo_from_matrix(
            matrix_0, interlaced=False, num_sites=num_sites, phys_dim=phys_dim
        )
        mpo_1 = mpo_from_matrix(
            matrix_1, interlaced=False, num_sites=num_sites, phys_dim=phys_dim
        )
        mpo_2 = mpo_from_matrix(
            matrix_2, interlaced=True, num_sites=num_sites, phys_dim=phys_dim
        )
        mpo_3 = mpo_from_matrix(
            matrix_3, interlaced=True, num_sites=num_sites, phys_dim=phys_dim
        )

        with pytest.raises(ValueError):
            mpo = create_random_mpo(
                num_sites,
                dims_unique,
                phys_dim=phys_dim,
                which=np.random.choice(["uniform", "normal", "randint"], size=1),
            )
            mpo[1] = np.ones(
                shape=(
                    2,
                    2,
                    2,
                    2,
                    2,
                )
            )
            mpo_to_matrix(mpo)

        matrix_00 = mpo_to_matrix(mpo_0, interlace=False, group=False)
        matrix_01 = mpo_to_matrix(mpo_1, interlace=False, group=True)
        matrix_02 = mpo_to_matrix(mpo_2, interlace=True, group=False)
        matrix_03 = mpo_to_matrix(mpo_3, interlace=True, group=True)

        assert np.isclose(abs(np.linalg.norm(matrix_0 - matrix_00)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_1 - matrix_01)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_2 - matrix_02)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_3 - matrix_03)), 0, atol=1e-6)

    for _ in range(10):
        num_sites = 4
        mpo = create_random_mpo(
            num_sites,
            np.random.randint(2, 6, size=num_sites - 1),
            phys_dim=2,
            which=np.random.choice(["uniform", "normal", "randint"], size=1),
        )

        matrix_0 = mpo_to_matrix(mpo, interlace=True, group=True)
        matrix_1 = mpo_to_matrix(mpo, interlace=True, group=False)
        matrix_2 = mpo_to_matrix(mpo, interlace=False, group=True)
        matrix_3 = mpo_to_matrix(mpo, interlace=False, group=False)

        matrix_00 = contract(
            "abij, bckl, cdmn, deop -> ijklmnop",
            mpo[0],
            mpo[1],
            mpo[2],
            mpo[3],
            optimize=[(0, 1), (0, 1), (0, 1)],
        ).reshape((2**num_sites, 2**num_sites))
        matrix_01 = contract(
            "abij, bckl, cdmn, deop -> ijklmnop",
            mpo[0],
            mpo[1],
            mpo[2],
            mpo[3],
            optimize=[(0, 1), (0, 1), (0, 1)],
        )
        matrix_02 = contract(
            "abij, bckl, cdmn, deop -> jlnpikmo",
            mpo[0],
            mpo[1],
            mpo[2],
            mpo[3],
            optimize=[(0, 1), (0, 1), (0, 1)],
        ).reshape((16, 16))
        matrix_03 = contract(
            "abij, bckl, cdmn, deop -> jlnpikmo",
            mpo[0],
            mpo[1],
            mpo[2],
            mpo[3],
            optimize=[(0, 1), (0, 1), (0, 1)],
        )

        assert np.isclose(abs(np.linalg.norm(matrix_0 - matrix_00)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_1 - matrix_01)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_2 - matrix_02)), 0, atol=1e-6)
        assert np.isclose(abs(np.linalg.norm(matrix_3 - matrix_03)), 0, atol=1e-6)


def test_create_random_mpo_rng_is_honoured():
    """A passed generator must control every distribution and be reproducible."""
    for which in ("uniform", "normal", "randint"):
        first = create_random_mpo(
            3, [2, 2], 2, which=which, rng=np.random.default_rng(5)
        )
        second = create_random_mpo(
            3, [2, 2], 2, which=which, rng=np.random.default_rng(5)
        )
        assert len(first) == 3, "empty result would satisfy the comparison below"
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))

        other = create_random_mpo(
            3, [2, 2], 2, which=which, rng=np.random.default_rng(6)
        )
        assert not all(np.array_equal(a, b) for a, b in zip(first, other, strict=True))

        # An explicit generator must ignore the global stream entirely.
        np.random.seed(0)
        again = create_random_mpo(
            3, [2, 2], 2, which=which, rng=np.random.default_rng(5)
        )
        assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))


def test_create_random_mpo_global_seed_is_reproducible():
    """Without a generator the legacy global stream is used, so seeding works."""
    np.random.seed(77)
    first = create_random_mpo(3, [2, 2], 2)
    np.random.seed(77)
    second = create_random_mpo(3, [2, 2], 2)
    assert len(first) == 3, "empty result would satisfy the comparison below"
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_svd_accepts_infinite_chi_max(rng):
    """``chi_max=np.inf`` must mean "no truncation", not raise.

    Four notebooks use ``np.inf`` as the first entry of their ``bond_dims``
    sweep to get the untruncated reference curve. Coercing it with
    ``int(chi_max)`` before taking the min raises OverflowError, which used to
    abort those notebooks partway through.
    """
    mat = rng.normal(size=(12, 9)) + 1j * rng.normal(size=(12, 9))

    u_l, singular_values, v_h, _ = svd(mat, cut=1e-16, chi_max=np.inf)

    # Nothing is dropped: the reconstruction is exact.
    reconstructed = u_l @ np.diag(singular_values) @ v_h
    assert np.allclose(reconstructed, mat)
    assert len(singular_values) == min(mat.shape)

    # And a finite chi_max still truncates.
    _, truncated, _, _ = svd(mat, cut=1e-16, chi_max=4)
    assert len(truncated) == 4


def test_split_two_site_tensor_accepts_infinite_chi_max(rng):
    """The same idiom has to survive the split_two_site_tensor wrapper."""
    tensor = rng.normal(size=(2, 3, 3, 2))

    _, _, _ = split_two_site_tensor(tensor, chi_max=np.inf, cut=1e-16)[:3]


def test_qr_accepts_infinite_chi_max(rng):
    """The qr branch takes the same np.inf idiom as svd.

    Covered separately because ``split_two_site_tensor`` defaults to
    ``strategy="svd"``, so the svd tests never reach this truncation site.
    """
    mat = rng.normal(size=(10, 6))

    q_mat, r_mat, _ = qr(mat, cut=1e-16, chi_max=np.inf)

    assert np.allclose(q_mat @ r_mat, mat)
    assert q_mat.shape[1] == min(mat.shape)

    # A finite chi_max still truncates.
    q_small, r_small, _ = qr(mat, cut=1e-16, chi_max=3)
    assert q_small.shape[1] == 3
    assert r_small.shape[0] == 3


def test_svd_rectangular_reduction_matches_direct_svd():
    """Deterministic coverage of the QR/LQ-reduced SVD branches.

    The reduction triggers on aspect ratio >= 2 in either orientation, so
    each fixed matrix below pins one branch (the 31-column one pins the
    boundary's open side, staying on the direct path): singular values,
    reconstruction, orthogonality, and chi_max truncation must all agree
    with a direct numpy SVD.
    """
    rng = np.random.default_rng(20240903)
    shapes_and_paths = [
        ((16, 32), "wide, reduced"),
        ((16, 31), "wide, direct boundary"),
        ((32, 16), "tall, reduced"),
        ((31, 16), "tall, direct boundary"),
        ((16, 16), "square, direct"),
    ]
    for complex_case in (False, True):
        for shape, label in shapes_and_paths:
            mat = rng.standard_normal(shape)
            if complex_case:
                mat = mat + 1j * rng.standard_normal(shape)
            u_l, s, v_h, _ = svd(mat, cut=0.0, chi_max=int(1e4))
            u_ref, s_ref, v_ref = np.linalg.svd(mat, full_matrices=False)
            assert np.allclose(s, s_ref, rtol=0.0, atol=1e-11), (label, complex_case)
            assert np.allclose((u_l * s) @ v_h, mat, rtol=0.0, atol=1e-11), (
                label,
                complex_case,
            )
            eye = np.eye(u_l.shape[1])
            assert np.allclose(u_l.conj().T @ u_l, eye, rtol=0.0, atol=1e-12), label
            assert np.allclose(v_h @ v_h.conj().T, eye, rtol=0.0, atol=1e-12), label

            chi = 5
            u_t, s_t, v_t, err = svd(
                mat, cut=0.0, chi_max=chi, return_truncation_error=True
            )
            assert s_t.shape == (chi,)
            assert np.allclose(s_t, s_ref[:chi], rtol=0.0, atol=1e-11), label
            assert np.isclose(
                err, float(np.linalg.norm(s_ref[chi:]) ** 2), atol=1e-12
            ), label


def test_svd_nonfinite_input_takes_the_fallback_chain():
    """qr on a non-finite matrix returns garbage silently, so the reduced
    path must not see one; the direct call raises into the fallbacks, whose
    jitter attempt cannot rescue a NaN either -- the whole call must raise."""
    mat = np.full((8, 32), np.nan)
    with pytest.raises(RuntimeError, match="All SVD methods failed"):
        svd(mat)
