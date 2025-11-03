"""This module contains miscellaneous utilities."""

from typing import Tuple, Optional, List
from itertools import chain
import numpy as np
import scipy
from opt_einsum import contract

# --- Backend shim: prefer your GPU/array backend if available, else NumPy ---
try:
    # expected to export a NumPy-like API (e.g., NumPy or CuPy)
    from mdopt.backend import array as xp  # type: ignore
except Exception:
    import numpy as xp  # type: ignore


def _to_numpy(a):
    """Convert backend arrays (e.g., CuPy) to NumPy without copying if possible."""
    try:
        import cupy as cp  # type: ignore

        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def svd(
    mat: np.ndarray,
    cut: float = float(1e-12),
    chi_max: int = int(1e4),
    renormalise: bool = False,
    return_truncation_error: bool = False,
) -> Tuple[np.ndarray, List[float], np.ndarray, Optional[float]]:
    """
    Performs Singular Value Decomposition with different features.

    Parameters
    ----------
    mat : np.ndarray
        Matrix provided as a ``np.ndarray`` with 2 dimensions.
    cut : float
        Singular values smaller than this will be discarded.
    chi_max : int
        Maximum number of singular values to keep.
    renormalise : bool
        Whether to renormalise the singular value spectrum after the cut.
    return_truncation_error : bool
        Whether to return the truncation error.

    Returns
    -------
    u_l : np.ndarray
        Unitary matrix having left singular vectors as columns.
    singular_values : list
        The singular values, sorted in non-increasing order.
    v_r : np.ndarray
        Unitary matrix having right singular vectors as rows.
    truncation_error : Optional[float]
        The truncation error. Only returned if `return_truncation_error` is True.

    Raises
    ------
    ValueError
        If the input matrix does not have 2 dimensions.
    """
    if mat.ndim != 2:
        raise ValueError(
            f"A valid matrix must have 2 dimensions while the one given has {mat.ndim}."
        )

    # Try backend SVD first (GPU-friendly), then fall back to SciPy variants.
    last_exception: Optional[Exception] = None
    u_l = s = v_h = None  # type: ignore
    a = xp.asarray(mat)
    for attempt in ("xp", "gesdd", "gesvd", "jitter"):
        try:
            if attempt == "xp":
                u_l, s, v_h = xp.linalg.svd(a, full_matrices=False)  # returns U, S, Vh
            elif attempt == "gesdd":
                u_l, s, v_h = scipy.linalg.svd(
                    _to_numpy(a),
                    full_matrices=False,
                    compute_uv=True,
                    check_finite=True,
                    lapack_driver="gesdd",
                )
            elif attempt == "gesvd":
                u_l, s, v_h = scipy.linalg.svd(
                    _to_numpy(a),
                    full_matrices=False,
                    compute_uv=True,
                    check_finite=True,
                    lapack_driver="gesvd",
                )
            else:
                an = _to_numpy(a)
                u_l, s, v_h = scipy.linalg.svd(
                    an + np.eye(an.shape[0], an.shape[1]) * 1e-12,
                    full_matrices=False,
                    compute_uv=True,
                    check_finite=True,
                    lapack_driver="gesvd",
                )
            break
        except Exception as e:
            last_exception = e
            continue
    else:
        raise RuntimeError(f"All SVD methods failed. Last error: {last_exception}")

    # Convert to NumPy for downstream consistency with the current codebase
    u_l = _to_numpy(u_l)
    s = _to_numpy(s).astype(float, copy=False)  # singular values are real non-negative
    v_h = _to_numpy(v_h)

    # Truncate by cut and chi_max
    max_num = min(int(chi_max), int(np.sum(s > cut)))
    residual = s[max_num:]
    truncation_error = float(np.linalg.norm(residual) ** 2)
    u_l = u_l[:, :max_num]
    s = s[:max_num]
    v_h = v_h[:max_num, :]

    if renormalise and s.size > 0:
        norm = float(np.linalg.norm(s))
        if norm > 0:
            s = s / norm

    if return_truncation_error:
        return (
            np.asarray(u_l),
            np.asarray(s),
            np.asarray(v_h),
            truncation_error,
        )  # type: ignore
    return np.asarray(u_l), np.asarray(s), np.asarray(v_h), None  # type: ignore


def qr(
    mat: np.ndarray,
    cut: float = float(1e-12),
    chi_max: int = int(1e4),
    renormalise: bool = False,
    return_truncation_error: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Performs QR Decomposition with a possibility for truncation.

    Parameters
    ----------
    mat : np.ndarray
        Matrix provided as a ``np.ndarray`` with 2 dimensions.
    cut : float
        Threshold below which the diagonal values of R are discarded.
    chi_max : int
        Maximum number of columns/rows to keep after truncation.
    renormalise : bool
        Whether to renormalise the matrix after truncation.
    return_truncation_error : bool
        Whether to return the truncation error.

    Returns
    -------
    Q : np.ndarray
        The orthogonal matrix after truncation.
    R : np.ndarray
        The upper triangular matrix after truncation.
    truncation_error : Optional[float]
        The truncation error.
        Returned only if `return_truncation_error` is True.

    Raises
    ------
    ValueError
        If the input matrix does not have 2 dimensions.

    Notes
    -----
    The truncation is based on the magnitudes of the absolute values
    of the diagonal elements of R.
    """

    if len(mat.shape) != 2:
        raise ValueError(f"Input matrix must have 2 dimensions, got {len(mat.shape)}.")

    q_l, r_r, pivots = scipy.linalg.qr(
        _to_numpy(mat), pivoting=True, mode="economic", check_finite=False
    )

    # Determine effective rank and truncate
    abs_diag_r = np.abs(np.diag(r_r))
    effective_rank = min(int(chi_max), int(np.sum(abs_diag_r > cut)))
    trunc_idx = list(range(effective_rank))

    # Undo column pivoting without building a dense permutation matrix
    inv_piv = np.argsort(pivots)
    r_r = r_r[trunc_idx, :][:, inv_piv]
    q_l = q_l[:, trunc_idx]

    if renormalise and effective_rank > 0:
        diag = np.diag(r_r[:effective_rank, :effective_rank])
        nrm = float(np.linalg.norm(diag))
        if nrm > 0:
            r_r = r_r / nrm

    if return_truncation_error:
        trunc_error = float(np.linalg.norm(_to_numpy(mat) - q_l @ r_r))
        return np.asarray(q_l), np.asarray(r_r), trunc_error

    return np.asarray(q_l), np.asarray(r_r), None


def kron_tensors(
    tensor_1: np.ndarray,
    tensor_2: np.ndarray,
    conjugate_second: bool = False,
    merge_physicals: bool = True,
) -> np.ndarray:
    """
    Computes a Kronecker product of 2 MPS tensors.

    Parameters
    ----------
    tensor_1 : np.ndarray
        The first tensor of the product.
    tensor_2 : np.ndarray
        The second tensor of the product.
    conjugate_second : bool
        Whether to complex-conjugate the second tensor.
    merge_physicals : bool
        Whether to merge physical indices.

    Returns
    -------
    product
        The resulting Kronecker product.

    Raises
    ------
    ValueError
        If the first MPS tensor is not three-dimensional.
    ValueError
        If the second MPS tensor is not three-dimensional.

    Notes
    -----
    This function acts according to the following diagram::

           tensor_2
              i
              |                      i
        j ---( )--- k                |
                        -->    jl---( )---kn
        l ---( )--- n                |
              |                      m
              m
           tensor_1

    The legs of the resulting tensor are indexed as ``(jl, m, i, kn)``.
    Indices `i` and `m` can be merged if `merge_physicals=True`.
    """
    if tensor_1.ndim != 3:
        raise ValueError(
            f"The number of dimensions of the first tensor is {tensor_1.ndim},"
            "but the number of dimensions expected is 3."
        )
    if tensor_2.ndim != 3:
        raise ValueError(
            f"The number of dimensions of the second tensor is {tensor_2.ndim},"
            "but the number of dimensions expected is 3."
        )

    t2 = np.conjugate(tensor_2) if conjugate_second else tensor_2
    product = np.kron(tensor_1, t2)

    if not merge_physicals:
        return product.reshape(
            (
                tensor_1.shape[0] * tensor_2.shape[0],
                tensor_1.shape[1],
                tensor_2.shape[1],
                tensor_1.shape[2] * tensor_2.shape[2],
            )
        )
    return product


def split_two_site_tensor(
    tensor: np.ndarray,
    chi_max: int = int(1e4),
    cut: float = float(1e-12),
    renormalise: bool = False,
    strategy: str = "svd",
    return_truncation_error: bool = False,
) -> Tuple:
    """
    Split and truncate a two-site MPS tensor according to the following diagram
    (in case of the SVD strategy, similarly but without the singular vals for the others)::

                                             m         n
       i ---(tensor)--- l     ->    i ---(A)---diag(S)---(B)--- l
             |   |                        |               |
             j   k                        j               k

    Parameters
    ----------
    tensor : np.ndarray
        Two-site tensor ``(i, j, k, l)``.
    chi_max : int
        Maximum number of singular/diagonal values to keep.
    cut : float
        Discard any singular/diagonal values smaller than this.
    renormalise : bool
        Whether to renormalise the singular value spectrum or the R diagonal
        after the cut.
    strategy : str
        Which strategy to use for the splitting.
        Available options: ``svd`` and ``qr``.
    return_truncation_error : bool
        Whether to return the truncation error.

    Returns
    -------
    a_l : np.ndarray
        Left isometry ``(i, j, m)``.
    singular_values : np.ndarray
        List of singular values.
        Only returned if the decomposition strategy is set to ``svd``.
    b_r : np.ndarray
        Right isometry ``(n, k, l)``.
    truncation_error : Optional[float]
        The truncation error.

    Raises
    ------
    ValueError
        If the tensor is not four-dimensional.
    ValueError
        If the strategy is not one of the available ones.
    """
    if tensor.ndim != 4:
        raise ValueError(
            "A valid two-site tensor must have 4 legs"
            f"while the one given has {tensor.ndim}."
        )
    if strategy not in ["svd", "qr"]:
        raise ValueError("The strategy must be either `svd` or `qr`.")

    chi_v_l, phys_l, phys_r, chi_v_r = tensor.shape
    mat = tensor.reshape((chi_v_l * phys_l, phys_r * chi_v_r))

    if strategy == "svd":
        u_l, singular_values, v_h, truncation_error = svd(
            mat=mat,
            cut=cut,
            chi_max=chi_max,
            renormalise=renormalise,
            return_truncation_error=return_truncation_error,
        )
        s = np.asarray(singular_values, dtype=float)
        chi_v_cut = s.size
        u_l = u_l.reshape((chi_v_l, phys_l, chi_v_cut))
        v_r = v_h.reshape((chi_v_cut, phys_r, chi_v_r))
        if return_truncation_error:
            return u_l, singular_values, v_r, truncation_error  # type: ignore[return-value]
        return u_l, singular_values, v_r

    # strategy == "qr"
    q_l, r_r, truncation_error = qr(
        mat=mat,
        cut=cut,
        chi_max=chi_max,
        renormalise=renormalise,
        return_truncation_error=return_truncation_error,
    )
    chi_v_cut = min(chi_max, q_l.shape[1])
    q_l = q_l.reshape((chi_v_l, phys_l, chi_v_cut))
    r_r = r_r.reshape((chi_v_cut, phys_r, chi_v_r))
    if return_truncation_error:
        return q_l, r_r, truncation_error, None
    return q_l, r_r, None, None


def create_random_mpo(
    num_sites: int,
    bond_dimensions: List[int],
    phys_dim: int,
    which: str = "uniform",
) -> List[np.ndarray]:
    """
    Creates a random complex-valued Matrix Product Operator.

    Parameters
    ----------
    num_sites : int
        The number of sites for the MPO.
        This will be equal to the number of tensors.
    bond_dimensions : List[int]
        A list of bond dimensions.
    phys_dim : int
        Physical dimension of the tensors.
    which : str
        Specifies the distribution from which
        the matrix elements are being taken.
        Options: "uniform", "normal", "randint".

    Returns
    -------
    mpo : List[np.ndarray]
        The resulting MPO.

    Notes
    -----
    The ``bond_dimensions`` argument should be given as a list of right virtual dimensions
    without the last trivial virtual dimension. Thus, the length of the list is ``num_sites - 1``.

    The distributions available: uniform(0,1), normal, random integer {0,1}.

    Each tensor in the MPO list has legs ``(vL, vR, pU, pD)``, where ``v`` stands for "virtual",
    ``p`` -- for "physical", and ``L, R, U, D`` -- for "left", "right", "up", "down" accordingly.
    """
    bonds = [[dim, dim] for dim in bond_dimensions]
    bonds.append([1])
    bonds.insert(0, [1])
    bond_dims = list(chain.from_iterable(bonds))

    shapes = [
        (bond_dims[i], bond_dims[i + 1], phys_dim, phys_dim)
        for i in range(0, len(bond_dims) - 1, 2)
    ]

    if which == "randint":
        mpo = [
            (
                np.random.randint(0, 2, size=shapes[i])
                + 1j * np.random.randint(0, 2, size=shapes[i])
            )
            for i in range(num_sites)
        ]
    elif which == "normal":
        mpo = [
            (np.random.normal(size=shapes[i]) + 1j * np.random.normal(size=shapes[i]))
            for i in range(num_sites)
        ]
    else:
        mpo = [
            (np.random.uniform(size=shapes[i]) + 1j * np.random.uniform(size=shapes[i]))
            for i in range(num_sites)
        ]
    return mpo


def mpo_to_matrix(
    mpo: List[np.ndarray], interlace: bool = False, group: bool = False
) -> np.ndarray:
    """
    Creates a matrix from an MPO.

    Parameters
    ----------
    mpo : List[np.ndarray]
        The MPO to convert to a matrix.
    interlace : bool
        Whether to interlace the matrix' legs or not.
    group : bool
        Whether to group the matrix' legs or not, see the notes.
        Grouping means merging all the up legs into one leg and the same for the down legs.

    Returns
    -------
    matrix : np.ndarray
        The resulting matrix.

    Raises
    ------
    ValueError
        If any of the MPO tensors is not four-dimensional.

    Notes
    -----
    If ``interlace==True``, the matrix' legs will go as
    ``(p0U, p0D, p1U, p1D, ...)``, which means
    physical legs sticking up and down with the site number.
    If ``interlace==False``, the matrix' legs will go as
    ``(p0D, p1D, ..., p0U, p1U, ...)``, which means listing first
    all physical legs sticking down with the site number,
    and then all physical legs sticking up.
    This is done to adjust the matrix to the ``@`` numpy-native matrix-vector multiplication.
    Note, grouping (if wanted) is being done after the interlacing.

    An example of a matrix with ungrouped legs on three sites::

          p0U p1U p2U
         __|___|___|__
        |            |
        |____________|
           |   |   |
          p0D p1D p2D

    Each tensor in the MPO list has legs ``(vL, vR, pU, pD)``, where v stands for "virtual",
    p -- for "physical", and L, R, U, D -- for "left", "right", "up", "down" accordingly.
    Warning: will cause memory overflow for number of sites > ~20.
    """
    for i, tensor in enumerate(mpo):
        if tensor.ndim != 4:
            raise ValueError(
                f"A valid MPO tensor must have 4 legs while tensor {i} has {tensor.ndim}."
            )

    phys_dim = mpo[0].shape[2]
    num_sites = len(mpo)

    matrix = (
        np.tensordot(mpo[0], mpo[1], (1, 0))
        .transpose((0, 3, 1, 2, 4, 5))
        .reshape((-1, mpo[1].shape[1], phys_dim**2, phys_dim**2))
    )
    for i in range(num_sites - 2):
        matrix = (
            np.tensordot(matrix, mpo[i + 2], (1, 0))
            .transpose((0, 3, 1, 2, 4, 5))
            .reshape(
                (-1, mpo[i + 2].shape[1], phys_dim ** (i + 3), phys_dim ** (i + 3))
            )
        )

    matrix = matrix.reshape(([phys_dim] * num_sites * 2))

    if not interlace:
        order = list(range(1, 2 * num_sites, 2)) + list(range(0, 2 * num_sites, 2))
        matrix = matrix.transpose(order)

    if group:
        matrix = matrix.reshape((phys_dim**num_sites, phys_dim**num_sites))

    return matrix


def mpo_from_matrix(
    matrix: np.ndarray,
    num_sites: int,
    interlaced: bool = True,
    orthogonalise: bool = False,
    phys_dim: int = 2,
    chi_max: int = int(1e4),
) -> List[np.ndarray]:
    """
    Creates an MPO from a matrix (either grouped 2D or ungrouped 2N-D).

    If orthogonalise=True, carry singular values to the orthogonality centre
    (isometric tensors except centre). Otherwise split sqrt singular values
    to both sides (non-isometric, no single centre).
    """
    hilbert_space_dim = phys_dim**num_sites
    phys_dims = [phys_dim] * num_sites * 2

    if (matrix.shape != (hilbert_space_dim, hilbert_space_dim)) and (
        matrix.shape != tuple(phys_dims)
    ):
        raise ValueError(
            f"The matrix' shape should be either {(hilbert_space_dim, hilbert_space_dim)}, "
            f"or {tuple(phys_dims)}, instead, the matrix given has shape {matrix.shape}."
        )

    mat = np.array(matrix, copy=True)

    if mat.shape == (hilbert_space_dim, hilbert_space_dim):
        mat = mat.reshape(tuple(phys_dims))

    if not interlaced:
        correct_order = [(i + num_sites, i) for i in range(num_sites)]
        mat = mat.transpose(list(chain.from_iterable(correct_order)))

    # Treat as MPS with squared physical dimension
    mpo: List[np.ndarray] = []
    mps_dim = phys_dim**2
    mat2 = mat.reshape((-1, mps_dim))

    U, S, Vh, _ = svd(mat2, chi_max=chi_max, renormalise=False)
    S_np = np.asarray(S, dtype=float)
    Vh_np = np.asarray(Vh)

    if not orthogonalise:
        # v_r = sqrt(S)[:, None] * Vh
        sqrtS = np.sqrt(S_np)
        Vh_np = sqrtS[:, None] * Vh_np
    mpo.append(np.expand_dims(Vh_np, -1))

    while U.shape[0] >= mps_dim:
        if not orthogonalise:
            S_np = np.sqrt(S_np)

        # matrix = U @ diag(S)  → scale columns of U by S
        if S_np.size:
            U = U * S_np[None, :]

        bond_dim = U.shape[-1]
        U_reshaped = U.reshape((-1, mps_dim * bond_dim))
        U, S, Vh, _ = svd(U_reshaped, chi_max=chi_max, renormalise=False)
        S_np = np.asarray(S, dtype=float)
        Vh_np = np.asarray(Vh)

        if not orthogonalise:
            sqrtS = np.sqrt(S_np)
            Vh_np = sqrtS[:, None] * Vh_np

        Vh_np = Vh_np.reshape((-1, mps_dim, bond_dim))
        mpo.insert(0, Vh_np)

    # Contract orthogonality centre (avoid diag by broadcasting)
    if not orthogonalise:
        S_np = np.sqrt(S_np)
    # mpo[0] = contract("ij, jk, klm", U, diag(S), mpo[0])  -> scale rows of mpo[0] by S, then contract
    if S_np.size:
        mpo0_scaled = S_np[:, None, None] * mpo[0]  # (r, mps_dim, bond_dim)
    else:
        mpo0_scaled = mpo[0]
    mpo[0] = contract("ij, jlm -> ilm", U, mpo0_scaled, optimize=[(0, 1)])

    # Convert MPS back to MPO layout (vL, vR, pU, pD)
    mpo = [tensor.transpose((0, 2, 1)) for tensor in mpo]
    mpo = [
        tensor.reshape((tensor.shape[0], tensor.shape[1], phys_dim, phys_dim))
        for tensor in mpo
    ]
    return mpo
