"""This module contains miscellaneous utilities."""

from typing import Tuple, Optional, List, cast
from itertools import chain
import numpy as np
import scipy
from opt_einsum import contract


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

    svd_methods = [
        lambda: np.linalg.svd(
            mat, full_matrices=False, compute_uv=True, hermitian=False
        ),
        lambda: scipy.linalg.svd(
            mat,
            full_matrices=False,
            compute_uv=True,
            check_finite=True,
            lapack_driver="gesdd",
        ),
        lambda: scipy.linalg.svd(
            mat,
            full_matrices=False,
            compute_uv=True,
            check_finite=True,
            lapack_driver="gesvd",
        ),
        lambda: scipy.linalg.svd(
            mat + np.eye(mat.shape[0], mat.shape[1]) * 1e-12,
            full_matrices=False,
            compute_uv=True,
            check_finite=True,
            lapack_driver="gesvd",
        ),
    ]
    last_exception: Optional[Exception] = None
    for method in svd_methods:
        try:
            u_l, singular_values, v_r = method()
            break
        except Exception as e:
            last_exception = e
    else:
        raise RuntimeError(f"All SVD methods failed. Last error: {last_exception}")

    max_num = min(chi_max, np.sum(singular_values > cut))
    residual_spectrum = singular_values[max_num:]
    truncation_error = np.linalg.norm(residual_spectrum) ** 2
    u_l, singular_values, v_r = (
        u_l[:, :max_num],
        singular_values[:max_num],
        v_r[:max_num, :],
    )

    if renormalise:
        singular_values /= np.linalg.norm(singular_values)

    if return_truncation_error:
        return (
            np.asarray(u_l),
            cast(list, singular_values),
            np.asarray(v_r),
            float(truncation_error),
        )

    return np.asarray(u_l), cast(list, singular_values), np.asarray(v_r), None


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
        mat, pivoting=True, mode="economic", check_finite=False
    )

    abs_diag_r = np.absolute(np.diag(r_r))
    effective_rank = min(chi_max, int(np.sum(abs_diag_r > cut)))
    truncation_indices = list(range(len(abs_diag_r)))[:effective_rank]

    permutation_matrix = np.eye(mat.shape[1])[:, pivots]
    r_r = r_r[truncation_indices, :] @ permutation_matrix.T
    q_l = q_l[:, truncation_indices]

    if renormalise:
        r_r /= np.linalg.norm(np.diag(r_r))

    if return_truncation_error:
        truncation_error = np.linalg.norm(mat - np.dot(q_l, r_r))
        return q_l, r_r, float(truncation_error)

    return q_l, r_r, None


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

    if len(tensor_1.shape) != 3:
        raise ValueError(
            f"The number of dimensions of the first tensor is {len(tensor_1.shape)},"
            "but the number of dimensions expected is 3."
        )
    if len(tensor_2.shape) != 3:
        raise ValueError(
            f"The number of dimensions of the second tensor is {len(tensor_2.shape)},"
            "but the number of dimensions expected is 3."
        )

    if conjugate_second:
        product = np.kron(tensor_1, np.conjugate(tensor_2))
    else:
        product = np.kron(tensor_1, tensor_2)

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
    tensor = tensor.reshape((chi_v_l * phys_l, phys_r * chi_v_r))

    if strategy == "svd":
        u_l, singular_values, v_r, truncation_error = svd(
            mat=tensor,
            cut=cut,
            chi_max=chi_max,
            renormalise=renormalise,
            return_truncation_error=return_truncation_error,
        )
        chi_v_cut = len(singular_values)
        u_l = u_l.reshape((chi_v_l, phys_l, chi_v_cut))
        v_r = v_r.reshape((chi_v_cut, phys_r, chi_v_r))
        if return_truncation_error:
            return (
                u_l,
                singular_values,
                v_r,
                truncation_error,
            )  # pylint: disable=unbalanced-tuple-unpacking
        return u_l, singular_values, v_r

    if strategy == "qr":
        q_l, r_r, truncation_error = qr(
            mat=tensor,
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

    return tuple()


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
            np.random.randint(0, 2, size=shapes[i])
            + 1j * np.random.randint(0, 2, size=shapes[i])
            for i in range(num_sites)
        ]

    elif which == "normal":
        mpo = [
            np.random.normal(size=shapes[i]) + 1j * np.random.normal(size=shapes[i])
            for i in range(num_sites)
        ]

    else:
        mpo = [
            np.random.uniform(size=shapes[i]) + 1j * np.random.uniform(size=shapes[i])
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
    Creates an MPO from a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to convert to an MPO.
        Can be given with either physical legs grouped together or merged (see notes).
    num_sites : int
        The number of sites in the MPO.
    interlaced : bool
        Whether the matrix' legs are interlaced or not.
    orthogonalise: bool
        Whether to make the MPO tensors isometric
        with respect to 2 physical legs and one virtual.
    phys_dim : int
        Local dimension of the physical legs.
    chi_max : int
        Maximum bond dimension allowed in the MPO.

    Returns
    -------
    mpo : List[np.ndarray]
        The resulting Matrix Product Operator.

    Raises
    ------
    ValueError
        If the matrix' shape does not correspond to ``phys_dim`` and ``num_sites``.

    Notes
    -----
    If ``interlaced==True``, the matrix' legs are considered to go as
    ``(p0U, p0D, p1U, p1D, ...)``, which means physical legs sticking up and down with site number.
    If ``interlaced==False``, the matrix' legs are considered to go as
    ``(p0D, p1D, ..., p0U, p1U, ...)``, which means listing first all physical legs sticking down
    with the site number, and then all physical legs sticking up.
    This is done to adjust the matrix to the ``@`` numpy-native matrix-vector multiplication.

    If ``orthogonalise==True``, the singular values at each bond are being carried further to the
    orthogonality centre thus making all the tensors except the latter isometric.
    The orthogonality centre is then isometric up to a multiplier which would be its norm.
    If ``orthogonalise==False``, the singular values at each bond are being splitted by taking
    the square root and distribbuted to both MPO tensors at each bond thus leaving all the tensors
    nonisometric. There is no orthogonality centre in this case.

    An example of a matrix with ungrouped legs on three sites::

          p0U p1U p2U
         __|___|___|__
        |            |
        |____________|
           |   |   |
          p0D p1D p2D

    Each tensor in the ``mpo`` list has legs ``(vL, vR, pU, pD)``, where ``v`` stands for "virtual",
    ``p`` -- for "physical", and ``L, R, U, D`` -- for "left", "right", "up", "down" accordingly.
    """

    hilbert_space_dim = phys_dim**num_sites
    phys_dims = [phys_dim] * num_sites * 2

    # Checking the matrix has correct shapes for
    # a given physical dimension and number of sites.
    if (matrix.shape != tuple([hilbert_space_dim] * 2)) and (
        matrix.shape != tuple(phys_dims)
    ):
        raise ValueError(
            f"The matrix' shape should be either {tuple([hilbert_space_dim] * 2)}, "
            f"or {tuple(phys_dims)}, instead, the matrix given has shape {matrix.shape}."
        )

    # Copying the matrix not to change the original one inplace.
    matrix = matrix.copy()

    # Reshaping the matrix.
    if matrix.shape == tuple([hilbert_space_dim] * 2):
        matrix = matrix.reshape(tuple(phys_dims))

    # Dealing with possible interlacing.
    if not interlaced:
        correct_order = list([i + num_sites, i] for i in range(num_sites))
        matrix = matrix.transpose(list(chain.from_iterable(correct_order)))

    # Treating the MPO as an MPS with squared physical dimensions.
    mpo = []
    mps_dim = phys_dim**2
    matrix = matrix.reshape((-1, mps_dim))
    matrix, singular_values, v_r, _ = svd(matrix, chi_max=chi_max, renormalise=False)
    if not orthogonalise:
        v_r = np.matmul(np.diag(np.sqrt(singular_values)), v_r)
    mpo.append(np.expand_dims(v_r, -1))
    while matrix.shape[0] >= mps_dim:
        if not orthogonalise:
            singular_values = list(np.sqrt(np.array(singular_values)))
        matrix = np.matmul(matrix, np.diag(singular_values))
        bond_dim = matrix.shape[-1]
        matrix = matrix.reshape((-1, mps_dim * bond_dim))
        matrix, singular_values, v_r, _ = svd(
            matrix, chi_max=chi_max, renormalise=False
        )
        if not orthogonalise:
            v_r = np.matmul(np.diag(np.sqrt(singular_values)), v_r)
        v_r = v_r.reshape((-1, mps_dim, bond_dim))
        mpo.insert(0, v_r)

    # Contracting in the orthogonality centre.
    if not orthogonalise:
        singular_values = list(np.sqrt(np.array(singular_values)))
    mpo[0] = contract(
        "ij, jk, klm",
        matrix,
        np.diag(singular_values),
        mpo[0],
        optimize=[(0, 1), (0, 1)],
    )

    # Converting the MPS back to the MPO.
    mpo = [tensor.transpose((0, 2, 1)) for tensor in mpo]
    mpo = [
        tensor.reshape((tensor.shape[0], tensor.shape[1], phys_dim, phys_dim))
        for tensor in mpo
    ]

    return mpo
