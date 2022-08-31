"""This module contains miscellaneous utilities."""

import numpy as np
import scipy
from opt_einsum import contract


def svd(
    mat: np.ndarray,
    cut: np.float64 = 1e-12,
    chi_max: np.int16 = 1e4,
    renormalise: bool = False,
) -> tuple[np.ndarray]:
    """
    Performs the Singular Value Decomposition with different features.

    Parameters
    ----------
    mat : np.ndarray
        Matrix provided as a tensor with 2 dimensions.
    cut : np.float64
        Singular values smaller than this will be discarded.
    chi_max : np.int16
        Maximum number of singular values to keep.
    renormalise : bool
        Whether to renormalise the singular value spectrum.

    Returns
    -------
    u_l : np.ndarray
        Unitary matrix having left singular vectors as columns.
    singular_values : np.ndarray
        The singular values, sorted in non-increasing order.
    v_r : np.ndarray
        Unitary matrix having right singular vectors as rows.

    Raises
    ------
    ValueError
        If the tensor provided is not two-dimensional.
    """

    if len(mat.shape) != 2:
        raise ValueError(
            f"A valid matrix must have 2 dimensions while the one given has {len(mat.shape)}."
        )
    try:
        u_l, singular_values, v_r = np.linalg.svd(
            mat, full_matrices=False, compute_uv=True, hermitian=False
        )
    except np.linalg.LinAlgError:
        u_l, singular_values, v_r = scipy.linalg.svd(
            mat, full_matrices=False, compute_uv=True, lapack_driver="gesdd"
        )
    except scipy.linalg.LinAlgError:
        u_l, singular_values, v_r = scipy.linalg.svd(
            mat, full_matrices=False, compute_uv=True, lapack_driver="gesvd"
        )

    max_num = min(chi_max, np.sum(singular_values > cut))
    ind = np.argsort(singular_values)[::-1][:max_num]

    u_l, singular_values, v_r = u_l[:, ind], singular_values[ind], v_r[ind, :]

    if renormalise:
        singular_values /= np.linalg.norm(singular_values)

    return u_l, singular_values, v_r


def kron_tensors(
    tensor_1: np.ndarray,
    tensor_2: np.ndarray,
    conjugate_second: bool = False,
    merge_physicals: bool = True,
) -> np.ndarray:
    """
    Computes a kronecker product of 2 MPS tensors with different features.

    An utility function which is used to compute
    different versions of a kronecker product of 2 MPS tensors.
    This function acts according to the following cartoon.
    Note, that indices `i` and `m` can be merged if `merge_physicals=True`.

    ```
           tensor_2
              i
              |                      i
        j ---( )--- k                |
                        -->    jl---( )---kn == (jl, m, i, kn)
        l ---( )--- n                |
              |                      m
              m
           tensor_1
    ```

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
        The resulting kronecker product.

    Raises
    ------
    ValueError
        If the first MPS tensor is not three-dimensional.
    ValueError
        If the second MPS tensor is not three-dimensional.
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
    chi_max: np.int16 = 1e4,
    cut: np.float64 = 1e-12,
    renormalise: bool = False,
) -> tuple:
    """
    Split a two-site MPS tensor as follows:

    ```
                                                 m         m
           i ---(tensor)--- l     ->    i ---(A)---diag(S)---(B)--- l
                 |   |                         |               |
                 j   k                         j               k
    ```

    Parameters:
        tensor :
            Two-site tensor `(i, j, k, l)`.
        chi_max :
            Maximum number of singular values to keep.
        eps :
            Discard any singular values smaller than eps.

    Returns
        a_l :
            Left isometry `(i, j, m)`.
        singular_values :
            List of singular values.
        b_r :
            Right isometry `(m, k, l)`.
    """

    if len(tensor.shape) != 4:
        raise ValueError(
            "A valid two-site tensor must have 4 legs"
            f"while the one given has {len(tensor.shape)}."
        )

    chi_v_l, d_l, d_r, chi_v_r = tensor.shape
    tensor = tensor.reshape((chi_v_l * d_l, d_r * chi_v_r))

    u_l, singular_values, v_r = svd(
        tensor, cut=cut, chi_max=chi_max, renormalise=renormalise
    )

    chi_v_cut = len(singular_values)
    a_l = u_l.reshape((chi_v_l, d_l, chi_v_cut))
    b_r = v_r.reshape((chi_v_cut, d_r, chi_v_r))

    return a_l, singular_values, b_r


def create_random_mpo(
    num_sites: np.int16,
    bond_dimensions: list[int],
    phys_dim: np.int16,
    which: str = "uniform",
) -> list[np.ndarray]:
    """Creates a random complex-valued Matrix Product Operator.

    The `bond_dimensions` argument should be given as a list of right virtual dimensions
    without the last trivial virtual dimension. It means, the length of the list is `num_sites - 1`.

    The distributions available: `uniform(0,1)`, `normal`, `random integer {0,1}`.

    Each tensor in the MPO list has legs (vL, vR, pU, pD), where v stands for "virtual",
    p -- for "physical", and L, R, U, D -- for "left", "right", "up", "down" accordingly.

    Parameters
        num_sites:
            The number of sites for the MPO.
            This will be equal to the number of tensors.
        bond_dimensions:
            A list of bond dimensions.
        phys_dim:
            Physical dimension of the tensors.
        which:
            Specifies the distribution from which
            the matrix elements are being taken.
            Options: "uniform", "normal", "randint".
    """

    bonds = [[dim, dim] for dim in bond_dimensions]
    bonds.append([1])
    bonds.insert(0, [1])
    bonds = [item for sublist in bonds for item in sublist]

    shapes = [
        (bonds[i], bonds[i + 1], phys_dim, phys_dim)
        for i in range(0, len(bonds) - 1, 2)
    ]

    if which == "uniform":
        mpo = [
            np.random.uniform(size=shapes[i]) + 1j * np.random.uniform(size=shapes[i])
            for i in range(num_sites)
        ]

    if which == "normal":
        mpo = [
            np.random.normal(size=shapes[i]) + 1j * np.random.normal(size=shapes[i])
            for i in range(num_sites)
        ]

    if which == "randint":
        mpo = [
            np.random.randint(0, 2, size=shapes[i])
            + 1j * np.random.randint(0, 2, size=shapes[i])
            for i in range(num_sites)
        ]

    return mpo


def mpo_to_matrix(
    mpo: list[np.ndarray], interlace: bool = False, group: bool = False
) -> np.ndarray:
    """Creates a matrix from an MPO.

    If `interlace` is `True`, the matrix' legs will go as
    (p0U, p0D, p1U, p1D, ...), which means
    physical legs sticking up and down with the site number.
    If `interlace` is `False`, the matrix' legs will go as
    (p0D, p1D, ..., p0U, p1U, ...), which means listing first
    all physical legs sticking down with the site number,
    and then all physical legs sticking up.
    This is done to adjust the matrix to the @ matrix-vector multiplication.
    Note, grouping (if wanted) is being done after the interlacing.

    ```
        p0U p1U p2U
     __|___|___|__
    |            | -- example of a 3-site matrix with ungrouped legs.
    |____________|
       |   |   |
      p0D p1D p2D
    ```

    Each tensor in the MPO list has legs (vL, vR, pU, pD), where v stands for "virtual",
    p -- for "physical", and L, R, U, D -- for "left", "right", "up", "down" accordingly.
    Warning: will cause memory overflow for number of sites > ~20.

    Parameters
        mpo:
            The MPO to convert to a matrix.
        interlace:
            Whether to interlace the matrix' legs or not.
        group:
            Whether to group the matrix' legs or not.
            Grouping means merging all the up legs into one leg and the same for the down legs.
    """

    for i, tensor in enumerate(mpo):
        if len(tensor.shape) != 4:
            raise ValueError(
                f"A valid MPO tensor must have 4 legs while tensor {i} has {len(tensor.shape)}."
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
    mat: np.ndarray,
    num_sites: np.int16,
    interlaced: bool = True,
    phys_dim: np.int16 = 2,
    chi_max: np.int16 = 1e4,
) -> list[np.ndarray]:
    """Creates an MPO from a matrix.

    A utility function allowing the creation of a Matrix Product Operator from a matrix.

    If `interlaced` is `True`, the matrix' legs are considered to go as
    (p0U, p0D, p1U, p1D, ...), which means physical legs sticking up and down with the site number.
    If `interlaced` is `False`, the matrix' legs are considered to go as
    (p0D, p1D, ..., p0U, p1U, ...), which means listing first all physical legs sticking down
    with the site number, and then all physical legs sticking up.
    This is done to adjust the matrix to the @ matrix-vector multiplication.

        p0U p1U p2U
     ___|___|___|___
    |              | -- example of a 3-site matrix
    |______________|
       |   |   |
      p0D p1D p2D

    Each tensor in the `mpo` list will have legs (vL, vR, pU, pD), where v stands for "virtual",
    p -- for "physical", and L, R, U, D -- for "left", "right", "up", "down" accordingly.

    Parameters
        matrix:
            The matrix to convert to an MPO.
            Can be given with either physical legs grouped together or merged.
        interlaced:
            Whether the matrix legs are interlaced or not.
        num_sites:
            The number of sites in the MPO.
        phys_dim:
            Local dimension of the physical legs.
        chi_max:
            Maximum bond dimension allowed in the MPO.
    """

    hilbert_space_dim = phys_dim**num_sites
    phys_dims = [phys_dim] * num_sites * 2
    mps_dim = phys_dim**2

    # Checking the matrix has correct shapes for
    # a given physical dimension and number of sites.
    if (mat.shape != tuple([hilbert_space_dim] * 2)) and (
        mat.shape != tuple(phys_dims)
    ):
        raise ValueError(
            f"The matrix' shape should be either {tuple([hilbert_space_dim] * 2)}, "
            f"or {tuple(phys_dims)}, instead, the matrix given has shape {mat.shape}."
        )

    # Copying the matrix not to change the original one inplace.
    mat = mat.copy()

    # Reshaping the matrix.
    if mat.shape == tuple([hilbert_space_dim] * 2):
        mat = mat.reshape(tuple(phys_dims))

    # Dealing with possible interlacing.
    if not interlaced:
        correct_order = list([i + num_sites, i] for i in range(num_sites))
        correct_order = [item for sublist in correct_order for item in sublist]
        mat = mat.transpose(correct_order)

    # Treating the MPO as an MPS with squared physical dimensions.
    mpo = []
    mat = mat.reshape((-1, mps_dim))
    mat, singular_values, v_r = svd(mat, chi_max=chi_max, renormalise=False)
    v_r = np.matmul(np.diag(np.sqrt(singular_values)), v_r)
    mpo.append(np.expand_dims(v_r, -1))
    while mat.shape[0] >= mps_dim:
        mat = np.matmul(mat, np.diag(np.sqrt(singular_values)))
        bond_dim = mat.shape[-1]
        mat = mat.reshape((-1, mps_dim * bond_dim))
        mat, singular_values, v_r = svd(mat, chi_max=chi_max, renormalise=False)
        v_r = np.matmul(np.diag(np.sqrt(singular_values)), v_r)
        v_r = v_r.reshape((-1, mps_dim, bond_dim))
        mpo.insert(0, v_r)

    # Contracting the orthogonality centre.
    mpo[0] = contract(
        "ij, jk, klm",
        mat,
        np.diag(np.sqrt(singular_values)),
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
