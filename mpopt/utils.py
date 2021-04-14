"""
    Some tensor-network utilities.
    See the mps.py module for more documentation.
"""

import numpy as np
from scipy.linalg import svd


def dagger(tensor):
    """
    Returns the daggered version of a given tensor.
    """

    return np.conjugate(np.transpose(tensor))


def trimmed_svd(
    mat,
    cut=1e-12,
    max_num=1e6,
    normalise=True,
    init_norm=True,
    norm_ord=2,
    limit_max=False,
    err_th=1e-12,
):
    """
    Return the Singular Value Decomposition of a matrix M.

    Arguments:
        mat: ndarray[ndim=2]
            Matrix given as a ndarray with 2 dimensions.
        cut: float
            Norm value cut for lower singular values.
        max_num: int
            Maximum number of singular values to keep.
        normalise: bool
            Activates normalisation of the final singular value spectrum.
        init_norm: bool
            Activates the use of relative norm for unormalised tensor's decomposition.
        norm_ord: int
            choose the vector normalisation order.
        limit_max: bool
            Activate an upper limit to the spectrum's size.
        err_th: float
            Singular value spectrum norm error.

    Returns:
        u_l: ndarray
            Unitary matrix having left singular vectors as columns.
        singular_values: ndarray
            The singular values, sorted in non-increasing order.
        v_r: ndarray
            Unitary matrix having right singular vectors as rows.
    """

    u_l, singular_values, v_r = svd(mat, full_matrices=False, lapack_driver="gesdd")

    # Relative norm calculated for cut evaluation
    if init_norm:
        norm = np.linalg.norm(singular_values.reshape(-1, 1), norm_ord)
        norm_singular_values = singular_values / norm
    else:
        norm_singular_values = singular_values

    norm_sum = 0
    i = 0  # vfor last svd value kept index
    one_norm = 1
    one_norms = []

    # Evaluating final SVD value kept (index), for size limit fixed or not
    if limit_max:
        while (
            (norm_sum < (1 - cut))
            and (i < max_num)
            and (i < singular_values.size)
            and (one_norm > err_th)
        ):
            one_norm = np.power(norm_singular_values[i], norm_ord)
            norm_sum += one_norm
            one_norms.append(one_norm)
            i += 1
    else:
        while norm_sum < (1 - cut) and i < singular_values.size and one_norm > err_th:
            one_norm = np.power(norm_singular_values[i], norm_ord)
            norm_sum += one_norm
            one_norms.append(one_norm)
            i += 1

    if normalise:
        # Final renormalisation of SVD values kept or not, returning the correct dimensions
        norm = np.linalg.norm(singular_values[:i].reshape(-1, 1), norm_ord)
        singular_values = singular_values[:i] / norm
        return u_l[:, :i], singular_values, v_r[:i, :]

    return u_l[:, :i], singular_values[:i], v_r[:i, :]


def tensor_product_with_dagger(tensor):
    """
    Computes the tensor product of a MPS tensor with its dagger.
    That is, the input tensor's legs being (vL, i, vR) and the output's (vL * vL, i, i, vR * vR).
    """

    product = np.kron(tensor, dagger(tensor))
    return product.reshape(
        (
            tensor.shape[0] ** 2,
            tensor.shape[1],
            tensor.shape[1],
            tensor.shape[2] ** 2,
        )
    )
