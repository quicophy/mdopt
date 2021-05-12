"""
    Some tensor-network utilities.
    See the mps.py module for more documentation.
"""

import numpy as np
from scipy.linalg import svd


def trimmed_svd(
    mat,
    cut=1e-16,
    max_num=1e6,
    normalise=False,
    init_norm=False,
    limit_max=False,
    err_th=1e-16,
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
        norm = np.linalg.norm(singular_values)
        norm_singular_values = singular_values / norm
    else:
        norm_singular_values = singular_values

    norm_sum = 0
    i = 0  # last kept singular value index
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
            one_norm = np.power(norm_singular_values[i], 2)
            norm_sum += one_norm
            one_norms.append(one_norm)
            i += 1
    else:
        while norm_sum < (1 - cut) and i < singular_values.size and one_norm > err_th:
            one_norm = np.power(norm_singular_values[i], 2)
            norm_sum += one_norm
            one_norms.append(one_norm)
            i += 1

    if normalise:
        # Final renormalisation of SVD values kept or not, returning the correct dimensions
        norm = np.linalg.norm(singular_values[:i].reshape(-1, 1))
        singular_values = singular_values[:i] / norm
        return u_l[:, :i], singular_values, v_r[:i, :]

    return u_l[:, :i], singular_values[:i], v_r[:i, :]


def interlace_tensors(tensor_1, tensor_2, conjugate_second=False, merge_virtuals=True):
    """
    An utility function which is used to compute
    different versions of a Kronecker product of 2 MPS tensors.

    Arguments:
        tensor_1: np.array[ndim=3]
            The first tensor of the product.
        tensor_2: np.array[ndim=3]
            The second tensor of the product.
        conjugate_second: bool
            Whether to complex-conjugate the second tensor.
        merge_virtuals: bool
            Whether to merge virtual indices.
    """

    if len(tensor_1.shape) != len(tensor_2.shape):
        raise ValueError("The tensors must have equal numbers of dimensions.")

    if len(tensor_1.shape) != 3:
        raise ValueError(
            f"The number of dimensions given was ({len(tensor_1.shape)}), "
            "but the number of dimensions expected is 3."
        )

    if conjugate_second:
        product = np.kron(tensor_1, np.conjugate(tensor_2))
    else:
        product = np.kron(tensor_1, tensor_2)

    if merge_virtuals:
        return product.reshape(
            (
                tensor_1.shape[0] * tensor_2.shape[0],
                tensor_1.shape[1],
                tensor_2.shape[1],
                tensor_1.shape[2] * tensor_2.shape[2],
            )
        )

    return product
