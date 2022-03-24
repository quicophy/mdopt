"""
This module contains functions which contract Matrix Product States with Matrix Product Operators.
"""


import numpy as np
from opt_einsum import contract
from mpopt.utils.utils import split_two_site_tensor


def mps_mpo_contract(
    mps_can, mpo, start_site=0, renormalise=False, chi_max=1e5, cut=1e-20
):
    """
    Applies an operator (not necessarily unitary) in the MPO format
    to a canonical MPS with the orthogonality centre at site `start_site`
    while optionally renormalising singular values at each bond.
    Returning the updated MPS in the canonical form.

    Note, this algorithm goes from left to right.
    In order to run from right to left, reverse both the MPS and the MPO.

    The initial configuration looks as follows with
    ---O--- depicting the orthogonality centre.

        ...---()---O----()---()---...---()---...
                   |    |    |          |
                   []---[]---[]---...---[]
                   |    |    |          |

    The contraction proceeds as described in the supplementary notes.

    Arguments:
        mpo : list[np.array[ndim=4]]
            A list of tensors, where each tensor is corresponding to
            an operator applied at a certain site.
            The operators should be ordered in correspondence with the sites.
            According to our convention, each operator has legs (vL, vR, pU, pD),
            where v stands for "virtual", p -- for "physical",
            and L, R, U, D stand for "left", "right", "up", "down".
        mps : list[np.array[ndim=3]]
            Matrix product state in the canonical form.
        start_site : int
            Index of the starting site.
        renormalise : bool
            Whether to renormalise the singular values after each contraction
            involving two neigbouring MPS sites.
        chi_max : int
            Maximum bond dimension to keep.
        cut : float
            Cutoff for the singular values.

    Returns:
        mps : list[np.array[ndim=3]]
            Resulting matrix product state in the canonical form.
    """

    mps = mps_can.copy()

    if start_site + len(mpo) > len(mps):
        raise ValueError(
            "The length of the MPO should correspond to the number of sites where it is applied, "
            f"given MPO of length {len(mpo)} and the starting site {start_site}, "
            f"while your MPS has length {len(mps)}."
        )

    orth_centre_index = start_site

    two_site_mps_mpo_tensor = contract(
        "ijk, klm, nojp, oqlr -> iprqm",
        mps[start_site],
        mps[start_site + 1],
        mpo[0],
        mpo[1],
        optimize=[(0, 1), (1, 2), (0, 1)],
    ).reshape(
        (
            mps[start_site].shape[0],
            mpo[0].shape[3],
            mpo[1].shape[3],
            mps[start_site + 1].shape[2] * mpo[1].shape[1],
        )
    )

    for i in range(len(mpo) - 2):

        mps[orth_centre_index], singular_values, b_r = split_two_site_tensor(
            two_site_mps_mpo_tensor, chi_max=chi_max, cut=cut, renormalise=renormalise
        )

        orth_centre_index += 1

        mps[orth_centre_index] = contract(
            "ij, jkl -> ikl", np.diag(singular_values), b_r, optimize=[(0, 1)]
        ).reshape(
            (
                len(singular_values),
                mpo[i + 1].shape[3],
                mpo[i + 1].shape[1],
                mps[orth_centre_index + 1].shape[0],
            )
        )

        two_site_mps_mpo_tensor = contract(
            "ijkl, lmn, komp -> ijpon",
            mps[orth_centre_index],
            mps[orth_centre_index + 1],
            mpo[i + 2],
            optimize=[(0, 1), (0, 1)],
        ).reshape(
            (
                len(singular_values),
                mps[orth_centre_index].shape[1],
                mpo[i + 2].shape[3],
                mps[orth_centre_index + 1].shape[2] * mpo[i + 2].shape[1],
            )
        )

    mps[orth_centre_index], singular_values, b_r = split_two_site_tensor(
        two_site_mps_mpo_tensor, chi_max=chi_max, cut=cut, renormalise=renormalise
    )

    mps[orth_centre_index + 1] = contract(
        "ij, jkl -> ikl", np.diag(singular_values), b_r, optimize=[(0, 1)]
    )

    return mps


def apply_two_site_unitary(lambda_0, b_1, b_2, unitary):
    """
    Applies a two-site unitary to a right-canonical MPS without
    having to compute the inverse of any singular value matrix.
    Returns back the resulting MPS tensors in the right-canonical form.

    --(lambda_0)--(b_1)--(b_2)--    ->    --(b_1_updated)--(b_2_updated)--
                    |      |                      |             |
                    (unitary)                     |             |
                    |      |                      |             |

    Unitary has legs (pUL pUR, pDL pDR), where p stands for "physical", and
    L, R, U, D -- for "left", "right", "up", "down" accordingly.

    Arguments:
        lambda_0 : list
            A list of singular values to the left of first MPS tensor.
        b_1 : np.array[ndim=3]
            The first MPS right-isometric tensor to apply the unitary to.
        b_2 : np.array[ndim=3]
            The second MPS right-isometric tensor to apply the unitary to.
        unitary : np.array[ndim=4]
            The unitary tensor we apply.

    Returns:
        b_1_updated : np.array[ndim=3]
            The first updated MPS right-isometric tensor.
        b_2_updated : np.array[ndim=3]
            The second updated MPS right-isometric tensor.
    """

    two_site_tensor_with_lambda_0 = contract(
        "ij, jkl, lmn -> ikmn", np.diag(lambda_0), b_1, b_2, optimize=[(0, 1), (0, 1)]
    )
    two_site_tensor_with_lambda_0 = contract(
        "ijkl, jkmn -> imnl", two_site_tensor_with_lambda_0, unitary, optimize=[(0, 1)]
    )

    two_site_tensor_wo_lambda_0 = contract("ijk, klm", b_1, b_2, optimize=[(0, 1)])
    two_site_tensor_wo_lambda_0 = contract(
        "ijkl, jkmn -> imnl", two_site_tensor_wo_lambda_0, unitary, optimize=[(0, 1)]
    )

    _, _, b_2_updated = split_two_site_tensor(two_site_tensor_with_lambda_0)
    b_1_updated = contract(
        "ijkl, mkl -> ijm",
        two_site_tensor_wo_lambda_0,
        np.conj(b_2_updated),
        optimize=[(0, 1)],
    )

    return b_1_updated, b_2_updated


def apply_one_site_unitary(t_1, unitary):
    """
    A function which applies a one-site unitary to a canonical MPS.

    --(t_1)--    ->    --(t_1_updated)--
        |                     |
     (unitary)                |
        |                     |

    Unitary has legs (pU, pD), where p stands for "physical", and
    U, D -- for "up", "down" accordingly.

    Arguments:
        t_1 : np.array[ndim=3]
            The MPS tensor to apply the unitary to.
        unitary : np.array[ndim=4]
            The unitary tensor we apply.

    Returns:
        t_1_updated : list[np.array[ndim=3]]
            Resulting MPS tensor.
    """
    t_1_updated = contract("ijk, jl -> ilk", t_1, unitary, optimize=[(0, 1)])
    return t_1_updated
