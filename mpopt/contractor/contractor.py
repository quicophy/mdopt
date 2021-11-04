"""
This module contains functions which contract Matrix Product States and Matrix Product Operators.
"""

import numpy as np
from opt_einsum import contract
from mpopt.mps.explicit import split_two_site_tensor


def apply_two_site_unitary(lambda_0, b_1, b_2, unitary):
    """
    A convenient way to apply a two-site unitary to a right-canonical MPS and switching back
    to the right canonical form, without having to compute the inverse of Schmidt value matrix.

    --(lambda_0)--(b_1)--(b_2)--    ->    --(b_1_updated)--(b_2_updated)--
                    |      |                      |             |
                    (unitary)                     |             |
                    |      |                      |             |

    Unitary has legs (pUL pUR, pDL pDR), where p stands for "physical", and
    L, R, U, D -- for "left", "right", "up", "down" accordingly.
    """

    two_site_tensor_with_lambda_0 = contract(
        "ij, jkl, lmn -> ikmn", np.diag(lambda_0), b_1, b_2
    )
    two_site_tensor_with_lambda_0 = contract(
        "ijkl, jkmn -> imnl", two_site_tensor_with_lambda_0, unitary
    )

    two_site_tensor_wo_lambda_0 = contract("ijk, klm", b_1, b_2)
    two_site_tensor_wo_lambda_0 = contract(
        "ijkl, jkmn -> imnl", two_site_tensor_wo_lambda_0, unitary
    )

    _, _, b_2_updated = split_two_site_tensor(two_site_tensor_with_lambda_0)
    b_1_updated = contract(
        "ijkl, mkl -> ijm", two_site_tensor_wo_lambda_0, np.conj(b_2_updated),
    )

    return b_1_updated, b_2_updated


def apply_one_site_unitary(b_1, unitary):
    """
    A function which applies a one-site unitary to a right-canonical MPS.

    --(b_1)--    ->    --(b_1_updated)--
        |                     |
     (unitary)                |
        |                     |

    Unitary has legs (pU, pD), where p stands for "physical", and
    U, D -- for "up", "down" accordingly.
    """

    b_1_updated = contract("ijk, jl -> ilk", b_1, unitary)

    return b_1_updated
