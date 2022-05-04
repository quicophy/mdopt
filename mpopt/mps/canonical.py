"""
This module contains functions to operate with MPS in canonical form.
The canonical Matrix Product State does not have its class, but is given
as a list of 3-dimensional tensors satisfying one of the following restrictions:

1) Right-canonical: all tensors are right isometries, i.e.,

                    ---()---      ---
                        |   |       |
                        |   |  ==   |
                        |   |       |
                    ---()---      ---

2) Left-canonical: all tensors are left isometries, i.e.,

                    ---()---      ---
                    |   |         |
                    |   |    ==   |
                    |   |         |
                    ---()---      ---

3) Mixed-canonical: all but one tensors are left or right isometries.
   This exceptional tensor will be hereafter called the orthogonality centre.

Note, that a tensor with a physical leg sticking up is considered to be complex-conjugated.
"""

from functools import reduce
from opt_einsum import contract
import numpy as np
from mpopt.mps.explicit import ExplicitMPS
from mpopt.utils.utils import split_two_site_tensor, kron_tensors


def find_orth_centre(mps, return_flags=False):
    """
    Returns a list of integers corresponding to
    positions of orthogonality centres of an MPS.

    Arguments:
        mps : list of np.arrays[ndim=3]
            Matrix Product State given as a list of tensors.
            Note that the MPS is not given as `ExplicitMPS` instance here.
    """

    mps = _add_ghost_dimensions(mps)

    length = len(mps)

    flags_left = []
    flags_right = []

    centres = []

    for i, _ in enumerate(mps):

        to_be_identity_left = contract(
            "ijk, ijl -> kl", mps[i], np.conj(mps[i]), optimize=[(0, 1)]
        )
        to_be_identity_right = contract(
            "ijk, ljk -> il", mps[i], np.conj(mps[i]), optimize=[(0, 1)]
        )

        identity_left = np.identity(to_be_identity_left.shape[0], dtype=np.float64)
        identity_right = np.identity(to_be_identity_right.shape[0], dtype=np.float64)

        norm_left = np.linalg.norm(to_be_identity_left - identity_left)
        norm_right = np.linalg.norm(to_be_identity_right - identity_right)

        flags_left.append(np.isclose(norm_left, 0, atol=1e-12))
        flags_right.append(np.isclose(norm_right, 0, atol=1e-12))

        if not (np.isclose(norm_left, 0, atol=1e-12)) and not (
            np.isclose(norm_right, 0, atol=1e-12)
        ):
            centres.append(i)

    # Handling exceptions, right- and left-canonical forms, and cases
    # when the orthogonality centre might be left- or right- isometry at
    # the boundaries, while all the other tensors are the opposite isometries.

    if flags_left in ([True] + [False] * (length - 1), [False] * length):
        if flags_right == [not flag for flag in flags_left]:
            centres.append(0)

    if flags_left in ([True] * (length - 1) + [False], [True] * length):
        if flags_right == [not flag for flag in flags_left]:
            centres.append(length - 1)

    # Handling a product state.
    if (flags_left == [True] * length) and (flags_right == [True] * length):
        centres.append(0)

    if return_flags:
        return centres, flags_left, flags_right

    return centres


def move_orth_centre(mps, init_pos, final_pos):
    """
    Given an MPS with an orthogonality centre at site init_pos, returns an MPS
    with the orthogonality centre at site final_pos.

    Arguments:
        mps : list of np.arrays[ndim=3]
            Matrix Product State given as a list of tensors containing an orthogonality centre.
        init_pos : int
            Initial position of the orthogonality centre.
        final_pos : int
            Final position of the orthogonality centre.

    Exceptions:
        ValueError:
            If `init_pos` or `final_pos` does not match the MPS length.
    """

    mps = _add_ghost_dimensions(mps)

    length = len(mps)

    if init_pos >= length:
        raise ValueError(
            "Initial orthogonality centre position index does not match the MPS length."
        )

    if final_pos >= length:
        raise ValueError(
            "Final orthogonality centre position index does not match the MPS length."
        )

    # Check the sweeping direction
    # If going from left to right, keep the direction
    if init_pos < final_pos:
        begin, final = init_pos, final_pos
    # If going from right to left, reverse the direction, reverse the MPS
    elif init_pos > final_pos:
        mps = [np.transpose(M) for M in reversed(mps)]
        begin, final = (length - 1) - init_pos, (length - 1) - final_pos
    else:
        return mps

    for i in range(begin, final):
        two_site_tensor = np.tensordot(mps[i], mps[i + 1], (2, 0))
        u_l, singular_values, v_r = split_two_site_tensor(two_site_tensor)
        mps_new = np.tensordot(np.diag(singular_values), v_r, (1, 0))
        mps[i] = u_l
        mps[i + 1] = mps_new

    # Reverse the MPS again if we were going from right to left
    # to keep the state the same
    if init_pos > final_pos:
        mps = [np.transpose(M) for M in reversed(mps)]

    return mps


def _move_orth_centre_singular_values(mps, init_pos, final_pos):
    """
    Given an MPS with an orthogonality centre at site init_pos, returns an MPS
    with the orthogonality centre at site final_pos
    and the singular-value tensors at each covered bond.

    Arguments:
        mps : list of np.arrays[ndim=3]
            Matrix Product State given as a list of tensors containing an orthogonality centre.
        init_pos : int
            Initial position of the orthogonality centre.
        final_pos : int
            Final position of the orthogonality centre.

    Exceptions:
        ValueError:
            If the MPS is not given in any of the canonical forms.
        ValueError:
            If the orthogonality centre is found at the position different from `init_pos`.
        ValueError:
            If inital_pos or final_pos does not match the MPS length.
    """

    length = len(mps)

    mps = _add_ghost_dimensions(mps)

    # Check the sweeping direction
    # If going from left to right, keep the direction
    if init_pos < final_pos:
        begin, final = init_pos, final_pos

    # If going from right to left, reverse the direction, reverse the MPS
    elif init_pos > final_pos:
        mps = [np.transpose(M) for M in reversed(mps)]
        begin, final = (length - 1) - init_pos, (length - 1) - final_pos
    else:
        return mps, []

    singular_values = []
    for i in range(begin, final):
        two_site_tensor = np.tensordot(mps[i], mps[i + 1], (2, 0))
        u_l, singular_values_bond, v_r = split_two_site_tensor(two_site_tensor)
        singular_values_bond /= np.linalg.norm(singular_values_bond)

        singular_values.append(singular_values_bond)
        mps_new = np.tensordot(np.diag(singular_values_bond), v_r, (1, 0))

        mps[i] = u_l
        mps[i + 1] = mps_new

    # Reverse the MPS again if we were going from right to left
    # to keep the state the same
    if init_pos > final_pos:
        mps = [np.transpose(M) for M in reversed(mps)]
        singular_values = list(reversed(singular_values))

    return mps, singular_values


def is_canonical(mps):
    """
    Checks if the MPS is in any of the canonical forms.
    Note, that this function takes an MPS as a list of tensors,
    not the `ExplicitMPS` class instance.
    """

    mps = _add_ghost_dimensions(mps)

    # Check if the form is left- or right- canonical
    flags_left = []
    flags_right = []
    for _, tensor in enumerate(mps):

        to_be_identity_left = contract(
            "ijk, ijl -> kl", tensor, np.conj(tensor), optimize=[(0, 1)]
        )
        to_be_identity_right = contract(
            "ijk, ljk -> il", tensor, np.conj(tensor), optimize=[(0, 1)]
        )

        identity_left = np.identity(to_be_identity_left.shape[0], dtype=np.float64)
        identity_right = np.identity(to_be_identity_right.shape[0], dtype=np.float64)

        flags_left.append(
            np.isclose(
                np.linalg.norm(to_be_identity_left - identity_left), 0, atol=1e-12
            )
        )
        flags_right.append(
            np.isclose(
                np.linalg.norm(to_be_identity_right - identity_right), 0, atol=1e-12
            )
        )

    if np.array(flags_left).all() or np.array(flags_right).all():
        return True

    # Check if the form is mixed-canonical
    orth_centre_index = find_orth_centre(mps)

    return len(orth_centre_index) == 1


def _move_orth_centre_to_border(mps, init_orth_centre_index):

    mps = _add_ghost_dimensions(mps)
    length = len(mps)

    if init_orth_centre_index <= length / 2:
        (mps, _) = _move_orth_centre_singular_values(mps, init_orth_centre_index, 0)
        return (mps, "first")

    (mps, _) = _move_orth_centre_singular_values(
        mps, init_orth_centre_index, length - 1
    )
    return (mps, "last")


def to_explicit(mps):
    """
    Returns an MPS in the explicit form,
    given an MPS as a list of tensors in any of the three canonical forms.

    Arguments:
        mps : list of np.arrays[ndim=3]
            A list of tensors, where each tensor has dimenstions
            (virtual left, physical, virtual right), in short (vL, i, vR).
        chi_max : int
            Maximum bond dimension.
        eps : float
            Minimum singular values to keep.
    """

    length = len(mps)

    centres = find_orth_centre(mps)

    if len(centres) != 1:
        raise ValueError("There should be a single orthogonality centre in the MPS.")

    centre = centres[0]

    mps = _add_ghost_dimensions(mps)

    (mps, border) = _move_orth_centre_to_border(mps, centre)

    if border == "first":
        tensors, singular_values = _move_orth_centre_singular_values(mps, 0, length - 1)
    else:
        tensors, singular_values = _move_orth_centre_singular_values(mps, length - 1, 0)

    singular_values.insert(0, np.array([1.0]))
    singular_values.append(np.array([1.0]))

    ttensors = []
    for i in range(length):
        ttensors.append(
            np.tensordot(
                tensors[i], np.linalg.inv(np.diag(singular_values[i + 1])), (2, 0)
            )
        )

    return ExplicitMPS(ttensors, singular_values)


def _add_ghost_dimensions(mps):
    """
    Adds a ghost leg to the first and last tensor.
    This is a helper function.
    """

    for i in [0, -1]:
        if len(mps[i].shape) == 2:
            mps[i] = np.expand_dims(mps[i], i)  # convention, see the MPS class
    return mps


def inner_product(mps_1, mps_2):
    """
    Returns an inner product between 2 Matrix Product States.
    Note, that this function takes an MPS as a list of tensors,
    not the `ExplicitMPS` class instance.
    """

    if len(mps_1) != len(mps_2):
        raise ValueError(
            f"The number of sites in the first MPS is ({len(mps_1)}), while "
            f"the number of sites in the second MPS is ({len(mps_2)}). "
            "The MPS's must be of equal length."
        )

    length = len(mps_1)

    mps_1 = _add_ghost_dimensions(mps_1)
    mps_2 = _add_ghost_dimensions(mps_2)

    mps_1 = [np.conj(mps_1[i]) for i in range(length)]

    tensors = []

    for i in range(length):

        dims_1 = mps_1[i].shape
        dims_2 = mps_2[i].shape

        tensors.append(
            contract("ijk, ljm -> ilkm", mps_1[i], mps_2[i], optimize=[(0, 1)]).reshape(
                (dims_1[0] * dims_2[0], dims_1[2] * dims_2[2])
            )
        )

    product = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), tensors)

    return product[0][0]


def to_dense(mps, flatten=True):
    """
    Returns a dense representation of an MPS, given as a list of tensors.
    Attention: will cause memory overload for number of sites > 20!

    Options:
        flatten: bool
            Whether to merge all the physical indices to form a vector.
    """

    dense = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), mps)

    if flatten:
        return dense.flatten()

    return dense


def to_density_mpo(mps):
    """
    Returns the MPO representation (as a list of tensors)
    of the density matrix defined by the MPS in a canonical form.
    Each tensor in the MPO list has legs (vL, vR, pU, pD),
    where v stands for "virtual", p -- for "physical",
    and L, R, U, D stand for "left", "right", "up", "down".

    This operation is depicted in the following picture.
    In the cartoon, {i,j,k,l} and {a,b,c,d} are indices.
    Here, the ()'s represent the MPS tensors, the []'s ---the MPO tensors.
    The MPS with the physical legs up is complex-conjugated element-wise.
    The empty line between the MPS and its complex-conjugated version
    stands for the tensor (kronecker) product.

          i     j
    a     |    |    c            i    j
    ..---()---()---..      ab   |    |    cd
                      --> ..---[]---[]---..
    ..---()---()---..          |    |
    b    |     |    d          k    l
         k     l
    """

    mpo = map(
        lambda t: kron_tensors(
            t, t, conjugate_second=True, merge_physicals=False
        ).transpose((0, 3, 2, 1)),
        mps,
    )

    return list(mpo)
