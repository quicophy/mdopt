"""
This module contains the explicit MPS class and relevant functions.
"""

from functools import reduce
from opt_einsum import contract
import numpy as np


class ExplicitMPS:
    """
    Class for a finite-size matrix product state (MPS) with open boundary conditions.

    We index sites with `i` from `0` to `L-1`, with bond `i` being left of site `i`.
    Notation: the index inside the square brackets means that it is being contracted.

    The state is stored in the following format: for each tensor at site `i`,
    there exists a Schmidt value diagonal matrix at bond `i`.
    For "ghost" bonds at indices `0`, `L-1` (i.e., bonds of dimension 1),
    the corresponding Schmidt tensors at the boundaries
    would simply be the identities of the same dimension.

    As a convention, we will call this form the "explicit form" of MPS.

    Attributes:
        tensors : list of `np.arrays[ndim=3]`
            The "physical" tensors of the MPS, one for each physical site.
            Each tensor has legs (virtual left, physical, virtual right), in short `(vL, i, vR)`.
        schmidt_values : list of `np.arrays[ndim=1]`
            The Schmidt values at each of the bonds, `schmidt_values[i]` is left of `tensors[i]`.
            Each Schmidt value list at each bond is normalised to 1.
        nsites : int
            Number of sites.
        nbonds : int
            Number of non-trivial bonds: `nsites - 1`.
        tolerance: float
            Absolute tolerance of the normalisation of the singular value spectrum at each bond.

    Exceptions:
        ValueError:
            If `tensors` and `schmidt_values` do not have corresponding lengths.
            The number of Schmidt value matrices should be equal to the number of tensors + 1,
            because there are two trivial Schmidt value matrices at each of the ghost bonds.
    """

    def __init__(self, tensors, schmidt_values, tolerance=1e-12):

        if len(tensors) != len(schmidt_values) - 1:
            raise ValueError(
                f"The number of tensors ({len(tensors)}) should correspond to the number "
                f"of non-trivial Schmidt value matrices ({len(tensors) - 1}), instead "
                f"the number of non-trivial Schmidt value matrices is ({len(schmidt_values) - 2})."
            )

        for i, _ in enumerate(schmidt_values):
            norm = np.linalg.norm(schmidt_values[i])
            if abs(norm - 1) > tolerance:
                raise ValueError(
                    "The norm of each Schmidt values tensor must be 1, "
                    f"instead the norm is ({norm}) at bond ({i+1})"
                )

        self.tensors = tensors
        self.schmidt_values = schmidt_values
        self.nsites = len(tensors)
        self.nbonds = self.nsites - 1
        self.tolerance = tolerance

    def __len__(self):
        """
        Returns the number of sites in the MPS.
        """
        return self.nsites

    def __iter__(self):
        """
        Returns an iterator over (schmidt_values, tensors) pair for each site.
        """
        return zip(self.schmidt_values, self.tensors)

    def single_site_left_iso(self, site: int):
        """
        Computes single-site left isometry at a given site.
        The returned array has legs `(vL, i, vR)`.
        """

        if site >= self.nsites:
            raise ValueError(
                f"Site given ({site}), with the number of sites in the MPS ({self.nsites})."
            )

        return np.tensordot(
            np.diag(self.schmidt_values[site]), self.tensors[site], (1, 0)
        )

    def single_site_right_iso(self, site: int):
        """
        Computes single-site right isometry at a given site.
        The returned array has legs `(vL, i, vR)`.
        """

        next_site = site + 1

        if site >= self.nsites:
            raise ValueError(
                f"Sites given ({site}, {next_site}), "
                f"with the number of sites in the MPS ({self.nsites})."
            )

        return np.tensordot(
            self.tensors[site], np.diag(self.schmidt_values[next_site]), (2, 0)
        )

    def two_site_left_tensor(self, site: int):
        """
        Calculates a two-site tensor on a given site and the following one
        from two single-site left isometries.
        The returned array has legs `(vL, i, j, vR)`.
        """

        if site >= self.nsites:
            raise ValueError(
                f"Sites given ({site}, {site + 1}), "
                f"with the number of sites in the MPS ({self.nsites})."
            )

        return np.tensordot(
            self.single_site_left_iso(site),
            self.single_site_left_iso(site + 1),
            (2, 0),
        )

    def two_site_right_tensor(self, site: int):
        """
        Calculates a two-site tensor on a given site and the following one
        from two single-site right isometries.
        The returned array has legs `(vL, i, j, vR)`.
        """

        if site >= self.nsites:
            raise ValueError(
                f"Sites given ({site}, {site + 1}), "
                f"with the number of sites in the MPS ({self.nsites})."
            )

        return np.tensordot(
            self.single_site_right_iso(site),
            self.single_site_right_iso(site + 1),
            (2, 0),
        )

    def single_site_left_iso_iter(self):
        """
        Returns an iterator over the left isometries for every site.
        """
        return (self.single_site_left_iso(i) for i in range(self.nsites))

    def single_site_right_iso_iter(self):
        """
        Returns an iterator over the right isometries for every site.
        """
        return (self.single_site_right_iso(i) for i in range(self.nsites))

    def to_right_canonical(self):
        """
        Returns the MPS in the right-canonical form
        (see (19) in https://arxiv.org/abs/1805.00055v2 for reference),
        given the MPS in the explicit form.
        """

        return list(self.single_site_right_iso_iter())

    def to_left_canonical(self):
        """
        Returns the MPS in the left-canonical form
        (see (19) in https://arxiv.org/abs/1805.00055v2 for reference),
        given the MPS in the explicit form.
        """

        return list(self.single_site_left_iso_iter())

    def to_mixed_canonical(self, orth_centre_index: int):
        """
        Returns the MPS in the mixed-canonical form,
        with the orthogonality centre being located at `orth_centre_index`.

        Arguments:
            orth_centre_index: int
                An integer which can take values 0, 1, ..., nsites-1.
                Denotes the position of the orthogonality centre --
                the only non-isometry in the new MPS.
        """

        if orth_centre_index >= self.nsites:
            raise ValueError(
                f"Orthogonality centre index given ({orth_centre_index}), "
                f"with the number of sites in the MPS ({self.nsites})."
            )

        def mixed_can_routine(orth_centre_index):
            mixed_can_mps = []
            for i in range(orth_centre_index):
                mixed_can_mps.append(self.single_site_right_iso(i))
            mixed_can_mps.append(self.tensors[orth_centre_index])
            for i in range(orth_centre_index + 1, self.nsites):
                mixed_can_mps.append(self.single_site_left_iso(i))
            return mixed_can_mps

        if orth_centre_index == 0:
            return self.to_right_canonical()

        if orth_centre_index == self.nsites - 1:
            return self.to_left_canonical()

        return mixed_can_routine(orth_centre_index)

    def bond_dims(self):
        """
        Returns an iterator over all bond dimensions.
        """
        return (self.tensors[i].shape[2] for i in range(self.nbonds))

    def phys_dims(self):
        """
        Returns an iterator over all physical dimensions.
        """
        return (self.tensors[i].shape[1] for i in range(self.nsites))

    def reverse(self):
        """
        Returns an inverse version of a given MPS.
        """

        reversed_tensors = list(np.transpose(t) for t in reversed(self.tensors))

        return ExplicitMPS(reversed_tensors, self.schmidt_values[::-1])

    def entanglement_entropy(self):
        """
        Returns the (von Neumann) entanglement entropy for bipartitions at all of the bonds.
        """

        def xlogx(arg):
            if arg == 0:
                return 0
            return arg * np.log(arg)

        entropy = np.zeros(shape=(self.nbonds,), dtype=np.float64)

        for bond in range(self.nbonds):
            schmidt_values = self.schmidt_values[bond].copy()
            schmidt_values[schmidt_values < self.tolerance] = 0
            schmidt_values2 = schmidt_values * schmidt_values
            entropy[bond] = -np.sum(
                np.fromiter((xlogx(s) for s in schmidt_values2), dtype=float)
            )
        return entropy

    def to_dense(self, flatten=True):
        """
        Returns the dense representation of the MPS.
        Attention: will cause memory overload for number of sites > 18!
        """

        tensors = list(self.single_site_right_iso_iter())
        dense = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), tensors)

        if flatten:
            return dense.flatten()

        return dense

    def density_mpo(self):
        """
        Returns the MPO representation (as a list of tensors)
        of the density matrix defined by a given MPS.
        Each tensor in the MPO list has legs (vL, pU, pD, vR),
        where v stands for "virtual", p -- for "physical",
        and L, R, U, D stand for "left", "right", "up", "down".
        """

        tensors = list(self.single_site_right_iso_iter())

        mpo = map(
            lambda t: interlace_tensors(
                t, t, conjugate_second=True, merge_virtuals=True
            ),
            tensors,
        )

        return list(mpo)


def mps_from_dense(psi, dim=2, limit_max=False, max_num=1e6, tolerance=1e-12):
    """
    Returns the Matrix Product State in an explicit form,
    given a state in the dense (state vector) form.

    Arguments:
        psi: np.array
            State vector.
        dim: int
            Dimensionality of the local Hilbert space, d=2 for qubits.
        limit_max: bool
            Activate an upper limit to the spectrum's size.
        max_num: int
            Maximum number of the singular values to keep.
        tolerance: float
            Absolute tolerance of the normalisation of the singular value spectrum at each bond.

    Returns:
        mps(tensors, schmidt_values):
    """

    # checking the state vector to be the correct shape
    if psi.flatten().shape[0] % dim != 0:
        raise ValueError(
            "The dimension of the flattened vector is incorrect "
            "(does not correspond to the product of local dimensions)."
        )

    tensors = []
    schmidt_values = []

    psi = psi.reshape((-1, dim))

    # Getting the first tensor and schmidt_value tensors
    psi, singular_values, v_r = trimmed_svd(psi, limit_max=limit_max, max_num=max_num)

    # Adding the first tensor and schmidt-value tensor to the corresponding lists
    # Note adding the ghost dimension to the first tensor v_r
    tensors.append(np.expand_dims(v_r, -1))
    schmidt_values.append(singular_values)

    while psi.shape[0] >= dim:

        psi = np.matmul(psi, np.diag(singular_values))

        bond_dim = psi.shape[-1]
        psi = psi.reshape((-1, dim * bond_dim))
        psi, singular_values, v_r = trimmed_svd(
            psi, limit_max=limit_max, max_num=max_num
        )
        v_r = v_r.reshape((-1, dim, bond_dim))

        # Adding the v_r and singular_values tensors to the corresponding lists
        tensors.insert(0, v_r)
        schmidt_values.insert(0, singular_values)

    # Trivial Schmidt value matrix for the ghost bond at the end
    schmidt_values.append(np.array([1.0]))

    # Fixing back the gauge
    for i, _ in enumerate(tensors):

        tensors[i] = np.tensordot(
            tensors[i], np.linalg.inv(np.diag(schmidt_values[i + 1])), (2, 0)
        )

    return ExplicitMPS(tensors, schmidt_values, tolerance=tolerance)


def split_two_site_tensor(theta, chi_max=1e5, eps=1e-13):
    """
    Split a two-site MPS tensor as follows:
          vL --(theta)-- vR     ->    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Parameters:
        theta : np.array[ndim=4]
            Two-site wave function, with legs vL, i, j, vR.
        chi_max : int
            Maximum number of singular values to keep.
        eps : float
            Discard any singular values smaller than eps.

    Returns:
        a_l : np.array[ndim=3]
            Left isometry on site i, with legs vL, i, vC.
        schmidt_values : np.array[ndim=1]
            List of Schmidt values.
        b_r : np.array[ndim=3]
            Right isometry on site j, with legs vC, j, vR.
    """

    # merge the legs to form a matrix to feed into svd
    chi_v_l, d_l, d_r, chi_v_r = theta.shape
    theta = theta.reshape((chi_v_l * d_l, d_r * chi_v_r))

    # do a trimmed svd
    u_l, schmidt_values, v_r = trimmed_svd(
        theta, cut=eps, max_num=chi_max, init_norm=True
    )

    # split legs of u_l and v_r
    chi_v_cut = len(schmidt_values)
    a_l = u_l.reshape((chi_v_l, d_l, chi_v_cut))
    b_r = v_r.reshape((chi_v_cut, d_r, chi_v_r))

    return a_l, schmidt_values, b_r


def find_orth_centre(mps):
    """
    Returns a list of integers,
    corresponding to the positions of the orthogonality centres of an MPS.

    Arguments:
        mps: list of np.arrays[ndim=3]
            Matrix Product State given as a list of tensors.
            Note that the MPS is not given as a class instance here.
    """

    mps = _add_ghost_dimensions(mps)

    length = len(mps)

    flags_left = []
    flags_right = []

    centres = []

    for i, _ in enumerate(mps):

        to_be_identity_left = contract("ijk, ijl -> kl", mps[i], np.conj(mps[i]))
        to_be_identity_right = contract("ijk, ljk -> il", mps[i], np.conj(mps[i]))

        identity_left = np.identity(to_be_identity_left.shape[0], dtype=np.float64)
        identity_right = np.identity(to_be_identity_right.shape[0], dtype=np.float64)

        flags_left.append(
            np.isclose(
                np.linalg.norm(to_be_identity_left - identity_left), 0, atol=1e-13
            )
        )

        flags_right.append(
            np.isclose(
                np.linalg.norm(to_be_identity_right - identity_right), 0, atol=1e-13
            )
        )

        if not (
            np.isclose(
                np.linalg.norm(to_be_identity_left - identity_left), 0, atol=1e-13
            )
        ) and not (
            np.isclose(
                np.linalg.norm(to_be_identity_right - identity_right), 0, atol=1e-13
            )
        ):
            centres.append(i)

    # Handling exceptions, right- and left-canonical forms, and cases
    # when the orthogonality centre might be left- or right- isometry at
    # the boundaries, while all the other tensors are the opposite isometries.
    if (flags_left == [True] + [False] * (length - 1)) or (
        flags_left == [False] * (length)
    ):
        if flags_right == [not flag for flag in flags_left]:
            centres.append(0)

    if (flags_left == [True] * (length - 1) + [False]) or (
        flags_left == [True] * (length)
    ):
        if flags_right == [not flag for flag in flags_left]:
            centres.append(length - 1)

    return centres


def move_orth_centre(mps, init_pos, final_pos):
    """
    Given an MPS with an orthogonality centre at site init_pos, returns an MPS
    with the orthogonality centre at site final_pos.

    Arguments:
        mps: list of np.arrays[ndim=3]
            Matrix Product State given as a list of tensors containing an orthogonality centre.
        init_pos: int
            Initial position of the orthogonality centre.
        final_pos: int
            Final position of the orthogonality centre.

    Exceptions:
        ValueError:
            If the MPS is not given in any of the canonical forms.
        ValueError:
            If the orthogonality centre is found at the position different from `init_pos`.
        ValueError:
            If inital_pos or final_pos does not match the MPS length.
    """

    mps = _add_ghost_dimensions(mps)

    mps = mps.copy()
    length = len(mps)
    centre = find_orth_centre(mps)

    if centre != [init_pos]:
        raise ValueError(
            f"The orthogonality centre positions ({centre}) "
            f"do not correspond to given initial position ({init_pos})."
        )

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
        u_l, schmidt_values, v_r = split_two_site_tensor(two_site_tensor)
        mps_new = np.tensordot(np.diag(schmidt_values), v_r, (1, 0))
        mps[i] = u_l
        mps[i + 1] = mps_new

    # Reverse the MPS again if we were going from right to left
    # to keep the state the same
    if init_pos > final_pos:
        mps = [np.transpose(M) for M in reversed(mps)]

    return mps


def _move_orth_centre_sigma(mps, init_pos, final_pos):
    """
    Given an MPS with an orthogonality centre at site init_pos, returns an MPS
    with the orthogonality centre at site final_pos
    and the Schmidt tensors (sigma) at each covered bond.

    Arguments:
        mps: list of np.arrays[ndim=3]
            Matrix Product State given as a list of tensors containing an orthogonality centre.
        init_pos: int
            Initial position of the orthogonality centre.
        final_pos: int
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
    mps = mps.copy()

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

    sigmas = []
    for i in range(begin, final):
        two_site_tensor = np.tensordot(mps[i], mps[i + 1], (2, 0))
        u_l, schmidt_values, v_r = split_two_site_tensor(two_site_tensor)
        schmidt_values /= np.linalg.norm(schmidt_values)

        sigmas.append(schmidt_values)
        mps_new = np.tensordot(np.diag(schmidt_values), v_r, (1, 0))

        mps[i] = u_l
        mps[i + 1] = mps_new

    # Reverse the MPS again if we were going from right to left
    # to keep the state the same
    if init_pos > final_pos:
        mps = [np.transpose(M) for M in reversed(mps)]
        sigmas = list(reversed(sigmas))

    return mps, sigmas


def is_canonical(mps):
    """
    Checks if the MPS is in any of the canonical forms.
    Note, that this function takes an MPS as a list of tensors,
    not the class instance.
    """

    mps = _add_ghost_dimensions(mps)

    # Check if the form is left- or right- canonical
    flags_left = []
    flags_right = []
    for _, tensor in enumerate(mps):

        to_be_identity_left = contract("ijk, ijl -> kl", tensor, np.conj(tensor))
        to_be_identity_right = contract("ijk, ljk -> il", tensor, np.conj(tensor))

        identity_left = np.identity(to_be_identity_left.shape[0], dtype=np.float64)
        identity_right = np.identity(to_be_identity_right.shape[0], dtype=np.float64)

        flags_left.append(
            np.isclose(
                np.linalg.norm(to_be_identity_left - identity_left), 0, atol=1e-13
            )
        )
        flags_right.append(
            np.isclose(
                np.linalg.norm(to_be_identity_right - identity_right), 0, atol=1e-13
            )
        )

    if np.array(flags_left).all() or np.array(flags_right).all():
        return True

    # Check if the form is mixed-canonical
    orth_centre_index = find_orth_centre(mps)

    return len(orth_centre_index) == 1


def _move_orth_centre_to_border(mps, init_orth_centre_index):

    mps = _add_ghost_dimensions(mps)
    mps = mps.copy()
    length = len(mps)

    if init_orth_centre_index <= length / 2:
        (mps, _) = _move_orth_centre_sigma(mps, init_orth_centre_index, 0)
        return (mps, "first")

    (mps, _) = _move_orth_centre_sigma(mps, init_orth_centre_index, length - 1)
    return (mps, "last")


def to_explicit_form(mps):
    """
    Returns an MPS in the explicit form,
    given an MPS as a list of tensors in any of the three canonical forms.

    Arguments:
        mps: list of np.arrays[ndim=3]
            A list of tensors, where each tensor has dimenstions
            (virtual left, physical, virtual right), in short (vL, i, vR).
        chi_max: int
            Maximum bond dimension.
        eps: float
            Minimum singular values to keep.
    """

    length = len(mps)

    centres = find_orth_centre(mps)

    if len(centres) != 1:
        raise ValueError("There should be a single orthogonality centre in the MPS.")

    centre = centres[0]

    mps = _add_ghost_dimensions(mps)
    mps = mps.copy()

    (mps, border) = _move_orth_centre_to_border(mps, centre)

    if border == "first":
        tensors, sigmas = _move_orth_centre_sigma(mps, 0, length - 1)
    else:
        tensors, sigmas = _move_orth_centre_sigma(mps, length - 1, 0)

    sigmas.insert(0, np.array([1.0]))
    sigmas.append(np.array([1.0]))

    ttensors = []
    for i in range(length):
        ttensors.append(
            np.tensordot(tensors[i], np.linalg.inv(np.diag(sigmas[i + 1])), (2, 0))
        )

    return ExplicitMPS(ttensors, sigmas)


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
            contract("ijk, ljm -> ilkm", mps_1[i], mps_2[i]).reshape(
                (dims_1[0] * dims_2[0], dims_1[2] * dims_2[2])
            )
        )

    product = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), tensors)

    return product[0][0]


def to_dense(mps, flatten=True):
    """
    Returns a dense representation of an MPS, given as a list of tensors.
    Attention: will cause memory overload for number of sites > 18!

    Options:
        flatten: bool
            Whether to merge all the physical indices to form a vector.
    """

    dense = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), mps)

    if flatten:
        return dense.flatten()

    return dense


def apply_two_site_unitary(b_1, b_2, lambda_0, unitary, local_dim=2):
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

    two_site_tensor = np.tensordot(b_1, b_2, (2, 0))
    two_site_tensor = contract("ijkl, jkmn -> imnl", two_site_tensor, unitary)
    dims = two_site_tensor.shape
    two_site_tensor = two_site_tensor.reshape((dims[0] * dims[1], dims[2] * dims[3]))

    theta = contract("ij, jkl, lmn -> ikmn", np.diag(lambda_0), b_1, b_2)
    theta = contract("ijkl, jkmn -> imnl", theta, unitary)
    _, _, b_2_updated = split_two_site_tensor(theta)

    dims = b_2_updated.shape

    b_1_updated = np.tensordot(
        two_site_tensor,
        np.conj(b_2_updated.reshape((dims[1] * dims[2], dims[0]))),
        (1, 0),
    )
    b_1_updated = b_1_updated.reshape((-1, local_dim, b_2_updated.shape[0]))

    return b_1_updated, b_2_updated


def apply_one_site_unitary(b, unitary):
    """
    A function which applies a one-site unitary to a right-canonical MPS.

    --(b)--    ->    --(b_updated)--
       |                     |
    (unitary)                |
       |                     |

    Unitary has legs (pU, pD), where p stands for "physical", and
    U, D -- for "up", "down" accordingly.
    """

    b_updated = contract("ijk, jl -> ilk", b, unitary)

    return b_updated


def create_product_state(num_sites, local_dim=2):
    """
    Creates |0...0> as an MPS.
    """

    tensor = np.zeros((local_dim,))
    tensor[0] = 1.0
    tensor = tensor.reshape((1, local_dim, 1))

    sigma = [1.0]

    tensors = [tensor for _ in range(num_sites)]
    sigmas = [sigma for _ in range(num_sites + 1)]

    return ExplicitMPS(tensors, sigmas)


def trimmed_svd(
    mat,
    cut=1e-13,
    max_num=1e5,
    normalise=False,
    init_norm=False,
    limit_max=False,
    err_th=1e-13,
):
    """
    Returns the Singular Value Decomposition of a matrix `mat`.

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

    u_l, singular_values, v_r = np.linalg.svd(mat, full_matrices=False)

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
