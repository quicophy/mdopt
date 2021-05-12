"""
    This module contains the Matrix Product State (MPS) class and other MPS-related utilities.
"""

from functools import reduce
import numpy as np
from mpopt.utils import trimmed_svd, interlace_tensors


class MPS:
    """
    Class for a finite-size matrix product state with open boundary conditions.

    We index sites with i from 0 to L-1, with bond i being left of site i.
    Notation: the index inside the square brackets means that it is being contracted.

    The state is stored in the following format: for each tensor at site i,
    there exists a Schmidt value diagonal matrix at bond i.
    For "ghost" bonds at indices 0, L-1 (i.e., bonds of dimension 1),
    the corresponding Schmidt tensors at the boundaries
    would simply be the identities of the same dimension.

    As a convention, we will call this form the "explicit form" MPS.

    Parameters:
        tensors, schmidt_values:
            Same as attributes.

    Attributes:
        tensors : list of np.arrays[ndim=3]
            The tensors in right-canonical form, one for each physical site.
            Each tensor has legs (virtual left, physical, virtual right), in short (vL, i, vR).
        schmidt_values : list of np.arrays[ndim=1]
            The Schmidt values at each of the bonds, schmidt_values[i] is left of tensors[i].
        nsites : int
            Number of sites.
        nbonds : int
            Number of non-trivial bonds: nsites - 1.

    Exceptions:
        ValueError:
            If tensors and schmidt_values do not have corresponding lengths.
            The number of Schmidt value matrices should be equal to the number of tensors + 1,
            because there are two trivial Schmidt value matrices [1.] at each of the ghost bonds.
    """

    def __init__(self, tensors, schmidt_values):

        if len(tensors) != len(schmidt_values) - 1:
            raise ValueError(
                f"The number of tensors ({len(tensors)}) should correspond "
                f"to the number of non-trivial Schmidt value matrices ({len(tensors) - 1}), instead"
                f"given the number of non-trivial Schmidt value matrices ({len(schmidt_values) - 2})."
            )

        self.tensors = tensors
        self.schmidt_values = schmidt_values
        self.nsites = len(tensors)
        self.nbonds = self.nsites - 1

    def __len__(self):
        """
        Returns the number of sites in the MPS.
        """
        return self.nsites

    def __iter__(self):
        """
        Returns an iterator over (non-trivial schmidt_values, tensors) pair for each site.
        """
        return zip(self.schmidt_values[1:-1], self.tensors)

    def single_site_left_iso(self, site: int):
        """
        Computes single-site left isometry at a given site.
        The returned array has legs (vL, i, vR).
        """

        if site >= self.nsites:
            raise ValueError(
                f"Site given ({site}), with the number of sites in the MPS ({self.nsites})."
            )

        return np.tensordot(
            np.diag(self.schmidt_values[site]), self.tensors[site], axes=[1, 0]
        )  # vL [vL'], [vL] i vR

    def single_site_right_iso(self, site: int):
        """
        Computes single-site right isometry at a given site.
        The returned array has legs (vL, i, vR).
        """

        next_site = site + 1

        if site >= self.nsites:
            raise ValueError(
                f"Sites given ({site}, {next_site}), with the number of sites in the MPS ({self.nsites})."
            )

        return np.tensordot(
            self.tensors[site], np.diag(self.schmidt_values[next_site]), axes=[2, 0]
        )  # vL i [vR], [vR'] vR

    def two_site_left_tensor(self, site: int):
        """
        Calculates effective two-site tensor on the given site and the following one
        from two single-site left isometries.
        The returned array has legs (vL, i, j, vR).
        """

        next_site = site + 1

        if site >= self.nsites:
            raise ValueError(
                f"Sites given ({site}, {next_site}), with the number of sites in the MPS ({self.nsites})."
            )

        return np.tensordot(
            self.single_site_left_iso(site),
            self.single_site_left_iso(next_site),
            [2, 0],
        )  # vL i [vR], [vL] j vR

    def two_site_right_tensor(self, site: int):
        """
        Calculates effective two-site tensor on the given site and the following one
        from two single-site right isometries.
        The returned array has legs (vL, i, j, vR).
        """

        next_site = site + 1

        if site >= self.nsites:
            raise ValueError(
                f"Sites given ({site}, {next_site}), with the number of sites in the MPS ({self.nsites})."
            )

        return np.tensordot(
            self.single_site_right_iso(site),
            self.single_site_right_iso(next_site),
            [2, 0],
        )  # vL i [vR], [vL] j vR

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
        Returns the MPS in the right-canonical form (see formula (19) in https://arxiv.org/abs/1805.00055v2 for reference),
        given the MPS in the explicit form.
        """

        return list(self.single_site_right_iso_iter())

    def to_left_canonical(self):
        """
        Returns the MPS in the left-canonical form (see formula (19) in https://arxiv.org/abs/1805.00055v2 for reference),
        given the MPS in the explicit form.
        """

        return list(self.single_site_left_iso_iter())

    def to_mixed_canonical(self, orth_centre_index: int):
        """
        Returns the MPS in the mixed-canonical form with the orthogonality centre being located at orth_centre_index.

        Arguments:
            orth_centre_index: int
                An integer which can take values 0, 1, ..., nsites-1.
                Denotes the position of the orthogonality centre -- the only non-isometry in the new MPS.
        """

        if orth_centre_index >= self.nsites:
            raise ValueError(
                f"Orthogonality centre index given ({orth_centre_index}), with the number of sites in the MPS ({self.nsites})."
            )

        mixed_can_mps = []

        if orth_centre_index == 0:
            mixed_can_mps.append(self.tensors[0])
            for i in range(1, self.nsites):
                mixed_can_mps.append(self.single_site_right_iso(i))
            return mixed_can_mps

        if orth_centre_index == self.nsites - 1:
            for i in range(self.nsites - 1):
                mixed_can_mps.append(self.single_site_left_iso(i))
            mixed_can_mps.append(self.tensors[-1])
            return mixed_can_mps

        else:
            for i in range(orth_centre_index):
                mixed_can_mps.append(self.single_site_left_iso(i))

            mixed_can_mps.append(self.tensors[orth_centre_index])

            for i in range(orth_centre_index + 1, self.nsites):
                mixed_can_mps.append(self.single_site_right_iso(i))
            return mixed_can_mps

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

    def entanglement_entropy(self):
        """
        Return the (von Neumann) entanglement entropy for bipartitions at all of the bonds.
        """

        result = []

        for i in range(self.nbonds):

            schmidt_values = self.schmidt_values[i].copy()

            # 0*log(0) should give 0 in order to avoid warning or NaN
            schmidt_values[schmidt_values < 1e-20] = 0.0

            schmidt_values2 = schmidt_values ** 2

            if abs(np.linalg.norm(schmidt_values) - 1.0) >= 1e-14:
                raise ValueError("Schmidt spectrum not properly normalised.")

            entropy = -np.sum(schmidt_values2 * np.log(schmidt_values2))
            result.append(entropy)

        return result

    def to_dense(self):
        """
        Return the dense representation of the MPS.
        Attention: will cause memory overload for number of sites > 15!
        """

        tensors = list(self.single_site_right_iso_iter())

        return reduce(lambda a, b: np.tensordot(a, b, axes=(-1, 0)), tensors)

    def density_mpo(self):
        """
        Return the MPO representation of the density matrix defined by a given MPS.
        Each tensor in the MPO list has legs (vL, i_u, i_d, vR).
        """

        tensors = list(self.single_site_right_iso_iter())

        mpo = map(
            lambda t: interlace_tensors(
                t, t, conjugate_second=True, merge_virtuals=True
            ),
            tensors,
        )

        return mpo


def mps_from_dense(psi, dim=2, limit_max=False, max_num=1e6):
    """
    Return the Matrix Product State in an explicit form given a state in the dense (statevector) form.

    Arguments:
        psi: np.array
            State vector.
        dim: int
            Dimensionality of the local Hilbert space, d=2 for qubits.
        limit_max: bool
            Activate an upper limit to the spectrum's size.
        max_num: int
            Maximum number of the singular values to keep.

    Returns:
        mps(tensors, schmidt_values):
    """

    # checking the state vector to be the correct shape
    assert psi.flatten().shape[0] % dim == 0

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
            tensors[i], np.linalg.inv(np.diag(schmidt_values[i + 1])), axes=[2, 0]
        )

    return MPS(tensors, schmidt_values)


def split_two_site_tensor(theta, chi_max=1e5, eps=1e-16):
    """
    Split a two-site tensor.

    Split a two-site MPS tensor as follows:
          vL --(theta)-- vR     ->    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled vC).

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
        theta, cut=eps, max_num=chi_max, init_norm=True, limit_max=True
    )

    # split legs of u_l and v_r
    chi_v_cut = len(schmidt_values)
    a_l = u_l.reshape((chi_v_l, d_l, chi_v_cut))
    b_r = v_r.reshape((chi_v_cut, d_r, chi_v_r))

    return a_l, schmidt_values, b_r


def find_orth_centre(mps):
    """
    Returns an integer, corresponding to the position of the orthogonality centre of an MPS.

    Arguments:
        mps: list of np.arrays[ndim=3]
            Matrix Product State given as a list of tensors.
            Note that the MPS is not given as a class instance here.
    """

    flags_left = []
    flags_right = []

    for i, _ in enumerate(mps):

        to_be_identity_left = np.einsum("ijk, ijl -> kl", mps[i], np.conjugate(mps[i]))
        to_be_identity_right = np.einsum("ijk, ljk -> il", mps[i], np.conjugate(mps[i]))

        identity_left = np.identity(to_be_identity_left.shape[0])
        identity_right = np.identity(to_be_identity_right.shape[0])

        flags_left.append(
            np.isclose(np.linalg.norm(to_be_identity_left - identity_left), 0)
        )
        flags_right.append(
            np.isclose(np.linalg.norm(to_be_identity_right - identity_right), 0)
        )

        if not (
            np.isclose(np.linalg.norm(to_be_identity_left - identity_left), 0)
        ) and not (
            np.isclose(np.linalg.norm(to_be_identity_right - identity_right), 0)
        ):
            return i

    # Handling exceptions when the orthogonality centre might be left- or right- isometry,
    # while all the other tensors are right- or left- isometries.
    if flags_left == [True] + [False] * (len(mps) - 1) and flags_right == [
        not flag for flag in flags_left
    ]:
        return 0

    if flags_left == [True] * (len(mps) - 1) + [False] and flags_right == [
        not flag for flag in flags_left
    ]:
        return len(mps) - 1

    return None


def move_orth_centre(mps, init_pos, final_pos, d=2):
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
        d: int
            Dimensionality of the local Hilbert space.

    Exceptions:
        ValueError:
            If the MPS is not given in any of the canonical forms.
        ValueError:
            If the orthogonality centre is found at the position different from `init_pos`.
        ValueError:
            If inital_pos or final_pos does not match the MPS length.
    """

    if not is_canonical(mps):
        raise ValueError("The MPS is not in any of the canonical forms.")

    if find_orth_centre(mps) != init_pos:
        raise ValueError(
            f"The orthogonality centre position ({find_orth_centre(mps)}) "
            f"does not correspond to given initial position ({init_pos})."
        )

    L = len(mps)

    if init_pos == final_pos:

        return mps

    if init_pos >= L:
        raise ValueError(
            "Initial orthogonality centre position index does not match the MPS length."
        )

    if final_pos >= L:
        raise ValueError(
            "Final orthogonality centre position index does not match the MPS length."
        )

    # Check the sweeping direction

    # If going from left to right, keep the direction
    if init_pos < final_pos:
        begin, final = init_pos, final_pos
    # If going from right to left, reverse the direction, reverse the MPS
    if init_pos > final_pos:
        mps = [np.transpose(M, (2, 1, 0)) for M in mps[::-1]]
        begin, final = (L - 1) - init_pos, (L - 1) - final_pos

    for i in range(begin, final):

        left_most_bond = mps[i].shape[0]
        right_most_bond = mps[i + 1].shape[2]

        two_site_tensor = np.tensordot(mps[i], mps[i + 1], axes=[2, 0])
        two_site_tensor = two_site_tensor.reshape(
            (left_most_bond * d, d * right_most_bond)
        )

        u_l, schmidt_values, v_r = trimmed_svd(two_site_tensor)

        u_l = u_l.reshape((left_most_bond, d, -1))

        mps_new = np.dot(np.diag(schmidt_values), v_r)
        mps_new = mps_new.reshape((-1, d, right_most_bond))

        mps[i] = u_l
        mps[i + 1] = mps_new

    # reverse the mps again if we were going from right to left
    # to keep the state the same
    if init_pos > final_pos:
        mps = [M.transpose((2, 1, 0)) for M in mps[::-1]]

    return mps


def is_canonical(mps):
    """
    Checks if the MPS is in any of the canonical forms.
    Note, that this function takes an MPS as a list of tensors,
    not the class instance.
    """

    # check if the form is left- or right- canonical
    flags_left = []
    flags_right = []
    for _, tensor in enumerate(mps):

        to_be_identity_left = np.einsum("ijk, ijl -> kl", tensor, np.conjugate(tensor))
        to_be_identity_right = np.einsum("ijk, ljk -> il", tensor, np.conjugate(tensor))

        identity_left = np.identity(to_be_identity_left.shape[0], dtype=np.float64)
        identity_right = np.identity(to_be_identity_right.shape[0], dtype=np.float64)

        flags_left.append(
            np.isclose(np.linalg.norm(to_be_identity_left - identity_left), 0)
        )
        flags_right.append(
            np.isclose(np.linalg.norm(to_be_identity_right - identity_right), 0)
        )

    if np.array(flags_left).all() or np.array(flags_right).all():
        return True

    # check if the form is mixed-canonical
    orth_centre_index = find_orth_centre(mps)

    return orth_centre_index is not None


def to_explicit_form(mps, chi_max=1e5, eps=1e-16):
    """
    Return an MPS in the explicit form, given an MPS as a list of tensors in any of the three canonical forms.

    Arguments:
        mps: list of np.arrays[ndim=3]
            A list of tensors, where each tensor has dimenstions
            (virtual left, physical, virtual right), in short (vL, i, vR).
        chi_max: int
            Maximum bond dimension.
        eps: float
            Minimum singular values to keep.
    """

    n_sites = len(mps)
    n_bonds = n_sites - 1

    # check the "ghost" dimensions for the boundary tensors
    for i in [0, -1]:
        if len(mps[i].shape) == 2:
            mps[i] = np.expand_dims(mps[i], i)  # convention, see the MPS class

    if not is_canonical(mps):
        raise ValueError("The MPS is not in any of the canonical forms.")

    # initialise the lists for storing initial tensors and schmidt_values tensors
    mps_copy = mps.copy()
    schmidt_values = []
    tensors = []

    for i in range(n_bonds):
        j = i + 1

        two_site_tensor = np.tensordot(mps_copy[i], mps_copy[j], axes=[2, 0])
        a_i, s_i, b_j = split_two_site_tensor(two_site_tensor, chi_max=chi_max, eps=eps)

        tensors.append(a_i)
        schmidt_values.append(s_i)

    tensors.append(b_j)

    schmidt_values.append(np.array([1.0]))
    schmidt_values.insert(0, np.array([1.0]))

    return MPS(tensors, schmidt_values)


def inner_product(mps_1, mps_2):
    """
    Returns an inner product between 2 Matrix Product States.
    """

    assert len(mps_1) == len(mps_2)

    L = len(mps_1)

    mps_2 = [np.conjugate(mps_2[i]) for i in range(L)]

    tensors = []

    for i in range(L):

        dims_1 = mps_1[i].shape
        dims_2 = mps_2[i].shape

        tensors.append(
            np.einsum("ijk, ljm -> ilmk", mps_1[i], mps_2[i]).reshape(
                ((dims_1[0] * dims_2[0], dims_1[2] * dims_2[2]))
            )
        )

    product = reduce(lambda a, b: np.tensordot(a, b, axes=(-1, 0)), tensors)

    return product
