"""
    This module contains the explicit MPS construction.
"""

from functools import reduce
import numpy as np
from mpopt.utils import trimmed_svd, dagger, tensor_product_with_dagger, nlog


class ExplicitMPS:
    """
    Class for a finite-size matrix product state.

    We index sites with i from 0 to L-1, this means that bond i is left of site i.
    Note: the index inside the square brackets means that it is being contracted.

    The state is stored in the following format: for each tensor at site i,
    there exists a Schmidt value diagonal matrix at bond i.
    For "ghost" bonds (i.e., bonds of dimension 1) the corresponding Schmidt tensors
    at the boundaries would simply be the identities of the same dimension.

    As a convention, we will call this form the "explicit form" of an MPS.

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
            Number of non-trivial bonds: nsites - 1 for finite boundary conditions.

    Exceptions:
        ValueError:
            If tensors and schmidt_values have different length.
    """

    def __init__(self, tensors, schmidt_values, tolerance = 1e-14):
        if len(tensors) != len(schmidt_values):
            raise ValueError(
                f"There is a different number of tensors ({len(tensors)})"
                "and Schmidt values ({len(schmidt_values)}"
            )
        for s in schmidt_values:
            if abs(np.linalg.norm(s) - 1.0) > tolerance:
                raise ValueError("the norm of each Schmidt values tensor must be 1.0")
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
        Returns an iterator over (Schmidt value, tensor) pair for every site.
        """
        return zip(self.schmidt_values, self.tensors)

    def single_site_left_iso(self, site):
        """
        Computes single-site left isometry at a given site.
        The returned array has legs (vL, i, vR).
        """
        return np.tensordot(
            np.diag(self.schmidt_values[site]), self.tensors[site], (1, 0)
        )  # vL [vL'], [vL] i vR

    def single_site_right_iso(self, site):
        """
        Computes single-site right isometry at a given site.
        The returned array has legs (vL, i, vR).
        """

        next_site = (site + 1) % len(self)

        return np.tensordot(
            np.diag(self.tensors[site]), self.schmidt_values[next_site], (2, 0)
        )  # vL i [vR], [vR'] vR

    def two_site_left_tensor(self, site):
        """
        Calculates effective two-site tensor on the given site and the following one
        from two single-site left isometries.
        The returned array has legs (vL, i, j, vR).
        """

        next_site = (site + 1) % len(self)

        return np.tensordot(
            self.single_site_left_iso(site),
            self.single_site_left_iso(next_site),
            (2, 0),
        )  # vL i [vR], [vL] j vR

    def two_site_right_tensor(self, site):
        """
        Calculates effective two-site tensor on the given site and the following one
        from two single-site right isometries.
        The returned array has legs (vL, i, j, vR).
        """

        next_site = (site + 1) % len(self)

        return np.tensordot(
            self.single_site_right_iso(site),
            self.single_site_right_iso(next_site),
            (2, 0),
        )  # vL i [vR], [vL] j vR

    def single_site_left_iso_iter(self):
        """
        Returns an iterator over the left isometries for every site.
        """
        return (self.single_site_left_iso(i) for i in range(len(self)))

    def single_site_right_iso_iter(self):
        """
        Returns an iterator over the right isometries for every site.
        """
        return (self.single_site_right_iso(i) for i in range(len(self)))

    def bond_dims(self):
        """
        Returns an iterator over all bond dimensions.
        """
        return (self.tensors[i].shape[2] for i in range(self.nbonds))

    def entanglement_entropy(self):
        """
        Return the (von Neumann) entanglement entropy for a bipartition at any of the bonds.
        """
        entropy = np.zeros(self.nbonds)
        for bond in range(self.nbonds):
            schmidt_values = self.schmidt_values[bond].copy()
            schmidt_values[schmidt_values < self.tolerance] = 0.0
            schmidt_values2 = schmidt_values * schmidt_values
            entropy[bond](-np.sum(nlog(schmidt_values2)))
        return entropy

    def to_dense(self):
        """
        Return the dense representation of the MPS.
        """

        tensors = list(self.single_site_left_iso_iter())

        return reduce(lambda a, b: np.tensordot(a, b, axes=(-1, 0)), tensors)

    def density_mpo(self):
        """
        Return the MPO representation of the density matrix defined by a given MPS.
        Each tensor in the MPO list has legs (vL, i_u, i_d, vR).
        """

        sites = list(self.single_site_left_iso_iter())
        mpo = map(tensor_product_with_dagger, sites)

        return mpo

    def to_right_canonical(self):
        """
        Returns the MPS in the right-canonical form (see formula (19) in https://arxiv.org/abs/1805.00055v2 for reference)
        given the MPS in the explicit form.
        Now, the orthogonality centre (the only non-isometry in the MPS) is in the first position.
        """

        return list(self.single_site_right_iso_iter())

    def to_left_canonical(self):
        """
        Returns the MPS in the left-canonical form (see formula (19) in https://arxiv.org/abs/1805.00055v2 for reference)
        given the MPS in the explicit form.
        Now, the orthogonality centre (the only non-isometry in the MPS) is in the last position.
        """

        return list(self.single_site_left_iso_iter())

    def to_mixed_canonical(self, orth_centre_index):
        """
        Returns the MPS in the mixed-canonical form with the orthogonality centre being located at orth_centre_index.
        """

        last_site_index = len(self) - 1
        assert orth_centre_index <= last_site_index
        left_can_mps = self.to_left_canonical()

        return move_orth_centre(left_can_mps, last_site_index, orth_centre_index)


def mps_from_dense(psi, dim=2, limit_max=False, max_num=100):
    """
    Return the Matrix Product State in an explicit form given a state in the dense (statevector) form.

    Arguments:
        psi: ndarray
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

    # reshape into (..., dim)
    psi = psi.reshape((-1, dim))

    # Getting the first B and S tensors
    psi, singular_values, v_h = trimmed_svd(
        psi, normalise=False, limit_max=limit_max, max_num=max_num
    )

    # Adding the first tensor and schmidt-value tensor to the corresponding lists
    # Note adding the ghost dimension to the first B tensor
    tensors.insert(0, np.expand_dims(v_h, -1))
    schmidt_values.insert(0, singular_values)

    while psi.shape[0] >= dim:

        bond_dim = psi.shape[-1]
        psi = psi.reshape((-1, dim * bond_dim))
        psi, singular_values, v_h = trimmed_svd(
            psi, normalise=False, limit_max=limit_max, max_num=max_num
        )
        v_h = v_h.reshape((-1, dim, bond_dim))

        # Adding the B and S tensors to the corresponding lists
        tensors.insert(0, v_h)
        schmidt_values.insert(0, singular_values)

    return ExplicitMPS(tensors, schmidt_values)


def split_truncate_two_site_tensor(theta, chi_max, eps):
    """
    Split and truncate a two-site tensor.

    Split a two-site wave tensor as follows:
          vL --(theta)-- vR     ->    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled vC).

    Parameters:
        theta : ndarray[ndim=4]
            Two-site wave function, with legs vL, i, j, vR.
        chi_max : int
            Maximum number of singular values to keep
        eps : float
            Discard any singular values smaller than that.

    Returns:
        a_l : ndarray[ndim=3]
            Left isometry on site i, with legs vL, i, vC
        schmidt_values : ndarray[ndim=1]
            Matrix of Schmidt values.
        b_r : ndarray[ndim=3]
            Right isometry on site j, with legs vC, j, vR
    """

    # merge the legs to form a matrix to feed into svd
    chi_v_l, d_l, d_r, chi_v_r = theta.shape
    theta = theta.reshape((chi_v_l * d_l, d_r * chi_v_r))

    # do a trimmed svd
    u_l, schmidt_values, v_r = trimmed_svd(
        theta, cut=eps, max_num=chi_max, normalise=True, limit_max=True
    )

    # find the right truncation dimension
    # either chi_max, or the number of significant Schmidt values
    chi_v_cut = min(chi_max, np.sum(schmidt_values > eps))

    # keep the largest chi_v_cut singular values
    piv = np.argsort(StopIteration)[::-1][:chi_v_cut]
    u_l, schmidt_values, v_r = u_l[:, piv], schmidt_values[piv], v_r[piv, :]

    # split legs of u_l and v_r
    a_l = u_l.reshape((chi_v_l, d_l, chi_v_cut))
    b_r = v_r.reshape((chi_v_cut, d_r, chi_v_r))

    return a_l, schmidt_values, b_r


def find_orth_centre(mps):
    """
    Returns an integer, corresponding to the position of the orthogonality centre of an MPS.

    Arguments:
        mps: list of np.arrays[ndim=3]
            Matrix Product State given as a list of tensors containing an orthogonality centre
    """

    for i, tensor in enumerate(mps):
        to_be_identity = np.tensordot(tensor, dagger(tensor), axes=((0, 0), (1, 1)))
        identity = np.identity(to_be_identity.shape[0], dtype=np.float64)

        if not np.isclose(to_be_identity, identity).all():
            return i
    # TODO maybe return raiserror?
    return None


def move_orth_centre(mps, init_pos, final_pos, d=2):
    """
    Given an MPS with an orthogonality centre at site init_pos, returns an MPS
    with the orthogonality centre at site final_pos.

    Arguments:
        mps: list of np.arrays[ndim=3]
            Matrix Product State given as a list of tensors containing an orthogonality centre
        init_pos: int
            Initial position of the orthogonality centre
        final_pos: int
            Final position of the orthogonality centre
        d: int
            Dimension of the local Hilbert space

    Exceptions:
        ValueError:
            If inital_pos or final_pos do not match the MPS length
    """

    L = len(mps)

    if init_pos >= L:
        raise ValueError(
            "Initial orthogonality centre position index does not match the MPS length"
        )

    if final_pos >= L:
        raise ValueError(
            "Final orthogonality centre position index does not match the MPS length"
        )

    # check the sweeping direction

    # if going from left to right
    if init_pos < final_pos:
        begin, final = init_pos, final_pos
    # reverse the mps if going from right to left
    elif init_pos > final_pos:
        mps = [np.transpose(M, (2, 1, 0)) for M in mps[::-1]]
        begin, final = (L - 1) - init_pos, (L - 1) - final_pos
    else:
        return mps

    for i in range(begin, final):

        # bond dimensions
        left_most_bond = mps[i].shape[0]
        right_most_bond = mps[i + 1].shape[2]

        two_site_tensor = np.tensordot(mps[i], mps[i + 1], axes=(2, 0))
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
        mps = [np.transpose(M, (2, 1, 0)) for M in mps[::-1]]

    return mps


def is_canonical(mps):
    """
    Checks if the MPS is in any of the canonical forms.
    """

    orth_centre_pos = find_orth_centre(mps)

    return orth_centre_pos is not None


def to_explicit_form(mps, chi_max=1e6, eps=1e-12):
    """
    Return an MPS in the explicit form, given an MPS as a list of tensors with the orthogonality centre at unknown site (any of the canonical forms).

    Arguments:
        mps: list of np.arrays[ndim=3]
            A list of tensors, where each tensor has dimenstions
            (virtual left, physical, virtual right), in short (vL, i, vR).
        chi_max: int
            Maximum bond dimension.
        eps: float
            Minimum singular values to keep.
    """

    L = len(mps)
    n_bonds = L - 1

    # checking the "ghost" dimensions for the boundary tensors
    for i in [0, -1]:
        if len(mps[i].shape) == 2:
            mps[i] = np.expand_dims(mps[i], i)  # convention, see the MPS class

    if not is_canonical(mps):
        raise ValueError("The MPS is not in any of the canonical forms")

    # finding the orthogonality centre
    orth_centre_pos = find_orth_centre(mps)

    # move the orth centre to the first site
    mps = move_orth_centre(mps, orth_centre_pos, 0)

    # initialising the initial tensors and schmidt_values tensors
    schmidt_values = [np.ones((mps[i].shape[0]), dtype=float) for i in range(L)]
    schmidt_values[0] = np.diag(mps[0].squeeze())

    mps_class_instance = ExplicitMPS(mps, schmidt_values)

    for i in range(0, n_bonds, 2):
        j = (i + 1) % L
        theta_2 = mps_class_instance.two_site_left_tensor(i)
        a_i, s_j, b_j = split_truncate_two_site_tensor(theta_2, chi_max, eps)
        g_i = np.tensordot(schmidt_values[i] ** (-1), a_i, axes=(0, 0))
        mps_class_instance.tensors[i] = np.tensordot(g_i, np.diag(s_j), axes=(1, 0))
        mps_class_instance.schmidt_values[j] = s_j
        mps_class_instance.tensors[j] = b_j

    return mps_class_instance
