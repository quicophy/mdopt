"""
This module contains the explicit MPS class and relevant functions.
"""

from functools import reduce

import numpy as np
from opt_einsum import contract

from mpopt.utils.utils import kron_tensors, svd


class ExplicitMPS:
    """
    Class for a finite-size matrix product state (MPS) with open boundary conditions.

    We index sites with `i` from `0` to `L-1`, with bond `i` being left of site `i`.
    Notation: the index inside the square brackets means that it is being contracted.

    The state is stored in the following format: for each tensor at site `i`,
    there exists a singular values diagonal matrix at bond `i`.
    For "ghost" bonds at indices `0`, `L-1` (i.e., bonds of dimension 1),
    the corresponding singular value tensors at the boundaries
    would simply be the identities of the same dimension.

    As a convention, we will call this form the "explicit form" of MPS.

    Attributes:
        tensors : list of `np.arrays[ndim=3]`
            The "physical" tensors of the MPS, one for each physical site.
            Each tensor has legs (virtual left, physical, virtual right), in short `(vL, i, vR)`.
        singular_values : list of `np.arrays[ndim=1]`
            The singular values at each of the bonds, `singular_values[i]` is left of `tensors[i]`.
            Each singular values list at each bond is normalised to 1.
        nsites : int
            Number of sites.
        nbonds : int
            Number of non-trivial bonds: `nsites - 1`.
        tolerance : float
            Absolute tolerance of the normalisation of the singular value spectrum at each bond.

    Exceptions:
        ValueError:
            If `tensors` and `singular_values` do not have corresponding lengths.
            The number of singular value matrices should be equal to the number of tensors + 1,
            because there are two trivial singular value matrices at each of the ghost bonds.
    """

    def __init__(self, tensors, singular_values, tolerance=1e-12):

        if len(tensors) != len(singular_values) - 1:
            raise ValueError(
                f"The number of tensors ({len(tensors)}) should correspond "
                "to the number of non-trivial singular value matrices "
                f"({len(tensors) - 1}), instead the number of "
                f"non-trivial singular value matrices is ({len(singular_values) - 2})."
            )

        for i, _ in enumerate(singular_values):
            norm = np.linalg.norm(singular_values[i])
            if abs(norm - 1) > tolerance:
                raise ValueError(
                    "The norm of each singular values tensor must be 1, "
                    f"instead the norm is ({norm}) at bond ({i + 1})"
                )

        self.tensors = tensors
        self.singular_values = singular_values
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
        Returns an iterator over (singular_values, tensors) pair for each site.
        """
        return zip(self.singular_values, self.tensors)

    def copy(self):
        """
        Returns a copy of the current MPS.
        """
        return ExplicitMPS(self.tensors.copy(), self.singular_values.copy())

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
            np.diag(self.singular_values[site]), self.tensors[site], (1, 0)
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
            self.tensors[site], np.diag(self.singular_values[next_site]), (2, 0)
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

    def to_mixed_canonical(self, orth_centre_index):
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

        if orth_centre_index == 0:
            return self.to_right_canonical()

        if orth_centre_index == self.nsites - 1:
            return self.to_left_canonical()

        mixed_can_mps = []

        for i in range(orth_centre_index):
            mixed_can_mps.append(self.single_site_left_iso(i))

        orthogonality_centre = contract(
            "ij, jkl, lm -> ikm",
            np.diag(self.singular_values[orth_centre_index]),
            self.tensors[orth_centre_index],
            np.diag(self.singular_values[orth_centre_index + 1]),
            optimize=[(0, 1), (0, 1)],
        )
        mixed_can_mps.append(orthogonality_centre)

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

    def reverse(self):
        """
        Returns an inverse version of a given MPS.
        """

        reversed_tensors = list(np.transpose(t) for t in reversed(self.tensors))

        return ExplicitMPS(reversed_tensors, self.singular_values[::-1])

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
            singular_values = self.singular_values[bond].copy()
            singular_values[singular_values < self.tolerance] = 0
            singular_values2 = singular_values * singular_values
            entropy[bond] = -np.sum(
                np.fromiter((xlogx(s) for s in singular_values2), dtype=float)
            )
        return entropy

    def to_dense(self, flatten=True):
        """
        Returns the dense representation of the MPS.
        Attention: will cause memory overload for number of sites > ~20!
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
        Each tensor in the MPO list has legs (vL, vR, pU, pD),
        where v stands for "virtual", p -- for "physical",
        and L, R, U, D stand for "left", "right", "up", "down".

        This operation is depicted in the following picture.
        In the cartoon, {i,j,k,l} and {a,b,c,d} are indices.
        Here, the ()'s represent the MPS tensors, the O's ---
        the singular values tensors, the []'s --- the MPO tensors.
        The MPS with the physical legs up is complex-conjugated element-wise.
        The empty line between the MPS and its complex-conjugated version
        stands for the tensor (kronecker) product.

              i        j
          a   |        |       c              i    j
        ..---()---O---()---O---..         ab  |    |  cd
                                    --> ..---[]---[]---..
        ..---()---O---()---O---..            |    |
          b  |        |        d             k    l
             k        l
        """

        tensors = list(self.single_site_right_iso_iter())

        mpo = map(
            lambda t: kron_tensors(
                t, t, conjugate_second=True, merge_physicals=False
            ).transpose((0, 3, 2, 1)),
            tensors,
        )

        return list(mpo)


def mps_from_dense(state_vector, phys_dim=2, chi_max=1e5, tolerance=1e-12):
    """
    Returns the Matrix Product State in an explicit form,
    given a state in the dense (state-vector) form.

    Arguments:
        state_vector: np.array
            The state vector.
        phys_dim: int
            Dimensionality of the local Hilbert space.
        limit_max: bool
            Activate an upper limit to the spectrum's size.
        chi_max: int
            Maximum number of singular values to keep.
        tolerance: float
            Absolute tolerance of the normalisation of the singular value spectrum at each bond.

    Returns:
        mps(tensors, singular_values):
    """

    psi = np.copy(state_vector)

    # Checking the state vector to be the correct shape
    if psi.flatten().shape[0] % phys_dim != 0:
        raise ValueError(
            "The dimension of the flattened vector is incorrect "
            "(does not correspond to the product of local dimensions)."
        )

    tensors = []
    singular_values = []

    psi = psi.reshape((-1, phys_dim))

    # Getting the first tensor and singular values tensors
    psi, singular_values_local, v_r = svd(psi, chi_max=chi_max, renormalise=False)

    # Adding the first tensor and singular values tensor to the corresponding lists
    # Note adding the ghost dimension to the first tensor v_r
    tensors.append(np.expand_dims(v_r, -1))
    singular_values.append(singular_values_local)

    while psi.shape[0] >= phys_dim:

        psi = np.matmul(psi, np.diag(singular_values_local))

        bond_dim = psi.shape[-1]
        psi = psi.reshape((-1, phys_dim * bond_dim))
        psi, singular_values_local, v_r = svd(psi, chi_max=chi_max, renormalise=False)
        v_r = v_r.reshape((-1, phys_dim, bond_dim))

        # Adding the v_r and singular_values tensors to the corresponding lists
        tensors.insert(0, v_r)
        singular_values.insert(0, singular_values_local)

    # Trivial singular value matrix for the ghost bond at the end
    singular_values.append(np.array([1.0]))

    # Fixing back the gauge
    for i, _ in enumerate(tensors):

        tensors[i] = np.tensordot(
            tensors[i], np.linalg.inv(np.diag(singular_values[i + 1])), (2, 0)
        )

    return ExplicitMPS(tensors, singular_values, tolerance=tolerance)


def create_custom_product_state(string, phys_dim=2):
    """
    Creates a custom product state defined by the `string` argument as an MPS.
    """

    num_sites = len(string)
    tensors = []

    for k in string:

        tensor = np.zeros((phys_dim,))
        if k == "0":
            tensor[0] = 1.0
        if k == "1":
            tensor[-1] = 1.0
        if k == "+":
            for i in range(phys_dim):
                tensor[i] = 1 / np.sqrt(phys_dim)
        tensors.append(tensor)

    tensors = [tensor.reshape((1, phys_dim, 1)) for tensor in tensors]
    singular_values = [[1.0] for _ in range(num_sites + 1)]

    return ExplicitMPS(tensors, singular_values)


def create_simple_product_state(num_sites, which="0", phys_dim=2):
    """
    Creates |0...0>/|1...1>/|+...+> as an MPS.
    """

    tensor = np.zeros((phys_dim,))
    if which == "0":
        tensor[0] = 1.0
    if which == "1":
        tensor[-1] = 1.0
    if which == "+":
        for i in range(phys_dim):
            tensor[i] = 1 / np.sqrt(phys_dim)

    tensors = [tensor.reshape((1, phys_dim, 1)) for _ in range(num_sites)]
    singular_values = [[1.0] for _ in range(num_sites + 1)]

    return ExplicitMPS(tensors, singular_values)
