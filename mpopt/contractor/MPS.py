# This code was written by Alex Berezutskii inspired by TenPy in 2020-2021.

import numpy as np
from functools import reduce


class MPS:
    """
    Class for a finite-size matrix product state.

    We index sites with i from 0 to L-1, this means that bond i is left of site i.
    We assume that the state is in right-canonical form.

    Parameters
    ----------
    tensors, schmidt_values:
        Same as attributes.

    Attributes
    ----------
    tensors : list of np.arrays[ndim=3]
        The tensors in right-canonical form, one for each physical site
        (within the unit-cell for an infinite MPS).
        Each tensor has legs (virtual left, physical, virtual right), in short (vL, i, vR)
    schmidt_values : list of np.arrays[ndim=1]
        The Schmidt values at each of the bonds, schmidt_values[i] is left of tensors[i].
    nsites : int
        Number of sites (in the unit-cell for an infinite MPS).
    nbonds : int
        Number of (non-trivial) bonds: nsites - 1 for finite boundary conditions

    Exceptions
    __________
    ValueError :
        If tensors and schmidt_values have different length.
    """

    def __init__(self, tensors, schmidt_values):
        if len(tensors) != len(schmidt_values):
            raise ValueError(
                f"there is a different number of tensors ({len(tensors)}) and Schmidt values ({len(schmidt_values)}"
            )
        self.tensors = tensors
        self.schmidt_values = schmidt_values
        self.nsites = len(tensors)
        self.nbonds = self.nsites - 1

    def __len__(self):
        """Returns the number of sites in the MPS."""
        return self.nsites

    def __iter__(self):
        """Returns an iterator over (Schmidt value, tensor) pair for every site."""
        return zip(self.schmidt_values, self.tensors)

    def single_site_wavefunction(self, site):
        """ Calculates effective single-site wave function of the given site.

        The returned array has legs (vL, i, vR), as one of the tensors.
        """
        return np.tensordot(
            np.diag(self.schmidt_values[site]), self.tensors[site], [1, 0]
        )  # vL [vL'], [vL] i vR

    def two_sites_wavefunction(self, site):
        """ Calculates effective two-site wave function on the given site and the following one.

        The returned array has legs (vL, i, j, vR).
        """
        next_site = (site + 1) % len(self)
        return np.tensordot(
            self.single_site_wavefunction(site),
            self.tensors[next_site],
            [2, 0]
        )  # vL i [vR], [vL] j vR

    def single_site_wavefunctions(self):
        """ Returns an iterator over the effective wavefunction for every site. """
        return (self.single_site_wavefunction(i) for i in range(len(self)))

    def bond_dimensions(self):
        """ Returns an iterator over all bond dimensions.
        """
        return (self.tensors[i].shape[2] for i in range(self.nbonds))

    def entanglement_entropy(self):
        """
        Return the (von Neumann) entanglement entropy for a bipartition at any of the bonds.
        """
        bonds = range(1, len(self))
        result = []
        for i in bonds:
            S = self.schmidt_values[i].copy()
            # 0*log(0) should give 0; avoid warning or NaN.
            S[S < 1.0e-20] = 0.0
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.0) < 1.0e-14
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)

    def to_dense(self):
        """ Return the dense representation of the MPS. """
        tensors = self.single_site_wavefunctions()
        return reduce(lambda a, b: np.tensordot(a, b, axes=(-1, 0)), tensors)

    def density_mpo(self, dephased):
        """
        Return the MPO representation of the (dephased) density matrix defined by a given MPS.
        Each tensor in the MPO list has legs (virtual left, physical up, physical down, virtual right),
        in short (vL, i_u, i_d, vR).

        Parameters
        ----------
        dephased: bool
        Whether to apply dephasing channel to the MPO, i.e., rho = 0.5*(rho + Z*rho*Z.T).
        """
        sites = self.single_site_wavefunctions()
        mpo = map(tensor_product_with_dagger, sites)
        if dephased:
            mpo = dephase_mpo(mpo)
        return mpo


def tensor_product_with_dagger(tensor):
    """ Computes the tensor product of a MPS tensor with its dagger.

    That is, the input tensor have legs (vL, i, vR) and the output have legs (vL * vL, i, i, vR * vR).
    """
    product = np.kron(tensor, np.conjugate(tensor).T)
    return product.reshape(
        (
            tensor.shape[0] ** 2,
            tensor.shape[1],
            tensor.shape[1],
            tensor.shape[2] ** 2,
        )
    )


# Pauli Z
Z = np.asarray([[1.0, 0.0], [0.0, -1.0]])


def dephase_tensor(tensor):
    """ Returns 0.5*(T + Z*T*Z) for tensor T. """
    z_tensor_z = np.einsum(
        "ab, iacl, cd -> ibdl",
        Z,
        tensor,
        Z
    )
    return 0.5 * (tensor + z_tensor_z)


def dephase_mpo(mpo):
    """ Returns an iterator yielding all dephased tensor of the mpo. """
    return map(dephase_tensor, mpo)
