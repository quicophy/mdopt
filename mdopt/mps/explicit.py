"""
This module contains the :class:`ExplicitMPS` class.
Hereafter saying the MPS is in the explicit form will mean that
the state is stored in the following format:
for each three-dimensional tensor at site ``i`` denoted by ``( )``,
there exists a singular values diagonal matrix at bond ``i`` denoted by ``<>``::

           i    i    i+1  i+1
    ...---<>---( )---<>---( )---...
                |          |
                |          |

For "ghost" bonds at indices ``0``, ``L-1`` (i.e., bonds of dimension 1),
the corresponding singular value tensors at the boundaries
would be the identities of the same dimension.
We index sites with ``i`` from ``0`` to ``L-1``, with bond ``i`` being left of site ``i``.
Essentially, it corresponds to storing each ``Γ[i]`` and ``Λ[i]`` as shown in
fig.4b in reference `[1]`_.
"""

from functools import reduce
from copy import deepcopy
from typing import Iterable, List, Union, cast
import numpy as np
from opt_einsum import contract
from scipy.special import xlogy

import mdopt
from mdopt.mps.canonical import CanonicalMPS
from mdopt.utils.utils import kron_tensors


class ExplicitMPS:
    """
    Class for finite-size explicit matrix product states (MPS) with open boundary conditions.

    Attributes
    ----------
    tensors : List[np.ndarray]
        The "physical" tensors of the MPS, one for each physical site.
        Each tensor has legs (virtual left, physical, virtual right), in short ``(vL, i, vR)``.
    singular_values : List[List]
        The singular values at each of the bonds, ``singular_values[i]`` is left of ``tensors[i]``.
        Each singular values list at each bond is normalised to 1.
    tolerance : np.float32
        Absolute tolerance of the normalisation of the singular value spectrum at each bond.
    num_sites : int
        Number of sites.
    num_bonds : int
        Number of non-trivial bonds, which is equal to ``num_sites - 1``.

    Raises
    ------
    ValueError
        If the ``tensors`` and the ``singular_values`` lists do not have corresponding lengths.
        The number of singular value matrices should be equal to the number of tensors + 1,
        because there are two trivial singular value matrices at each of the ghost bonds.
    ValueError
        If any of the MPS tensors is not three-dimensional.
    ValueError
        If any of the singular-values tensors is not normalised within the `tolerance` attribute.
    """

    def __init__(
        self,
        tensors: List[np.ndarray],
        singular_values: List[List],
        tolerance: np.float32 = np.float32(1e-12),
        chi_max: int = int(1e4),
    ) -> None:
        self.tensors = tensors
        self.num_sites = len(tensors)
        self.num_bonds = self.num_sites - 1
        self.bond_dimensions = [
            self.tensors[i].shape[-1] for i in range(self.num_bonds)
        ]
        self.phys_dimensions = [self.tensors[i].shape[1] for i in range(self.num_sites)]
        self.singular_values = singular_values
        self.num_singval_mat = len(singular_values)
        self.dtype = tensors[0].dtype
        self.tolerance = tolerance
        self.chi_max = chi_max

        if self.num_sites != self.num_singval_mat - 1:
            raise ValueError(
                f"The number of tensors {self.num_sites} should correspond "
                "to the number of non-trivial singular value matrices "
                f"{len(tensors) - 1}, instead the number of "
                f"non-trivial singular value matrices is {self.num_singval_mat - 2}."
            )

        for i, tensor in enumerate(tensors):
            if len(tensor.shape) != 3:
                raise ValueError(
                    "A valid MPS tensor must have 3 legs"
                    f"while the one at site {i} has {len(tensor.shape)}."
                )

        for i, _ in enumerate(singular_values):
            norm = np.linalg.norm(singular_values[i])
            if abs(norm - 1) > tolerance:
                raise ValueError(
                    "The norm of each singular values tensor must be 1, "
                    f"instead the norm is {norm} at bond {i + 1}."
                )

    def __len__(self) -> int:
        """
        Returns the number of sites in the MPS.
        """
        return self.num_sites

    def __iter__(self) -> Iterable:
        """
        Returns an iterator over (singular_values, tensors) pair for each site.
        """
        return zip(self.singular_values, self.tensors)

    def copy(self) -> "ExplicitMPS":
        """
        Returns a copy of the current MPS.
        """
        return ExplicitMPS(
            deepcopy(self.tensors),
            deepcopy(self.singular_values),
            self.tolerance,
            self.chi_max,
        )

    def reverse(self) -> "ExplicitMPS":
        """
        Returns a reversed version of a given MPS.
        """

        reversed_tensors = list(np.transpose(t) for t in reversed(self.tensors))
        reversed_singular_values = list(reversed(self.singular_values))

        return ExplicitMPS(
            reversed_tensors, reversed_singular_values, self.tolerance, self.chi_max
        )

    def conjugate(self) -> "ExplicitMPS":
        """
        Returns a complex-conjugated version of the current MPS.
        """

        conjugated_tensors = [np.conjugate(tensor) for tensor in self.tensors]
        conjugated_sing_vals = [
            [np.conjugate(sigma) for sigma in sing_vals]
            for sing_vals in self.singular_values
        ]
        conjugated_sing_vals = cast(List[list], conjugated_sing_vals)
        return ExplicitMPS(
            conjugated_tensors,
            conjugated_sing_vals,
            self.tolerance,
            self.chi_max,
        )

    def one_site_left_iso(self, site: int) -> np.ndarray:
        """
        Computes a one-site left isometry at a given site.
        """

        if site not in range(self.num_sites):
            raise ValueError(
                f"Site given {site}, with the number of sites in the MPS {self.num_sites}."
            )

        return np.tensordot(
            np.diag(self.singular_values[site]), self.tensors[site], (1, 0)
        )

    def one_site_right_iso(self, site: int) -> np.ndarray:
        """
        Computes a one-site right isometry at a given site.
        """

        if site not in range(self.num_sites):
            raise ValueError(
                f"Site given {site}, with the number of sites in the MPS {self.num_sites}."
            )

        return np.tensordot(
            self.tensors[site], np.diag(self.singular_values[site + 1]), (2, 0)
        )

    def one_site_left_iso_iter(self) -> Iterable:
        """
        Returns an iterator over the left isometries for every site.
        """

        return (self.one_site_left_iso(i) for i in range(self.num_sites))

    def one_site_right_iso_iter(self) -> Iterable:
        """
        Returns an iterator over the right isometries for every site.
        """

        return (self.one_site_right_iso(i) for i in range(self.num_sites))

    def two_site_left_iso(self, site: int) -> np.ndarray:
        """
        Computes a two-site isometry on a given site and
        the following one from two one-site left isometries.
        """

        if site not in range(self.num_sites):
            raise ValueError(
                f"Sites given {site}, {site + 1}, "
                f"with the number of sites in the MPS {self.num_sites}."
            )

        return np.tensordot(
            self.one_site_left_iso(site),
            self.one_site_left_iso(site + 1),
            (2, 0),
        )

    def two_site_right_iso(self, site: int) -> np.ndarray:
        """
        Computes a two-site isometry on a given site and
        the following one from two one-site right isometries.
        """

        if site not in range(self.num_sites):
            raise ValueError(
                f"Sites given {site}, {site + 1}, "
                f"with the number of sites in the MPS {self.num_sites}."
            )

        return np.tensordot(
            self.one_site_right_iso(site),
            self.one_site_right_iso(site + 1),
            (2, 0),
        )

    def two_site_right_iso_iter(self) -> Iterable:
        """
        Returns an iterator over the two-site right isometries for every site and
        its right neighbour.
        """
        return (self.two_site_right_iso(i) for i in range(self.num_sites))

    def two_site_left_iso_iter(self) -> Iterable:
        """
        Returns an iterator over the two-site left isometries for every site and
        its right neighbour.
        """
        return (self.two_site_left_iso(i) for i in range(self.num_sites))

    def dense(self, flatten: bool = True) -> np.ndarray:
        """
        Returns dense representation of the MPS.

        Warning: will cause memory overload for number of sites > ~20!
        """

        tensors = list(self.one_site_right_iso_iter())
        dense = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), tensors)

        if flatten:
            return dense.flatten()

        return np.squeeze(dense)

    def density_mpo(self) -> List[np.ndarray]:
        """
        Returns the MPO representation (as a list of tensors)
        of the density matrix defined by a given MPS.
        This operation is depicted in the following picture::

                   i           j
              a    |           |          c          i     j
            ...---(*)---<>*---(*)---<>*---...    ab  |     |  cd
                                          --> ...---[ ]---[ ]---...
            ...---( )---<>----( )---<>----...        |     |
              b    |           |          d          k     l
                   k           l

        In the cartoon, `{i,j,k,l}` and `{a,b,c,d}` are single indices,
        while `ab` and `cd` denote multi indices.
        Here, the ( )'s represent the MPS tensors, the <>'s ---
        the singular values tensors, the [ ]'s --- the MPO tensors.
        The MPS with the physical legs up is complex-conjugated element-wise,
        this is denoted by the star sign.
        The empty line between the MPS and its complex-conjugated version
        stands for the tensor (kronecker) product.

        Each tensor in the MPO list has legs (vL, vR, pU, pD),
        where v stands for "virtual", p -- for "physical",
        and L, R, U, D stand for "left", "right", "up", "down".
        """

        tensors = list(self.one_site_right_iso_iter())

        mpo = map(
            lambda t: kron_tensors(
                t, t, conjugate_second=True, merge_physicals=False
            ).transpose((0, 3, 2, 1)),
            tensors,
        )

        return list(mpo)

    def entanglement_entropy(self) -> np.ndarray:
        """
        Returns the entanglement entropy for bipartitions at each of the bonds.
        """

        entropy = np.zeros(shape=(self.num_bonds,), dtype=np.float32)

        for bond in range(self.num_bonds):
            singular_values = self.singular_values[bond].copy()
            singular_values[singular_values < self.tolerance] = 0
            singular_values2 = [
                singular_value**2 for singular_value in singular_values
            ]
            entropy[bond] = -1 * np.sum(
                np.fromiter((xlogy(s, s) for s in singular_values2), dtype=np.float32)
            )
        return entropy

    def right_canonical(self) -> CanonicalMPS:
        """
        Returns the current MPS in the right-canonical form given the MPS in the explicit form.

        (see eq.19 in https://scipost.org/10.21468/SciPostPhysLectNotes.5 for reference),
        """

        return CanonicalMPS(
            list(self.one_site_right_iso_iter()),
            orth_centre=None,
            tolerance=self.tolerance,
            chi_max=self.chi_max,
        )

    def left_canonical(self) -> CanonicalMPS:
        """
        Returns the current MPS in the left-canonical form given the MPS in the explicit form.

        (see eq.19 in https://scipost.org/10.21468/SciPostPhysLectNotes.5 for reference),
        """

        return CanonicalMPS(
            list(self.one_site_left_iso_iter()),
            orth_centre=None,
            tolerance=self.tolerance,
            chi_max=self.chi_max,
        )

    def mixed_canonical(self, orth_centre: int) -> CanonicalMPS:
        """
        Returns the current MPS in the mixed-canonical form
        with the orthogonality centre being located at ``orth_centre``.

        Parameters
        ----------
        orth_centre_index : int
            An integer which can take values ``0, 1, ..., num_sites-1``.
            Denotes the position of the orthogonality centre --
            the only non-isometry in the new canonical MPS.
        """

        if orth_centre not in range(self.num_sites):
            raise ValueError(
                f"Orthogonality centre index given {orth_centre}, "
                f"with the number of sites in the MPS {self.num_sites}."
            )

        mixed_can_mps = []

        for i in range(orth_centre):
            mixed_can_mps.append(self.one_site_left_iso(i))

        orth_centre_tensor = contract(
            "ij, jkl, lm -> ikm",
            np.diag(self.singular_values[orth_centre]),
            self.tensors[orth_centre],
            np.diag(self.singular_values[orth_centre + 1]),
            optimize=[(0, 1), (0, 1)],
        )
        mixed_can_mps.append(orth_centre_tensor)

        for i in range(orth_centre + 1, self.num_sites):
            mixed_can_mps.append(self.one_site_right_iso(i))

        return CanonicalMPS(
            mixed_can_mps,
            orth_centre=orth_centre,
            tolerance=self.tolerance,
            chi_max=self.chi_max,
        )

    def norm(self) -> np.float64:
        """
        Computes the norm of the current MPS, that is,
        the modulus squared of its inner product with itself.
        """

        return abs(mdopt.mps.utils.inner_product(self, self)) ** 2  # type: ignore

    def one_site_expectation_value(
        self,
        site: int,
        operator: np.ndarray,
    ) -> Union[np.float64, np.complex128]:
        """
        Computes an expectation value of an arbitrary one-site operator
        (not necessarily unitary) on the given site.

        Parameters
        ----------
        site : int
            The site where the operator is applied.
        operator : np.ndarray
            The one-site operator

        Notes
        -----
        An example of a one-site expectation value is shown in the following diagram::

            ( )-<>-( )-<>-( )-<>-( )
             |      |      |      |
             |     (o)     |      |
             |      |      |      |
            ( )-<>-( )-<>-( )-<>-( )
        """

        if site not in range(self.num_sites):
            raise ValueError(
                f"Site given {site}, with the number of sites in the MPS {self.num_sites}."
            )

        if len(operator.shape) != 2:
            raise ValueError(
                "A valid one-site operator must have 2 legs"
                f"while the one given has {len(operator.shape)}."
            )

        orthogonality_centre = contract(
            "ij, jkl, lm -> ikm",
            np.diag(self.singular_values[site]),
            self.tensors[site],
            np.diag(self.singular_values[site + 1]),
            optimize=[(0, 1), (0, 1)],
        )

        return contract(
            "ijk, jl, ilk",
            orthogonality_centre,
            operator,
            np.conjugate(orthogonality_centre),
            optimize=[(0, 1), (0, 1)],
        )

    def two_site_expectation_value(
        self,
        site: int,
        operator: np.ndarray,
    ) -> Union[np.float64, np.complex128]:
        """
        Computes an expectation value of an arbitrary two-site operator
        (not necessarily unitary) on the given site and its next neighbour.
        The operator has legs ``(UL, DL, UR, DR)``, where ``L``, ``R``, ``U``, ``D``
        stand for "left", "right", "up", "down" accordingly.

        Parameters
        ----------
        site : int
            The first site where the operator is applied, the second site to be ``site + 1``.
        operator : np.ndarray
            The two-site operator.

        Notes
        -----
        An example of a two-site expectation value is shown in the following diagram::

            ( )-<>-( )-<>-( )-<>-( )
             |      |      |      |
             |     (operator)     |
             |      |      |      |
            ( )-<>-( )-<>-( )-<>-( )

        """

        if site not in range(self.num_sites - 1):
            raise ValueError(
                f"Sites given {site, site + 1},"
                f"with the number of sites in the MPS {self.num_sites}."
            )

        if len(operator.shape) != 4:
            raise ValueError(
                "A valid two-site operator must have 4 legs"
                f"while the one given has {len(operator.shape)}."
            )

        two_site_orthogonality_centre = contract(
            "ij, jkl, lm, mno, op -> iknp",
            np.diag(self.singular_values[site]),
            self.tensors[site],
            np.diag(self.singular_values[site + 1]),
            self.tensors[site + 1],
            np.diag(self.singular_values[site + 2]),
            optimize=[(0, 1), (0, 1), (0, 1), (0, 1)],
        )

        return contract(
            "ijkl, jmkn, imnl",
            two_site_orthogonality_centre,
            operator,
            np.conjugate(two_site_orthogonality_centre),
            optimize=[(0, 1), (0, 1)],
        )

    def marginal(self, sites_to_marginalise: List[int], canonicalise: bool = False) -> "CanonicalMPS":  # type: ignore
        r"""
        Computes a marginal over a subset of sites of an MPS.
        Attention, this method does not act inplace, but creates a new object.

        Parameters
        ----------
        sites_to_marginalise : List[int]
            The sites to marginalise over.
        canonicalise : bool
            Whether to put the result in the canonical form,
            i.e., whether to sweep with SVDs over the left bonds.

        Notes
        -----
        The algorithm proceeds by starting from the bond possessing the
        maximum dimension and contracting the marginalised tensors into
        the tensors left untouched. The list of sites to marginalise is then
        updated by removing the corresponding site from it. This subroutine
        continues until the list of sites to marginalise is empty.
        An example of marginalising is shown in the following diagram::

            -<>-(0)-<>-(1)-<>-(2)-<>-(3)-<>-    ---(2)---(3)---
                 |      |      |      |      ->     |     |
                 |      |      |      |             |     |
                (t)    (t)

        Here, the ``(t)`` (trace) tensor is a tensor consisting af all 1's.
        """

        mps_can = self.right_canonical()
        sites_all = list(range(self.num_sites))

        if sites_to_marginalise == []:
            return self.right_canonical()

        if sites_to_marginalise == sites_all:
            plus_state = mdopt.mps.utils.create_simple_product_state(  # type: ignore
                num_sites=self.num_sites, which="+", phys_dim=self.phys_dimensions[0]
            )
            result = mdopt.mps.utils.inner_product(self, plus_state)  # type: ignore
            result *= np.prod(self.phys_dimensions)
            return result  # type: ignore

        for site in sites_to_marginalise:
            if site not in sites_all:
                raise ValueError(
                    "The list of sites to marginalise must be a subset of the list of all sites."
                )

        # This subroutine will be used to update the list of sites to marginalise on the fly.
        def _update_sites_routine(site_checked, site):
            if site_checked < site:
                return site_checked
            if site_checked > site:
                return site_checked - 1
            return None

        # This subroutine will be used to update the MPS attributes on the fly.
        def _update_attributes_routine(
            tensors, num_sites, bond_dims, sites_all, sites_to_marg, site
        ):
            del tensors[site]
            try:
                del bond_dims[site]
            except IndexError:
                pass
            sites_to_marg = [
                _update_sites_routine(site_checked, site)
                for site_checked in sites_to_marg
                if site_checked != site
            ]
            return tensors, num_sites - 1, bond_dims, sites_all[:-1], sites_to_marg

        # Contracting in the "t" tensors.
        for site in sites_to_marginalise:
            phys_dim = mps_can.phys_dimensions[site]
            trace_tensor = np.ones((phys_dim,))
            mps_can.tensors[site] = np.tensordot(
                mps_can.tensors[site], trace_tensor, (1, 0)
            )

        bond_dims = mps_can.bond_dimensions

        while sites_to_marginalise:
            try:
                site = int(np.argmax(bond_dims))
            except ValueError:
                site = sites_to_marginalise[0]

            # Checking all possible tensor layouts on a given bond.
            try:
                if (
                    mps_can.tensors[site].ndim == 2
                    and mps_can.tensors[site + 1].ndim == 3
                ):
                    mps_can.tensors[site + 1] = np.tensordot(
                        mps_can.tensors[site], mps_can.tensors[site + 1], (-1, 0)
                    )
                    (
                        mps_can.tensors,
                        mps_can.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                    ) = _update_attributes_routine(
                        mps_can.tensors,
                        mps_can.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                        site,
                    )
                elif (
                    mps_can.tensors[site].ndim == 3
                    and mps_can.tensors[site + 1].ndim == 2
                ):
                    mps_can.tensors[site] = np.tensordot(
                        mps_can.tensors[site], mps_can.tensors[site + 1], (-1, 0)
                    )
                    (
                        mps_can.tensors,
                        mps_can.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                    ) = _update_attributes_routine(
                        mps_can.tensors,
                        mps_can.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                        site + 1,
                    )
                elif (
                    mps_can.tensors[site].ndim == 2
                    and mps_can.tensors[site + 1].ndim == 2
                ):
                    mps_can.tensors[site] = np.tensordot(
                        mps_can.tensors[site], mps_can.tensors[site + 1], (-1, 0)
                    )
                    (
                        mps_can.tensors,
                        mps_can.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                    ) = _update_attributes_routine(
                        mps_can.tensors,
                        mps_can.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                        site + 1,
                    )
            except IndexError:
                if (
                    mps_can.tensors[site].ndim == 2
                    and mps_can.tensors[site - 1].ndim == 3
                ):
                    mps_can.tensors[site - 1] = np.tensordot(
                        mps_can.tensors[site - 1], mps_can.tensors[site], (-1, 0)
                    )
                    (
                        mps_can.tensors,
                        mps_can.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                    ) = _update_attributes_routine(
                        mps_can.tensors,
                        mps_can.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                        site,
                    )
            else:
                try:
                    del bond_dims[site]
                except IndexError:
                    pass

        if canonicalise:
            return cast(
                CanonicalMPS,
                CanonicalMPS(
                    tensors=mps_can.tensors,
                    orth_centre=mps_can.num_sites - 1,
                    tolerance=mps_can.tolerance,
                    chi_max=mps_can.chi_max,
                ).move_orth_centre(0, renormalise=False),
            )

        return cast(
            CanonicalMPS,
            CanonicalMPS(
                tensors=mps_can.tensors,
                orth_centre=mps_can.num_sites - 1,
                tolerance=mps_can.tolerance,
                chi_max=mps_can.chi_max,
            ),
        )
