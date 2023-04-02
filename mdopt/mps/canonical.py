"""
This module contains the :class:`CanonicalMPS` class.
Hereafter, saying the MPS is in a canonical form will mean one of the following.

    1) Right-canonical: all tensors are right isometries, i.e.::

        ---( )---     ---
            |   |       |
            |   |  ==   |
            |   |       |
        ---(*)---     ---

    2) Left-canonical: all tensors are left isometries, i.e.::

        ---( )---     ---
        |   |         |
        |   |    ==   |
        |   |         |
        ---(*)---     ---

    3) Mixed-canonical: all but one tensors are left or right isometries.
    This exceptional tensor will be hereafter called the **orthogonality centre**.

    Note that, in the diagrams, a tensor with a star inside means that it is complex-conjugated.

    The Matrix Product State is stored as a list of three-dimensional tensors.
    Essentially, it corresponds to storing each ``A[i]`` or ``B[i]`` as shown in
    fig.4c in reference `[1]`_.

    .. _[1]: https://arxiv.org/abs/1805.00055
"""

from functools import reduce
from copy import deepcopy
from typing import Optional, Iterable, Tuple, Union, List, cast
import numpy as np
from opt_einsum import contract

import mdopt
from mdopt.utils.utils import kron_tensors, split_two_site_tensor


class CanonicalMPS:
    """
    Class for finite-size canonical matrix product states with open boundary conditions.

    Attributes
    ----------
    tensors : List[np.ndarray]
        The tensors of the MPS, one per each physical site.
        Each tensor has legs (virtual left, physical, virtual right), in short ``(vL, i, vR)``.
    orth_centre : Optional[int]
        Position of the orthogonality centre, does not support negative indexing.
        As a convention, this attribute is taken ``0`` for a right-canonical form,
        ``len(tensors) - 1`` for a left-canonical form, ``None`` for a product state.
    tolerance : np.float32
        Numerical tolerance to zero out the singular values in Singular Value Decomposition.
    chi_max : int
        The maximum bond dimension to keep in Singular Value Decompositions.
    bond_dimensions : List[int]
        The list of all bond dimensions of the MPS.
    phys_dimensions : List[int]
        The list of all physical dimensions of the MPS.
    num_sites : int
        Number of sites.
    num_bonds : int
        Number of bonds.

    Raises
    ------
    ValueError
        If the orthogonality centre position does not correspond to the number of sites.
    ValueError
        If any of the tensors does not have three dimensions.
    """

    def __init__(
        self,
        tensors: List[np.ndarray],
        orth_centre: Optional[int] = None,
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
        self.orth_centre = orth_centre
        self.dtype = tensors[0].dtype
        self.tolerance = tolerance
        self.chi_max = chi_max

        if orth_centre and orth_centre not in range(self.num_sites):
            raise ValueError(
                f"The orthogonality centre index must reside anywhere from site 0 "
                f"to {self.num_sites-1}, the one given is at position {orth_centre}."
            )

        for i, tensor in enumerate(tensors):
            if len(tensor.shape) != 3:
                raise ValueError(
                    "A valid MPS tensor must have 3 legs "
                    f"while the one at site {i} has {len(tensor.shape)}."
                )

    def __len__(self) -> int:
        """
        Returns the number of sites in the MPS.
        """
        return self.num_sites

    def copy(self) -> "CanonicalMPS":
        """
        Returns a copy of the current MPS.
        """
        return CanonicalMPS(
            deepcopy(self.tensors), self.orth_centre, self.tolerance, self.chi_max
        )

    def reverse(self) -> "CanonicalMPS":
        """
        Returns a reversed version of a given MPS.
        """

        reversed_tensors = [np.transpose(tensor) for tensor in reversed(self.tensors)]
        if self.orth_centre:
            reversed_orth_centre = (self.num_sites - 1) - self.orth_centre
            return CanonicalMPS(
                reversed_tensors, reversed_orth_centre, self.tolerance, self.chi_max
            )

        return CanonicalMPS(reversed_tensors, None, self.tolerance, self.chi_max)

    def conjugate(self) -> "CanonicalMPS":
        """
        Returns a complex-conjugated version of the current MPS.
        """

        conjugated_tensors = [np.conjugate(mps_tensor) for mps_tensor in self.tensors]
        return CanonicalMPS(
            conjugated_tensors, self.orth_centre, self.tolerance, self.chi_max
        )

    def one_site_tensor(self, site: int) -> np.ndarray:
        """
        Returs a particular MPS tensor located at the corresponding site.

        Parameters
        ----------
        site : int
            The site index of the tensor.
        """

        if site not in range(self.num_sites):
            raise ValueError(
                f"Site given {site}, with the number of sites in the MPS {self.num_sites}."
            )

        return self.tensors[site]

    def one_site_tensor_iter(self) -> Iterable:
        """
        Returns an iterator over the one-site tensors for every site.
        """
        return (self.one_site_tensor(i) for i in range(self.num_sites))

    def two_site_tensor_next(self, site: int) -> np.ndarray:
        """
        Computes a two-site tensor on a given site and the next one.

        Parameters
        ----------
        site : int
            The site index of the tensor.
        """

        if site not in range(self.num_sites - 1):
            raise ValueError(
                f"Sites given {site}, {site + 1}, "
                f"with the number of sites in the MPS {self.num_sites}."
            )

        return np.tensordot(
            self.one_site_tensor(site),
            self.one_site_tensor(site + 1),
            (2, 0),
        )

    def two_site_tensor_prev(self, site: int) -> np.ndarray:
        """
        Computes a two-site tensor on a given site and the previous one.

        Parameters
        ----------
        site : int
            The site index of the tensor.
        """

        if site not in range(1, self.num_sites):
            raise ValueError(
                f"Sites given {site - 1}, {site}, "
                f"with the number of sites in the MPS {self.num_sites}."
            )

        return np.tensordot(
            self.one_site_tensor(site - 1),
            self.one_site_tensor(site),
            (2, 0),
        )

    def two_site_tensor_next_iter(self) -> Iterable:
        """
        Returns an iterator over the two-site tensors for every site and its right neighbour.
        """
        return (self.two_site_tensor_next(i) for i in range(self.num_sites - 1))

    def two_site_tensor_prev_iter(self) -> Iterable:
        """
        Returns an iterator over the two-site tensors for every site and its left neighbour.
        """
        return (self.two_site_tensor_prev(i) for i in range(1, self.num_sites))

    def dense(self, flatten: bool = True) -> np.ndarray:
        """
        Returns a dense representation of an MPS, given as a list of tensors.

        Warning: this method will cause memory overload for number of sites > ~20!

        Parameters
        ----------
        flatten : bool
            Whether to merge all the physical indices to form a vector or not.
        """

        dense = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), self.tensors)

        if flatten:
            return dense.flatten()

        return np.squeeze(dense)

    def density_mpo(self) -> List[np.ndarray]:
        """
        Returns the MPO representation (as a list of tensors)
        of the density matrix defined by the MPS in a canonical form.
        Each tensor in the MPO list has legs ``(vL, vR, pU, pD)``,
        where ``v`` stands for "virtual", ``p`` -- for "physical",
        and ``L, R, U, D`` stand for "left", "right", "up", "down".

        Notes
        -----
        This operation is depicted in the following diagram::

                   i     j
             a     |     |    c                 i     j
            ...---(*)---(*)---...           ab  |     |  cd
                                    -->  ...---[ ]---[ ]---...
            ...---( )---( )---...               |     |
             b     |     |    d                 k     l
                   k     l

        In the cartoon, ``{i,j,k,l}`` and ``{a,b,c,d}`` are indices.
        Here, the ``( )``'s represent the MPS tensors, the ``[ ]``'s ---the MPO tensors.
        The MPS with the physical legs up is complex-conjugated element-wise.
        The empty line between the MPS and its complex-conjugated version
        stands in fact for the tensor (kronecker) product.

        Warning, this object can be memory-intensive for large bond dimensions!
        """

        mpo = map(
            lambda t: kron_tensors(
                t, t, conjugate_second=True, merge_physicals=False
            ).transpose((0, 3, 2, 1)),
            self.tensors,
        )

        return list(mpo)

    def entanglement_entropy(
        self, tolerance: np.float32 = np.float32(1e-12)
    ) -> np.ndarray:
        """
        Returns the entanglement entropy for bipartitions at each of the bonds.
        """
        return self.explicit(tolerance=tolerance).entanglement_entropy()

    def move_orth_centre(
        self,
        final_pos: int,
        return_singular_values: bool = False,
        renormalise: bool = True,
    ) -> Union["CanonicalMPS", Tuple["CanonicalMPS", List[list]]]:
        """
        Moves the orthogonality centre from its current position to ``final_pos``.

        Returns a new version of the current :class:`CanonicalMPS` instance with
        the orthogonality centre moved from ``self.orth_centre`` to ``final_pos``,
        returns also the singular value tensors from every covered bond as well.

        Parameters
        ----------
        final_pos : int
            Final position of the orthogonality centre.
        return_singular_values : bool
            Whether to return the singular values obtained at each involved bond.

        Raises
        ------
        ValueError
            If ``final_pos`` does not match the MPS length.
        """

        if final_pos not in range(self.num_sites):
            raise ValueError(
                "The final position of the orthogonality centre should be"
                f"from 0 to {self.num_sites-1}, given {final_pos}."
            )

        singular_values = []

        if self.orth_centre is None:
            _, flags_left, flags_right = mdopt.mps.utils.find_orth_centre(  # type: ignore
                self, return_orth_flags=True
            )

            if flags_left in (
                [True] + [False] * (self.num_sites - 1),
                [False] * self.num_sites,
            ):
                if flags_right == [not flag for flag in flags_left]:
                    self.orth_centre = 0

            if flags_left in (
                [True] * (self.num_sites - 1) + [False],
                [True] * self.num_sites,
            ):
                if flags_right == [not flag for flag in flags_left]:
                    self.orth_centre = self.num_sites - 1

            if all(flags_left) and all(flags_right):
                self.orth_centre = 0

        assert self.orth_centre is not None
        if self.orth_centre < final_pos:
            begin, final = self.orth_centre, final_pos
            mps = self.copy()
        elif self.orth_centre > final_pos:
            mps = self.reverse()
            begin = cast(int, mps.orth_centre)
            final = (self.num_sites - 1) - final_pos
        else:
            return self

        for i in range(begin, final):
            two_site_tensor = mps.two_site_tensor_next(i)
            u_l, singular_values_bond, v_r = split_two_site_tensor(
                two_site_tensor,
                chi_max=self.chi_max,
                renormalise=renormalise,
            )
            singular_values.append(singular_values_bond)
            mps.tensors[i] = u_l
            mps.tensors[i + 1] = np.tensordot(
                np.diag(singular_values_bond), v_r, (1, 0)
            )
            mps.orth_centre = i + 1

        if cast(int, self.orth_centre) > final_pos:
            mps = mps.reverse()
            singular_values = list(reversed(singular_values))

        if return_singular_values:
            return mps, singular_values

        return mps

    def move_orth_centre_to_border(self) -> Tuple["CanonicalMPS", str]:
        """
        Moves the orthogonality centre from its current position to the closest border.

        Returns a new version of the current :class:`CanonicalMPS` instance with
        the orthogonality centre moved to the closest (from the current position) border.
        """

        if self.orth_centre is None:
            _, flags_left, flags_right = mdopt.mps.utils.find_orth_centre(  # type: ignore
                self, return_orth_flags=True
            )

            if flags_left in (
                [True] + [False] * (self.num_sites - 1),
                [False] * self.num_sites,
            ):
                if flags_right == [not flag for flag in flags_left]:
                    return self.copy(), "first"

            if flags_left in (
                [True] * (self.num_sites - 1) + [False],
                [True] * self.num_sites,
            ):
                if flags_right == [not flag for flag in flags_left]:
                    return self.copy(), "last"

            if all(flags_left) and all(flags_right):
                return self.copy(), "first"

        else:
            if self.orth_centre <= int(self.num_bonds / 2):
                mps = self.move_orth_centre(final_pos=0, return_singular_values=False)
                return cast("CanonicalMPS", mps), "first"

            mps = self.move_orth_centre(
                final_pos=self.num_sites - 1, return_singular_values=False
            )
        return cast("CanonicalMPS", mps), "last"

    def explicit(
        self, tolerance: np.float32 = np.float32(1e-12)
    ) -> "mdopt.mps.explicit.ExplicitMPS":  # type: ignore
        """
        Transforms a :class:`CanonicalMPS` instance into a :class:`ExplicitMPS` instance.
        Essentially, retrieves each ``Γ[i]`` and ``Λ[i]`` from ``A[i]`` or ``B[i]``.
        See fig.4b in `[1]`_ for reference.
        """

        mps_canonical, border = self.move_orth_centre_to_border()

        if border == "first":
            self.orth_centre = 0
            mps_canonical, singular_values = cast(
                Tuple[CanonicalMPS, List[np.ndarray]],
                self.move_orth_centre(self.num_sites - 1, return_singular_values=True),
            )
        elif border == "last":
            self.orth_centre = self.num_sites - 1
            mps_canonical, singular_values = cast(
                Tuple[CanonicalMPS, List[np.ndarray]],
                self.move_orth_centre(0, return_singular_values=True),
            )

        singular_values.insert(0, np.array([1.0]))
        singular_values.append(np.array([1.0]))
        singular_values = [list(sigma) for sigma in singular_values]  # type: ignore

        explicit_tensors = []
        for i in range(self.num_sites):
            explicit_tensors.append(
                np.tensordot(
                    mps_canonical.tensors[i],
                    np.linalg.inv(np.diag(singular_values[i + 1])),
                    (2, 0),
                )
            )

        return mdopt.mps.explicit.ExplicitMPS(  # type: ignore
            explicit_tensors,
            singular_values,
            tolerance=tolerance,
        )  # type: ignore

    def right_canonical(self) -> "CanonicalMPS":
        """
        Returns the current MPS in the right-canonical form.
        See eq.19 in `[1]`_ for reference.
        """

        return cast(CanonicalMPS, self.move_orth_centre(0))

    def left_canonical(self) -> "CanonicalMPS":
        """
        Returns the current MPS in the left-canonical form.
        See eq.19 in `[1]`_ for reference.
        """

        return cast(CanonicalMPS, self.move_orth_centre(self.num_sites - 1))

    def mixed_canonical(self, orth_centre: int) -> "CanonicalMPS":
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

        return cast(CanonicalMPS, self.move_orth_centre(orth_centre))

    def norm(self) -> np.float32:
        """
        Computes the norm of the current MPS, that is,
        the modulus squared of its inner product with itself.
        """

        return np.float32(abs(mdopt.mps.utils.inner_product(self, self)) ** 2)  # type: ignore

    def one_site_expectation_value(
        self,
        site: int,
        operator: np.ndarray,
    ) -> Union[np.float32, np.complex128]:
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

            ( )---( )---( )---( )
             |     |     |     |
             |    (o)    |     |
             |     |     |     |
            ( )---( )---( )---( )
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

        orthogonality_centre = self.mixed_canonical(site).tensors[site]

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
    ) -> Union[np.float32, np.complex128]:
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

            ( )---( )---( )---( )
             |     |     |     |
             |    (operator)   |
             |     |     |     |
            ( )---( )---( )---( )

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

        mps_mixed = self.mixed_canonical(site)
        two_site_orthogonality_centre = contract(
            "ijk, klm -> ijlm",
            mps_mixed.tensors[site],
            mps_mixed.tensors[site + 1],
            optimize=[(0, 1)],
        )
        del mps_mixed

        return contract(
            "ijkl, jmkn, imnl",
            two_site_orthogonality_centre,
            operator,
            np.conjugate(two_site_orthogonality_centre),
            optimize=[(0, 1), (0, 1)],
        )

    def marginal(
        self, sites_to_marginalise: List[int], canonicalise: bool = False
    ) -> "CanonicalMPS":  # type: ignore
        r"""
        Computes a marginal over a subset of sites of an MPS.
        Attention, this method acts inplace. For the non-inplace version,
        take a look into the ``mps.utils`` module.

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

            ---(0)---(1)---(2)---(3)---    ---(2)---(3)---
                |     |     |     |     ->     |     |
                |     |     |     |            |     |
               (t)   (t)

        Here, the ``(t)`` (trace) tensor is a tensor consisting af all 1's.
        """

        sites_all = list(range(self.num_sites))

        if sites_to_marginalise == []:
            return self

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
            phys_dim = self.phys_dimensions[site]
            trace_tensor = np.ones((phys_dim,)) / np.sqrt(phys_dim)
            self.tensors[site] = np.tensordot(self.tensors[site], trace_tensor, (1, 0))

        bond_dims = self.bond_dimensions

        while sites_to_marginalise:
            try:
                site = int(np.argmax(bond_dims))
            except ValueError:
                site = sites_to_marginalise[0]

            # Checking all possible tensor layouts on a given bond.
            try:
                if self.tensors[site].ndim == 2 and self.tensors[site + 1].ndim == 3:
                    self.tensors[site + 1] = np.tensordot(
                        self.tensors[site], self.tensors[site + 1], (-1, 0)
                    )
                    (
                        self.tensors,
                        self.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                    ) = _update_attributes_routine(
                        self.tensors,
                        self.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                        site,
                    )
                elif self.tensors[site].ndim == 3 and self.tensors[site + 1].ndim == 2:
                    self.tensors[site] = np.tensordot(
                        self.tensors[site], self.tensors[site + 1], (-1, 0)
                    )
                    (
                        self.tensors,
                        self.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                    ) = _update_attributes_routine(
                        self.tensors,
                        self.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                        site + 1,
                    )
                elif self.tensors[site].ndim == 2 and self.tensors[site + 1].ndim == 2:
                    self.tensors[site] = np.tensordot(
                        self.tensors[site], self.tensors[site + 1], (-1, 0)
                    )
                    (
                        self.tensors,
                        self.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                    ) = _update_attributes_routine(
                        self.tensors,
                        self.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                        site + 1,
                    )
            except IndexError:
                if self.tensors[site].ndim == 2 and self.tensors[site - 1].ndim == 3:
                    self.tensors[site - 1] = np.tensordot(
                        self.tensors[site - 1], self.tensors[site], (-1, 0)
                    )
                    (
                        self.tensors,
                        self.num_sites,
                        bond_dims,
                        sites_all,
                        sites_to_marginalise,
                    ) = _update_attributes_routine(
                        self.tensors,
                        self.num_sites,
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
                    tensors=self.tensors,
                    orth_centre=self.num_sites - 1,
                    tolerance=self.tolerance,
                    chi_max=self.chi_max,
                ).move_orth_centre(0, renormalise=False),
            )

        return cast(
            CanonicalMPS,
            CanonicalMPS(
                tensors=self.tensors,
                orth_centre=self.num_sites - 1,
                tolerance=self.tolerance,
                chi_max=self.chi_max,
            ),
        )
