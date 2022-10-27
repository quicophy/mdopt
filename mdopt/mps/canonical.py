"""
This module contains the :class:`CanonicalMPS` class.
Hereafter, by saying the MPS is in a canonical form we will mean one of the following.

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

import mdopt
from mdopt.utils.utils import kron_tensors, split_two_site_tensor


class CanonicalMPS:
    """
    Class for finite-size canonical matrix product states with open boundary conditions.

    Attributes
    ----------
    tensors : list[np.ndarray]
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
    bond_dimensions : list[int]
        The list of all bond dimensions of the MPS.
    phys_dimensions : list[int]
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
        self.bond_dimensions = [self.tensors[i].shape[2] for i in range(self.num_bonds)]
        self.phys_dimensions = [self.tensors[i].shape[1] for i in range(self.num_sites)]
        self.orth_centre = orth_centre
        self.dtype = tensors[0].dtype
        self.tolerance = tolerance
        self.chi_max = chi_max

        if orth_centre and orth_centre not in range(self.num_sites):
            raise ValueError(
                f"The orthogonality centre index must reside anywhere from site 0"
                f"to {self.num_sites-1}, the one given is at position {orth_centre}."
            )

        for _, tensor in enumerate(tensors):
            if len(tensor.shape) != 3:
                raise ValueError(
                    "A valid MPS tensor must have 3 legs"
                    f"while the one given has {len(tensor.shape)}."
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

    def single_site_tensor(self, site: int) -> np.ndarray:
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

    def single_site_tensor_iter(self) -> Iterable:
        """
        Returns an iterator over the single-site tensors for every site.
        """
        return (self.single_site_tensor(i) for i in range(self.num_sites))

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
            self.single_site_tensor(site),
            self.single_site_tensor(site + 1),
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
            self.single_site_tensor(site - 1),
            self.single_site_tensor(site),
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

        return dense

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

    def entanglement_entropy(self) -> np.ndarray:
        """
        Returns the entanglement entropy for bipartitions at each of the bonds.
        """
        return self.explicit().entanglement_entropy()

    def move_orth_centre(
        self, final_pos: int, return_singular_values: bool = False
    ) -> Union["CanonicalMPS", Tuple["CanonicalMPS", List[list]]]:
        """
        Moves the orthogonality centre from its current position to ``final_pos``.

        Returns a new version of the current :class:`CanonicalMPS` instance with
        the orthogonality centre moved from `self.orth_centre` to `final_pos`,
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

            if (flags_left == [True] * self.num_sites) and (
                flags_right == [True] * self.num_sites
            ):
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
                renormalise=True,
            )
            singular_values.append(singular_values_bond)
            mps.tensors[i] = u_l
            mps.tensors[i + 1] = np.tensordot(
                np.diag(singular_values_bond), v_r, (1, 0)
            )

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

    def explicit(self) -> "mdopt.mps.explicit.ExplicitMPS":  # type: ignore
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
            explicit_tensors, cast(List[np.ndarray], singular_values)
        )

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
