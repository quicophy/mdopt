"""
This module contains the :class:`CanonicalMPS` class.
Hereafter, saying a MPS is in a canonical form will mean one of the following.

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

    Note, that in the diagrams, a tensor with a star inside means that it is complex-conjugated.

    A Matrix Product State is thus stored as a list of three-dimensional tensors
    as shown in the following diagram::

               i    i+1
        ...---( )---( )---...
               |     |
               |     |

    Essentially, this corresponds to storing each ``A[i]`` or ``B[i]`` as shown in
    fig.4c in reference `[1]`_.

    Note, that we enumerate the bonds from the right side of the tensors. For example,
    the bond ``0`` is a bond to the right of tensor ``0``.

    .. _[1]: https://arxiv.org/abs/1805.00055
"""

from functools import reduce
from copy import deepcopy
from typing import Optional, Literal, Iterable, Tuple, Union, List, cast
import numpy as np
from opt_einsum import contract

import mdopt
from mdopt.utils.utils import svd, kron_tensors, split_two_site_tensor


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
    tolerance : float
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
        Number of bonds. Note, that the "ghost" bonds are not included.

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
        tolerance: float = float(1e-12),
        chi_max: int = int(1e4),
    ) -> None:
        self.tensors = tensors
        self.num_sites = len(tensors)
        self.num_bonds = self.num_sites - 1
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
            if tensor.ndim != 3:
                raise ValueError(
                    "A valid MPS tensor must have 3 legs "
                    f"while the one at site {i} has {tensor.ndim}."
                )

    @property
    def bond_dimensions(self) -> List[int]:
        """
        Returns the list of all bond dimensions of the MPS.
        """
        return [tensor.shape[-1] for tensor in self.tensors[:-1]]

    @property
    def phys_dimensions(self) -> List[int]:
        """
        Returns the list of all physical dimensions of the MPS.
        """
        return [self.tensors[i].shape[1] for i in range(self.num_sites)]

    @property
    def all_dimensions(self) -> List[Tuple[int, ...]]:
        """
        Returns the list of all dimensions of the MPS.
        """
        return [self.tensors[i].shape for i in range(self.num_sites)]

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
        Returns a reversed version of the current MPS.
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

    def dense(
        self,
        flatten: bool = True,
        renormalise: bool = False,
        norm: Union[None, float, Literal["fro", "nuc"]] = 2,
    ) -> np.ndarray:
        """
        Returns a dense representation of the MPS.

        Warning: this method can cause memory overload for number of sites > ~20!

        Parameters
        ----------
        flatten : bool
            Whether to merge all the physical indices to form a vector or not.
        renormalise : bool
            Whether to renormalise the resulting tensor.
        norm : Any
            The order of the norm to use while renormalising, see ``numpy.linalg.norm``.
        """

        dense = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), self.tensors)
        dense = dense.squeeze()

        if flatten:
            dense = dense.flatten()

        if renormalise:
            norm = float(np.linalg.norm(dense, ord=norm))
            if norm > 1e-17:
                dense /= norm

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

    def entanglement_entropy(self, tolerance: float = float(1e-12)) -> np.ndarray:
        """
        Returns the entanglement entropy for bipartitions at each of the bonds.
        """
        return self.explicit(tolerance=tolerance).entanglement_entropy()

    def check_orth_centre(self) -> Optional[Union[int, List[int]]]:
        """
        Checks the current position of the orthogonality centre by checking each tensor
        for the isometry conditions.
        Note, this method does not update the current instance's ``orth_centre`` attribute.
        """

        orth_centres, flags_left, flags_right = mdopt.mps.utils.find_orth_centre(  # type: ignore
            self, return_orth_flags=True
        )

        if flags_left in (
            [True] + [False] * (self.num_sites - 1),
            [False] * self.num_sites,
        ):
            if flags_right == [not flag for flag in flags_left]:
                return 0

        if flags_left in (
            [True] * (self.num_sites - 1) + [False],
            [True] * self.num_sites,
        ):
            if flags_right == [not flag for flag in flags_left]:
                return self.num_sites - 1

        if all(flags_left) and all(flags_right):
            return 0

        return orth_centres

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
        renormalise : bool
            Whether to renormalise singular values during each SVD.

        Raises
        ------
        ValueError
            If ``final_pos`` does not match the MPS length.
        ValueError
            If ``self.orth_centre`` is still ``None`` after the search.
        """

        if final_pos not in range(self.num_sites):
            raise ValueError(
                "The final position of the orthogonality centre should be"
                f"from 0 to {self.num_sites-1}, given {final_pos}."
            )

        singular_values = []

        if self.orth_centre is None:
            self.orth_centre = self.check_orth_centre()  # type: ignore

        if self.orth_centre is None:
            raise ValueError("The orthogonality centre value is set to None.")

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
            u_l, singular_values_bond, v_r, _ = split_two_site_tensor(
                two_site_tensor,
                chi_max=self.chi_max,
                renormalise=renormalise,
                strategy="svd",
                return_truncation_error=True,
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

    def move_orth_centre_to_border(
        self, renormalise: bool = True
    ) -> Tuple["CanonicalMPS", str]:
        """
        Moves the orthogonality centre from its current position to the closest border.

        Parameters
        ----------
        renormalise : bool
            Whether to renormalise singular values during each SVD.

        Notes
        -----
        Returns a new version of the current :class:`CanonicalMPS` instance with
        the orthogonality centre moved to the closest (from the current position) border.
        """
        if self.orth_centre is None:
            self.orth_centre = self.check_orth_centre()  # type: ignore
            if self.orth_centre is None:
                raise ValueError("There is no orthogonality center present.")

        elif self.orth_centre == 0:
            return self.copy(), "first"

        elif self.orth_centre == self.num_sites - 1:
            return self.copy(), "last"

        else:
            if (self.orth_centre is not None) and (
                self.orth_centre <= int(self.num_bonds / 2)
            ):
                mps = self.move_orth_centre(
                    final_pos=0,
                    return_singular_values=False,
                    renormalise=renormalise,
                )
                return cast("CanonicalMPS", mps), "first"

            mps = self.move_orth_centre(
                final_pos=self.num_sites - 1,
                return_singular_values=False,
                renormalise=renormalise,
            )

        return cast("CanonicalMPS", mps), "last"

    def explicit(
        self, tolerance: float = float(1e-12), renormalise: bool = True
    ) -> "mdopt.mps.explicit.ExplicitMPS":  # type: ignore
        """
        Transforms a :class:`CanonicalMPS` instance into a :class:`ExplicitMPS` instance.
        Essentially, retrieves each ``Γ[i]`` and ``Λ[i]`` from ``A[i]`` or ``B[i]``.
        See fig.4b in `[1]`_ for reference.

        Parameters
        ----------
        tolerance : float
            Numerical tolerance for the singular values.
        renormalise : bool
            Whether to renormalise singular values during each SVD.
        """

        mps_canonical, border = self.move_orth_centre_to_border(renormalise=renormalise)

        if border == "first":
            self.orth_centre = 0
            mps_canonical, singular_values = cast(
                Tuple[CanonicalMPS, List[np.ndarray]],
                self.move_orth_centre(
                    self.num_sites - 1,
                    return_singular_values=True,
                    renormalise=renormalise,
                ),
            )
        else:
            self.orth_centre = self.num_sites - 1
            mps_canonical, singular_values = cast(
                Tuple[CanonicalMPS, List[np.ndarray]],
                self.move_orth_centre(
                    0,
                    return_singular_values=True,
                    renormalise=renormalise,
                ),
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
            singular_values,  # type: ignore
            tolerance=tolerance,
        )  # type: ignore

    def right_canonical(self, renormalise: bool = True) -> "CanonicalMPS":
        """
        Returns the current MPS in the right-canonical form.
        See eq.19 in `[1]`_ for reference.
        """

        return cast(CanonicalMPS, self.move_orth_centre(0, renormalise=renormalise))

    def left_canonical(self, renormalise: bool = True) -> "CanonicalMPS":
        """
        Returns the current MPS in the left-canonical form.
        See eq.19 in `[1]`_ for reference.
        """

        return cast(
            CanonicalMPS,
            self.move_orth_centre(self.num_sites - 1, renormalise=renormalise),
        )

    def mixed_canonical(
        self, orth_centre: int, renormalise: bool = True
    ) -> "CanonicalMPS":
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

        return cast(
            CanonicalMPS, self.move_orth_centre(orth_centre, renormalise=renormalise)
        )

    def norm(self) -> float:
        """
        Computes the norm of the current MPS, that is,
        the modulus squared of its inner product with itself.
        """

        if self.orth_centre:
            norm = contract(
                "ijk, ijk",
                np.conjugate(self.tensors[self.orth_centre]),
                self.tensors[self.orth_centre],
            )
            return float(abs(norm))

        return float(abs(mdopt.mps.utils.inner_product(self, self)))  # type: ignore

    def one_site_expectation_value(
        self,
        site: int,
        operator: np.ndarray,
    ) -> Union[float, np.complex128]:
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
    ) -> Union[float, np.complex128]:
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

    def compress_bond(
        self,
        bond: int,
        chi_max: int = int(1e4),
        cut: float = float(1e-17),
        renormalise: bool = False,
        strategy: str = "svd",
        return_truncation_error: bool = False,
    ) -> Tuple["CanonicalMPS", Optional[float]]:
        """
        Compresses the bond at a given site, i.e., reduces its bond dimension.

        Parameters
        ----------
        bond : int
            The index of the bond to compress.
        chi_max : int
            The maximum bond dimension to keep.
        cut : float
            Singular values smaller than this will be discarded.
        renormalise: bool
            Whether to renormalise the singular value spectrum after the cut.
        strategy : str
            Which strategy to use for decomposition at the bond.
            Available options: ``svd``, ``qr`` and ``svd_advanced``.
        return_truncation_error : bool
            Whether to return the truncation error.

        Returns
        -------
        compressed_mps: CanonicalMPS
            The new compressed Matrix Product State.
        truncation_error : Optional[float]
            The truncation error.
            Returned only if `return_truncation_error` is set to True.

        Raises
        ------
        ValueError
            If the bond index is out of range.
        ValueError
            If the compression strategy is not supported.

        Notes
        -----
        1) The bonds are being enumerated from the right side of the tensors.
        For example, the bond ``0`` is a bond to the right of tensor ``0``.
        2) The compression scheme ``svd_advanced`` follows
        the scheme outlined in Fig.2 of https://arxiv.org/abs/1708.08932.
        This strategy can give speed ups for large bond dimensions
        by doing two SVDs on moderate-size matrices instead of one SVD on a large matrix.
        """

        if bond not in range(self.num_bonds):
            raise ValueError(
                f"Bond given {bond}, with the number of bonds in the MPS {self.num_bonds}."
            )

        if strategy not in ["svd", "qr", "svd_advanced"]:
            raise ValueError(f"Unsupported compression strategy: {strategy}")

        mps_compressed = self.move_orth_centre(
            final_pos=bond, return_singular_values=False, renormalise=False
        )
        mps_compressed = cast("CanonicalMPS", mps_compressed)

        tensor_left = mps_compressed.tensors[bond]  # the orthogonality centre
        tensor_right = mps_compressed.tensors[bond + 1]

        if strategy == "svd":
            two_site_tensor = contract(
                "ijk, klm -> ijlm", tensor_left, tensor_right, optimize=[(0, 1)]
            )
            u_l, singular_values, v_r, truncation_error = split_two_site_tensor(
                tensor=two_site_tensor,
                chi_max=chi_max,
                cut=cut,
                renormalise=renormalise,
                strategy=strategy,
                return_truncation_error=return_truncation_error,
            )

            mps_compressed.tensors[bond] = contract(
                "ijk, kl -> ijl", u_l, np.diag(singular_values)
            )
            mps_compressed.tensors[bond + 1] = v_r

            if return_truncation_error:
                return mps_compressed, truncation_error

        if strategy == "qr":
            two_site_tensor = contract("ijk, klm -> ijlm", tensor_left, tensor_right)
            q_l, r_r, truncation_error, _ = split_two_site_tensor(
                tensor=two_site_tensor,
                chi_max=chi_max,
                cut=cut,
                renormalise=renormalise,
                strategy=strategy,
                return_truncation_error=return_truncation_error,
            )

            mps_compressed.tensors[bond] = q_l
            mps_compressed.tensors[bond + 1] = r_r

            if return_truncation_error:
                return mps_compressed, truncation_error

        if strategy == "svd_advanced":
            chi_left, phys_left, _ = tensor_left.shape
            _, phys_right, chi_right = tensor_right.shape
            tensor_left = tensor_left.reshape(chi_left * phys_left, -1)
            tensor_right = tensor_right.reshape(-1, phys_right * chi_right)

            iso_left_0, sing_vals_left, iso_left_1, _ = svd(
                mat=tensor_left,
                cut=cut,
                chi_max=chi_max,
                renormalise=renormalise,
                return_truncation_error=return_truncation_error,
            )
            iso_right_0, sing_vals_right, iso_right_1, _ = svd(
                mat=tensor_right,
                cut=cut,
                chi_max=chi_max,
                renormalise=renormalise,
                return_truncation_error=return_truncation_error,
            )

            two_site_tensor = contract(
                "ij, jk, kl, lm -> im",
                np.diag(sing_vals_left),
                iso_left_1,
                iso_right_0,
                np.diag(sing_vals_right),
                optimize=[(0, 1), (0, 1), (0, 1)],
            )

            iso_left_new, sing_vals_new, iso_right_new, truncation_error = svd(
                mat=two_site_tensor,
                cut=cut,
                chi_max=chi_max,
                renormalise=renormalise,
                return_truncation_error=return_truncation_error,
            )

            tensor_left_new = contract(
                "ij, jk, kl -> il",
                iso_left_0,
                iso_left_new,
                np.sqrt(np.diag(sing_vals_new)),
                optimize=[(0, 1), (0, 1)],
            )
            tensor_right_new = contract(
                "ij, jk, kl -> il",
                np.sqrt(np.diag(sing_vals_new)),
                iso_right_new,
                iso_right_1,
                optimize=[(0, 1), (0, 1)],
            )

            mps_compressed.tensors[bond] = tensor_left_new.reshape(
                chi_left, phys_left, len(sing_vals_new)
            )
            mps_compressed.tensors[bond + 1] = tensor_right_new.reshape(
                len(sing_vals_new), phys_right, chi_right
            )

            if return_truncation_error:
                return mps_compressed, truncation_error

        return mps_compressed, None

    def compress(
        self,
        chi_max: int = int(1e4),
        cut: float = float(1e-17),
        renormalise: bool = False,
        strategy: str = "svd",
        return_truncation_errors: bool = False,
    ) -> Tuple["CanonicalMPS", List[Optional[float]]]:
        """
        Compresses the MPS, i.e., runs the ``compress_bond`` method for each bond.

        Parameters
        ----------
        chi_max : int
            The maximum bond dimension to keep.
        cut : float
            Singular values smaller than this will be discarded.
        renormalise: bool
            Whether to renormalise the singular value spectrum after the cut.
        strategy : str
            Which strategy to use for decomposition at the bond.
            Available options: ``svd``, ``qr`` and ``svd_advanced``.
        return_truncation_errors : bool
            Whether to return the list of truncation errors (for each bond).

        Returns
        -------
        compressed_mps: CanonicalMPS
            The new compressed Matrix Product State.
        truncation_errors : Optional[List[float]]
            The truncation errors.
            Returned only if `return_truncation_errors` is set to True.

        Raises
        ------
        ValueError
            If the compression strategy is not supported.
        """

        if strategy not in ["svd", "qr", "svd_advanced"]:
            raise ValueError(f"Unsupported compression strategy: {strategy}")

        truncation_errors = []
        mps_compressed, position = self.move_orth_centre_to_border(renormalise=False)

        if position == "last":
            mps_compressed = mps_compressed.reverse()

        for bond in range(self.num_bonds):
            mps_compressed, truncation_error = mps_compressed.compress_bond(
                bond=bond,
                chi_max=chi_max,
                cut=cut,
                renormalise=renormalise,
                strategy=strategy,
                return_truncation_error=True,
            )
            truncation_errors.append(truncation_error)

        if position == "last":
            mps_compressed = mps_compressed.reverse()
            truncation_errors = truncation_errors[::-1]

        if return_truncation_errors:
            return mps_compressed, truncation_errors

        return mps_compressed, [None]

    def marginal(
        self,
        sites_to_marginalise: List[int],
        renormalise: bool = False,
    ) -> "CanonicalMPS":
        r"""
        Computes a marginal over a subset of sites of an MPS.
        Returns a new :class:`CanonicalMPS` instance.

        Parameters
        ----------
        sites_to_marginalise : List[int]
            The sites to marginalise over.
        renormalise : bool
            Whether to renormalise the resulting MPS.

        Notes
        -----
        Marginalizes (traces out) selected sites from the MPS by first
        contracting them with the trace tensor and then absorbing them.
        The contraction is done in the following way:
        if a traced tensor has a right neighbor, it is absorbed to the right.
        (If it is the last tensor, it is merged with its left neighbor.)

        In the absorption step the physical leg is traced out:
        (vL, i, vR) --> (vL, vR)
        and then the traced tensor is contracted with the neighbor so that
        new_tensor = tensordot(traced, neighbor, axes=([1],[0]))
        yields the expected dimensions.

        An example of marginalising is shown in the following diagram::

            ---(0)---(1)---(2)---(3)---    ---(2)---(3)---
                |     |     |     |     ->     |     |
                |     |     |     |            |     |
               (t)   (t)

        Here, the ``(t)`` (trace) tensor is a tensor consisting af all 1's
        and normalised by the square root of the physical dimension.
        """
        if not sites_to_marginalise:
            return self

        sites_all = list(range(self.num_sites))
        if set(sites_to_marginalise) - set(sites_all):
            raise ValueError(
                "The list of sites to marginalise must be a subset of the list of all sites."
            )

        # Process marginalized sites in descending order to avoid index shifts.
        for site in sorted(sites_to_marginalise, reverse=True):
            phys_dim = self.tensors[site].shape[1]
            trace_tensor = np.ones(phys_dim) / np.sqrt(phys_dim)
            # Trace out the physical legs.
            traced = np.tensordot(self.tensors[site], trace_tensor, axes=([1], [0]))

            if site < self.num_sites - 1:
                # Absorb into the right neighbor.
                new_tensor = np.tensordot(
                    traced, self.tensors[site + 1], axes=([1], [0])
                )
                # Remove the marginalized tensor and its right neighbor, insert the merged tensor.
                del self.tensors[site + 1]
                del self.tensors[site]
                self.tensors.insert(site, new_tensor)
            else:
                # If the marginalized tensor is the last, merge with the left neighbor.
                if site > 0:
                    new_tensor = np.tensordot(
                        self.tensors[site - 1], traced, axes=([-1], [0])
                    )
                    del self.tensors[site]
                    self.tensors[site - 1] = new_tensor
                else:
                    # Only one tensor exists; reshape to have three legs.
                    self.tensors[site] = traced.reshape(traced.shape + (1,))

            self.num_sites = len(self.tensors)
            self.num_bonds = self.num_sites - 1
            self.orth_centre = self.num_sites - 1

            if renormalise:
                orth_centre_norm = np.linalg.norm(self.tensors[self.orth_centre])
                self.tensors[self.orth_centre] /= orth_centre_norm

        return self
