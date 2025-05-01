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
where ``L`` is the length of the MPS,
the corresponding singular value tensors at the boundaries
would be the identities of the same dimension.
We index MPS tensors with ``i`` from ``0`` to ``L-1``,
while the singular values matrices are indexed from the left of each tensor, i.e.,
the singular values matrux with index ``i`` is to the left of site ``i``.
Essentially, it corresponds to storing each ``Γ[i]`` and ``Λ[i]`` as shown in
fig.4b in reference `[1]`_.
"""

from functools import reduce
from copy import deepcopy
from typing import Literal, Iterable, List, Union, Optional, Tuple, cast
import numpy as np
from opt_einsum import contract
from scipy.special import xlogy  # pylint: disable=E0611

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
    tolerance : float
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
        tolerance: float = float(1e-12),
        chi_max: int = int(1e4),
    ) -> None:
        self.tensors = tensors
        self.num_sites = len(tensors)
        self.num_bonds = self.num_sites - 1
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
            if tensor.ndim != 3:
                raise ValueError(
                    "A valid MPS tensor must have 3 legs"
                    f"while the one at site {i} has {tensor.ndim}."
                )

        for i, _ in enumerate(singular_values):
            norm = np.linalg.norm(singular_values[i])
            if abs(norm - 1) > tolerance:
                raise ValueError(
                    "The norm of each singular values tensor must be 1, "
                    f"instead the norm is {norm} at bond {i}."
                )

    @property
    def bond_dimensions(self) -> List[int]:
        """
        Returns the list of all bond dimensions of the MPS.
        """
        return [self.tensors[i].shape[-1] for i in range(self.num_bonds)]

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
        Returns a reversed version of the current MPS.
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
        conjugated_sing_vals = cast(List[List], conjugated_sing_vals)
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
        norm : Union[str, int]
            Which norm to use for renormalisation of the final tensor.
        """

        tensors = list(self.one_site_right_iso_iter())
        dense = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), tensors)
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
        of the density matrix defined by a given MPS.
        This operation is depicted in the following picture::

                   i           j
              a    |           |          c              i     j
            ...---(*)---<>*---(*)---<>*---...        ab  |     |  cd
                                              --> ...---[ ]---[ ]---...
            ...---( )---<>----( )---<>----...            |     |
              b    |           |          d              k     l
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

        entropy = np.zeros(shape=(self.num_bonds,), dtype=float)

        for bond in range(self.num_bonds):
            singular_values = self.singular_values[bond].copy()
            singular_values = np.array(singular_values)  # type: ignore
            singular_values[singular_values < self.tolerance] = 0  # type: ignore
            singular_values2 = [singular_value**2 for singular_value in singular_values]
            entropy[bond] = -1 * np.sum(
                np.fromiter((xlogy(s, s) for s in singular_values2), dtype=float)
            )

        return entropy

    def right_canonical(self) -> CanonicalMPS:
        """
        Returns the current MPS in the right-canonical form given the MPS in the explicit form.

        (see eq.19 in https://scipost.org/10.21468/SciPostPhysLectNotes.5 for reference),
        """

        return CanonicalMPS(
            list(self.one_site_right_iso_iter()),
            orth_centre=0,
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
            orth_centre=self.num_sites - 1,
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

    def norm(self) -> float:
        """
        Computes the norm of the current MPS, that is,
        the modulus squared of its inner product with itself.
        """

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

    def compress_bond(
        self,
        bond: int,
        chi_max: int = int(1e4),
        cut: float = float(1e-17),
        renormalise: bool = False,
        return_truncation_error: bool = False,
    ) -> Tuple["ExplicitMPS", Optional[float]]:
        """
        Compresses the bond at a given site, i.e., reduces its bond dimension.
        The compression is performed via trimming the singular values at the bond.

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
        return_truncation_error : bool
            Whether to return the truncation error.

        Returns
        -------
        compressed_mps: ExplicitMPS
            The new compressed Matrix Product State.
        truncation_error : Optional[float]
            The truncation error.
            Returned only if `return_truncation_error` is True.

        Raises
        ------
        ValueError
            If the bond index is out of range.

        Notes
        -----
        The bonds are being enumerated from the right side of the tensors.
        For example, the bond ``0`` is a bond to the right of tensor ``0``.
        Note, the singular value matrices obey a different numbering.
        """

        if bond not in range(self.num_bonds):
            raise ValueError(
                f"Bond given {bond}, with the number of bonds in the MPS {self.num_bonds}."
            )

        tensor_left = self.tensors[bond]
        singular_values = self.singular_values[bond + 1]
        tensor_right = self.tensors[bond + 1]

        max_num = min(chi_max, np.sum(singular_values > cut))  # type: ignore
        singular_values_new = singular_values[:max_num]
        residual_spectrum = singular_values[max_num:]
        truncation_error = np.linalg.norm(residual_spectrum) ** 2

        if renormalise:
            singular_values_new /= np.linalg.norm(singular_values_new)  # type: ignore

        self.tensors[bond] = tensor_left[..., :max_num]
        self.singular_values[bond + 1] = singular_values_new
        self.tensors[bond + 1] = tensor_right[:max_num, ...]

        if return_truncation_error:
            return self, float(truncation_error)

        return self, None

    def compress(
        self,
        chi_max: int = int(1e4),
        cut: float = float(1e-17),
        renormalise: bool = False,
        return_truncation_errors: bool = False,
    ) -> Tuple["ExplicitMPS", List[Optional[float]]]:
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
        return_truncation_errors : bool
            Whether to return the list of truncation errors (for each bond).

        Returns
        -------
        compressed_mps: CanonicalMPS
            The new compressed Matrix Product State.
        truncation_errors : Optional[List[float]]
            The truncation errors.
            Returned only if `return_truncation_errors` is set to True.
        """

        truncation_errors = []
        mps_compressed = self.copy()

        for bond in range(self.num_bonds):
            mps_compressed, truncation_error = mps_compressed.compress_bond(
                bond=bond,
                chi_max=chi_max,
                cut=cut,
                renormalise=renormalise,
                return_truncation_error=True,
            )
            truncation_errors.append(truncation_error)

        if return_truncation_errors:
            return mps_compressed, truncation_errors

        return mps_compressed, [None]

    def marginal(
        self,
        sites_to_marginalise: List[int],
        renormalise: bool = False,
    ) -> Union["ExplicitMPS", "CanonicalMPS"]:  # type: ignore
        r"""
        Computes a marginal over a subset of sites of an MPS.
        This method works via the CanonicalMPS class.
        """

        return self.right_canonical().marginal(
            sites_to_marginalise=sites_to_marginalise, renormalise=renormalise
        )
