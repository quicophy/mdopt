"""
This module contains different combinatorial optimisation utilities.

First, we define the tensors which represent logical operation.
We use the following tensors: IDENTITY, COPY, XOR, SWAP.
See the notes for additional information.

According to our convention, each tensor has legs (vL, vR, pU, pD),
where v stands for "virtual", p -- for "physical",
and L, R, U, D stand for "left", "right", "up", "down".
"""

from typing import List, cast
import numpy as np
from tqdm import tqdm
from matrex import msro
from numpy.random import Generator

from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.utils import find_orth_centre
from mdopt.utils.utils import mpo_to_matrix
from mdopt.contractor.contractor import mps_mpo_contract


IDENTITY = np.eye(2).reshape((1, 1, 2, 2))

COPY_LEFT = np.fromfunction(
    lambda i, j, k, l: np.eye(2)[j, l],
    (1, 2, 2, 2),
    dtype=int,
)

XOR_BULK = np.fromfunction(
    lambda i, j, k, l: (i ^ j ^ k ^ 1) * np.eye(2)[k, l],
    (2, 2, 2, 2),
    dtype=int,
)

XOR_LEFT = np.fromfunction(
    lambda i, j, k, l: np.eye(2)[j, k] * np.eye(2)[k, l],
    (1, 2, 2, 2),
    dtype=int,
)

XOR_RIGHT = np.fromfunction(
    lambda i, j, k, l: np.eye(2)[i, k] * np.eye(2)[k, l],
    (2, 1, 2, 2),
    dtype=int,
)

SWAP = np.fromfunction(
    lambda i, j, k, l: np.eye(2)[i, j] * np.eye(2)[k, l],
    (2, 2, 2, 2),
    dtype=int,
)


def parity(bitstring: str, indices: list[int]) -> int:
    """
    Returns the parity of the bits at the given indices in the bitstring.
    """
    return sum(int(bitstring[i]) for i in indices) % 2


def random_constraints(num_bits: int, constraint_size: int, rng: Generator) -> dict:
    """
    Generate random XOR and SWAP site constraints for a bitstring.

    Parameters
    ----------
    num_bits : int
        The total number of bits in the bitstring. Must be at least 3.
    constraint_size : int
        The maximum possible length of the constraint, i.e., the number of XOR sites.
    rng : numpy.random.Generator
        A NumPy random number generator instance for reproducibility.

    Returns
    -------
    dict
        A dictionary containing the following keys:

        - 'xor_left_sites': list of int
            List with one index selected as the left XOR site.
        - 'xor_bulk_sites': list of int
            List of indices selected as bulk XOR sites, between left and right.
        - 'xor_right_sites': list of int
            List with one index selected as the right XOR site.
        - 'swap_sites': list of int
            List of indices between left and right XOR sites, excluding the bulk XOR sites.
        - 'all_constrained_bits': list of int
            Sorted list of all selected XOR site indices (left, bulk, right).

    Raises
    ------
    ValueError
        If `num_bits` is less than 3.
    """
    if num_bits < 3:
        raise ValueError(
            "Bitstring must have at least 3 bits for meaningful XOR and swap sites."
        )

    indices = sorted(rng.choice(num_bits, size=constraint_size, replace=False))

    xor_left_sites = [indices[0]]
    xor_right_sites = [indices[-1]]
    xor_bulk_sites = indices[1:-1]

    swap_sites = [
        i for i in range(indices[0] + 1, indices[-1]) if i not in xor_bulk_sites
    ]

    return {
        "xor_left_sites": xor_left_sites,
        "xor_bulk_sites": xor_bulk_sites,
        "xor_right_sites": xor_right_sites,
        "swap_sites": swap_sites,
        "all_constrained_bits": indices,
    }


class ConstraintString:
    """
    Class for storing a string of logical constraints in the
    Matrix Product Operator format.
    Logical constraints are to be passed in the form of 4-dimensional MPO tensors.

    Attributes
    ----------
    constraints :  List[np.ndarray]
        A list of logical constraints of which the string consists.
    sites : List[List[int]]
        Each list inside corresponds to a constraint from the `constraints` list,
        and contains the sites to which each constraint is applied.
        For example, [[3, 5], [2, 4, 6], ...] means applying
        `constraints[0]` to sites 3 and 5, `constraints[1]` to sites 2, 4, 6 etc.

    Raises
    ------
    ValueError
        - Empty list of constraints.
        - Empty list of sites.
        - The ``sites`` list is longer than the ``constraints`` list.
        - Non-unique sites in the `sites` list.
        - The list of sites has gaps, indicating breaks in the constraints string.
    """

    def __init__(self, constraints: List[np.ndarray], sites: List[List[int]]) -> None:
        self.constraints = [
            np.array(constraint, dtype=float) for constraint in constraints
        ]
        self.sites = sites

        if self.constraints == []:
            raise ValueError("Empty list of constraints passed.")

        if self.sites == []:
            raise ValueError("Empty list of sites passed.")

        if len(self.sites) != len(self.constraints):
            raise ValueError(
                f"We have {len(self.constraints)} constraint(s) in the constraints list, "
                f"but {len(self.sites)} constraint(s) assumed by the sites list."
            )

        seen = set()
        uniq = [
            site for site in self.flat() if site not in seen and not seen.add(site)  # type: ignore
        ]
        if uniq != self.flat():
            raise ValueError("Non-unique sites encountered in the list.")

        if self.flat(sort=True) != list(range(min(self.flat()), max(self.flat()) + 1)):
            raise ValueError("The string should not have breaks in it.")

    def __getitem__(self, index: int) -> tuple:
        """
        Returns the pair of a list of sites together with the corresponding MPO.

        Parameters
        ----------
        index : int
            The index of the list of sites.
        """

        return self.sites[index], self.constraints[index]

    def flat(self, sort: bool = False) -> List[int]:
        """
        Returns a flattened list of sites.

        Parameters
        ----------
        sort : bool
            Whether to sort the flattened list.
        """

        if sort:
            return sorted([item for sites in self.sites for item in sites])

        return [item for sites in self.sites for item in sites]

    def span(self) -> int:
        """
        Returns the span (length) of the constraint string.
        """

        return max(self.flat()) - min(self.flat()) + 1

    def mpo(self) -> List[np.ndarray]:
        """
        Returns an MPO corresponding to the current ``ConstraintString`` instance.
        """

        mpo = [np.array(None) for _ in range(self.span())]
        for index, sites_sites in enumerate(self.sites):
            for site in sites_sites:
                mpo[site - min(self.flat())] = self.constraints[index]

        return mpo


def apply_constraints(
    mps: CanonicalMPS,
    strings: List[List[int]],
    logical_tensors: List[np.ndarray],
    chi_max: int = int(1e4),
    cut: float = float(1e-17),
    renormalise: bool = True,
    strategy: str = "Naive",
    silent: bool = False,
    dense: bool = False,
    return_entropies_and_bond_dims: bool = False,
) -> CanonicalMPS | np.ndarray:
    """
    This function applies logical constraints to an MPS.

    Parameters
    ----------
    mps : CanonicalMPS
        The MPS to which the logical constraints are being applied.
    strings : List[List[int]]
        The list of arguments for :class:`ConstraintString`.
    logical_tensors : List[np.ndarray]
        List of logical tensors for :class:`ConstraintString`.
    chi_max : int
        Maximum bond dimension to keep in the contractor.
    cut : float
        The lower boundary of the spectrum in the contractor.
        All singular values below this value are truncated.
    renormalise : bool
        Whether to renormalise the orthogonality centre after each constraint application.
    result_to_explicit : bool
        Whether to transform the resulting MPS into the Explicit form.
    strategy : str
        The contractor strategy. Available options are "Optimised" and "Naive".
    silent : bool
        Whether to show the progress bar or not.
    return_entropies_and_bond_dims : bool
        Whether to return the entanglement entropies and bond dimensions at each bond.
    dense : bool
        Whether to perform the calculations in the dense form.
        To be used only for small systems (<= 20 sites).

    Returns
    -------
    mps : CanonicalMPS
        The resulting MPS.
    entropies, bond_dims : List[float], List[int]
        The list of entanglement entropies at each bond.
        Returned only if ``return_entropies_and_bond_dims`` is set to ``True``.
    """

    entropies = []
    bond_dims = []

    if strategy == "Optimised":
        # Using matrix front minimization technique to optimise the order
        # in which to apply the checks.
        mpo_location_matrix = np.zeros((len(strings), mps.num_sites))
        for row_idx, sublist in enumerate(strings):
            for subsublist in sublist:
                for index in subsublist:  # type: ignore
                    mpo_location_matrix[row_idx][index] = 1

        optimised_order = msro(mpo_location_matrix)
        strings = [strings[index] for index in optimised_order]

    if dense:
        mps_dense = mps.dense(flatten=True)

    for string in tqdm(strings, disable=silent):
        string = ConstraintString(logical_tensors, string)
        mpo = string.mpo()

        start_site = min(string.flat())
        if dense:
            identities_l = [IDENTITY for _ in range(start_site)]
            identities_r = [IDENTITY for _ in range(len(mps) - len(mpo) - start_site)]
            full_mpo = identities_l + mpo + identities_r
            mpo_dense = mpo_to_matrix(full_mpo, interlace=False, group=True)
            mps_dense = mpo_dense @ mps_dense

        if mps.orth_centre is None:
            orth_centres, flags_left, flags_right = find_orth_centre(
                mps, return_orth_flags=True
            )

            # Managing possible issues with multiple orthogonality centres
            # arising if we do not renormalise while contracting.
            if orth_centres and len(orth_centres) == 1:
                mps.orth_centre = orth_centres[0]
            # Convention.
            if all(flags_left) and all(flags_right):
                mps.orth_centre = 0
            elif flags_left in ([True] + [False] * (mps.num_sites - 1)):
                if flags_right == [not flag for flag in flags_left]:
                    mps.orth_centre = mps.num_sites - 1
            elif flags_left in ([True] * (mps.num_sites - 1) + [False]):
                if flags_right == [not flag for flag in flags_left]:
                    mps.orth_centre = 0
            elif all(flags_right):
                mps.orth_centre = 0
            elif all(flags_left):
                mps.orth_centre = mps.num_sites - 1

            if isinstance(orth_centres, list) and len(orth_centres) > 1:
                mps.orth_centre = orth_centres[0]

            mps = mps.move_orth_centre(
                final_pos=start_site, renormalise=False, return_singular_values=False
            )  # type: ignore

        if not dense:
            mps = mps_mpo_contract(  # type: ignore
                mps=mps,
                mpo=mpo,
                start_site=start_site,
                chi_max=chi_max,
                cut=cut,
                renormalise=False,
                inplace=False,
            )
            if renormalise:
                orth_centre_index = int(mps.orth_centre)  # type: ignore
                norm = np.linalg.norm(mps.tensors[orth_centre_index])
                mps.tensors[orth_centre_index] /= norm

        if return_entropies_and_bond_dims and not dense:
            mps_copy = mps.copy()
            entropies.append(mps_copy.entanglement_entropy())
            bond_dims.append(mps_copy.bond_dimensions)

    if dense:
        return mps_dense

    if return_entropies_and_bond_dims:
        return cast(CanonicalMPS, mps), entropies, bond_dims  # type: ignore

    return cast(CanonicalMPS, mps)
