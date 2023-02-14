"""
This module contains different combinatorial optimization utilities.

First, we define the tensors which represent logical operation.
We use the following tensors: IDENTITY, COPY, XOR, SWAP.
See the notes for additional information.

According to our convention, each tensor has legs (vL, vR, pU, pD),
where v stands for "virtual", p -- for "physical",
and L, R, U, D stand for "left", "right", "up", "down".
"""

from typing import List
import numpy as np

IDENTITY = np.eye(2).reshape((1, 1, 2, 2))

COPY_RIGHT = np.fromfunction(
    lambda i, j, k: np.logical_and(i == j, j == k), (2, 2, 2), dtype=int
).reshape((2, 1, 2, 2))

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


class ConstraintString:
    """
    Class for storing a string of logical constraints in the
    Matrix Product Operator format.
    Logical constraints are passed in the form of 4-dimensional MPO tensors.

    Attributes
    ----------
    constraints : list
        A list of logical constraints of which the string consists.
    sites : List[List[int]]
        Each list inside corresponds to a constraint from the `constraints` list,
        and contains the sites to which each constraint is applied.
        For example, [[3, 5], [2, 4, 6], ...] means applying
        `constraints[0]` to sites 3 and 5, `constraints[1]` to sites 2, 4, 6, etc.

    Exceptions
    ----------
    ValueError
        Empty list of constraints.
    ValueError
        Empty list of sites.
    ValueError
        The `sites` list is longer than the `constraints` list.
    ValueError
        Non-unique sites in the `sites` list.
    """

    def __init__(self, constraints: List[np.ndarray], sites: List[List[int]]) -> None:
        self.constraints = constraints
        self.sites = sites

        if self.constraints == []:
            raise ValueError("Empty list of constraints passed.")

        if self.sites == []:
            raise ValueError("Empty list of sites passed.")

        if len(self.sites) > len(self.constraints):
            raise ValueError(
                f"We have {len(self.constraints)} constraints in the constraints list, "
                f"{len(self.sites)} constraints assumed by the sites list."
            )

        seen = set()
        uniq = [site for site in self.flat() if site not in seen and not seen.add(site)]  # type: ignore
        if uniq != self.flat():
            raise ValueError("Non-unique sites encountered in the list.")

        # if self.flat() != [site for site in range(min(self.sites), max(self.sites))]:  # type: ignore
        #    raise ValueError("The string should not have breaks.")

    def __getitem__(self, index: int) -> tuple:
        """
        Returns the pair of a list of sites together with the corresponding MPO.

        Parameters
        ----------
        index : int
            The index of the list of sites.
        """

        return self.sites[index], self.constraints[index]

    def flat(self) -> List[int]:
        """
        Returns a flattened list of sites.
        """

        return [item for sites in self.sites for item in sites]

    def span(self) -> int:
        """
        Returns the span (length) of the constraint string.
        """

        return max(self.flat()) - min(self.flat()) + 1

    def get_mpo(self) -> List[np.ndarray]:
        """
        Returns the constraint string in the MPO format.
        Note, that it will not include identities, which means
        it needs to be manually adjusted to a corresponding MPS,
        as the MPO can be smaller in size.
        """

        mpo = [None for _ in range(self.span())]
        for index, sites_sites in enumerate(self.sites):
            for site in sites_sites:
                mpo[site - min(self.flat())] = self.constraints[index]

        return mpo
