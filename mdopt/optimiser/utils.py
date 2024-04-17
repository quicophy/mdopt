"""
This module contains different combinatorial optimisation utilities.

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
        `constraints[0]` to sites 3 and 5, `constraints[1]` to sites 2, 4, 6, etc.

    Exceptions
    ----------
    ValueError
        Empty list of constraints.
    ValueError
        Empty list of sites.
    ValueError
        The ``sites`` list is longer than the ``constraints`` list.
    ValueError
        Non-unique sites in the `sites` list.
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
                f"We have {len(self.constraints)} constraints in the constraints list, "
                f"but {len(self.sites)} constraints assumed by the sites list."
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
