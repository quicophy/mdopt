"""
This module contains classes and functions we use to simulate the 1D Ising module.
"""

from typing import Union
import numpy as np
from scipy.sparse import csr_matrix, eye, kron

from mdopt.mps.explicit import ExplicitMPS
from mdopt.mps.canonical import CanonicalMPS
from mdopt.contractor.contractor import apply_one_site_operator, apply_two_site_unitary
from mdopt.mps.utils import inner_product, create_simple_product_state


def compute_one_site_expectation_value(
    mps: Union[CanonicalMPS, ExplicitMPS], operator: np.ndarray, site: int
) -> Union[np.float32, np.complex128]:
    """Computes a one-site expectation value of an operator (not necessarily unitary)."""

    if site not in range(mps.num_sites):
        raise ValueError(
            f"Site given {site}, with the number of sites in the MPS {mps.num_sites}."
        )

    mps = mps.right_canonical()
    mps_new = mps.copy()

    mps_new.tensors[site] = apply_one_site_operator(
        tensor=mps.single_site_tensor(site), operator=operator
    )

    return inner_product(mps, mps_new)


def compute_two_site_expectation_value(
    mps: Union[CanonicalMPS, ExplicitMPS], unitary: np.ndarray, site: int
) -> np.float32:
    """Computes a two-site expectation value of a unitary
    on the given site and its next neighbour.
    """

    if site not in range(mps.num_sites - 1):
        raise ValueError(
            f"Sites given {site, site + 1} with the number of sites in the MPS {mps.num_sites}."
        )

    mps_old = mps.right_canonical()
    mps_new = mps.copy()

    mps_new.tensors[site], mps_new.tensors[site + 1] = apply_two_site_unitary(
        lambda_0=mps.schmidt_values[site],
        b_1=mps.single_site_right_iso(site),
        b_2=mps.single_site_right_iso(site + 1),
        unitary=unitary,
    )

    return inner_product(mps_old, mps_new)


class IsingExact:
    """Class for exact representation of a transverse field Ising model in 1D.

    Attributes:
        num_sites:
            Number of spins in the chain.
        h_magnetic:
            Value of the transverse magnetic field scaled by the ZZ-interaction.
    """

    def __init__(self, num_sites: int = 2, h_magnetic: np.float32 = 0) -> None:
        self.num_sites = num_sites
        self.h_magnetic = h_magnetic
        self.identity = np.identity(2)
        self.pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
        self.two_qubit_hamiltonian = -kron(
            self.pauli_z, self.pauli_z
        ) - self.h_magnetic * (
            kron(self.pauli_x, self.identity) + kron(self.identity, self.pauli_x)
        )
        if num_sites < 2:
            raise ValueError(f"Number of sites should be >=2, given {num_sites}")

    def hamiltonian_sparse(self) -> csr_matrix:
        """
        Returns a sparse representation of the Hamiltonian.
        """
        if self.num_sites == 2:
            return self.two_qubit_hamiltonian
        ising_recursive = IsingExact(self.num_sites - 1, self.h_magnetic)
        return csr_matrix(
            kron(ising_recursive.hamiltonian_sparse(), self.identity)
            - kron(eye(2 ** (self.num_sites - 2)), kron(self.pauli_z, self.pauli_z))
            - self.h_magnetic * kron(eye(2 ** (self.num_sites - 1)), self.pauli_x)
        )

    def hamiltonian_dense(self) -> np.ndarray:
        """Returns a dense representation of the Hamiltonian.
        Warning: memory-intensive! Do not use for `num_sites` > 20.
        """

        return self.hamiltonian_sparse().todense()

    def energy(self, state: np.ndarray) -> np.float32:
        """
        Computes the energy corresponding to a quantum state `state`.
        """
        return np.conjugate(state.T) @ self.hamiltonian_sparse @ state

    def z_magnetisation(self, i: int, state: np.ndarray) -> np.float32:
        """
        Computes the z-magnetisation value
        corresponding to a quantum state `state`
        at site `i`.
        """
        if i == 0:
            return (
                np.conjugate(state.T)
                @ kron(self.pauli_z, eye(2 ** (self.num_sites - 1))).toarray()
                @ state
            )
        if i == self.num_sites - 1:
            return (
                np.conjugate(state.T)
                @ kron(eye(2 ** (self.num_sites - 1)), self.pauli_z).toarray()
                @ state
            )
        return (
            np.conjugate(state.T)
            @ kron(
                kron(eye(2**i), self.pauli_z), eye(2 ** (self.num_sites - i - 1))
            ).toarray()
            @ state
        )

    def x_magnetisation(self, i: int, state: np.ndarray) -> np.float32:
        """
        Computes the x-magnetisation value
        corresponding to a quantum state `state`
        at site `i`.
        """
        if i == 0:
            return (
                np.conjugate(state.T)
                @ kron(self.pauli_x, eye(2 ** (self.num_sites - 1))).toarray()
                @ state
            )
        if i == self.num_sites - 1:
            return (
                np.conjugate(state.T)
                @ kron(eye(2 ** (self.num_sites - 1)), self.pauli_x).toarray()
                @ state
            )
        return (
            np.conjugate(state.T)
            @ kron(
                kron(eye(2**i), self.pauli_x), eye(2 ** (self.num_sites - i - 1))
            ).toarray()
            @ state
        )

    def average_chain_z_magnetisation(self, state: np.ndarray) -> np.float32:
        """
        Computes the average z-magnetisation
        corresponding to a quantum state `state`
        of the whole chain.
        """
        return (
            sum([self.z_magnetisation(i, state) for i in range(self.num_sites)])
            / self.num_sites
        )

    def average_chain_x_magnetisation(self, state: np.ndarray) -> np.float32:
        """
        Computes the average x-magnetisation
        corresponding to a quantum state `state`
        of the whole chain.
        """
        return (
            sum([self.x_magnetisation(i, state) for i in range(self.num_sites)])
            / self.num_sites
        )


class IsingMPO:
    """
    Class for a Matrix Product Operator (MPO) representation of
    a transverse field Ising model in 1D and computing relevant physical observables.

    Attributes:
        num_sites :int
            Number of spins in the chain.
        h_magnetic : np.float32
            Value of the transverse magnetic field scaled by the ZZ-interaction.
    """

    def __init__(
        self, num_sites: int = int(2), h_magnetic: np.float32 = np.float32(0)
    ) -> None:
        self.num_sites = num_sites
        self.h_magnetic = h_magnetic
        self.identity = np.identity(2)
        self.pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
        if num_sites < 2:
            raise ValueError(f"Number of sites should be >=2, given {num_sites}")

    def hamiltonian_mpo(self) -> list[np.ndarray]:
        """Returns a Matrix Product Operator representation of the Hamiltonian.

        Follows the convention of indices from ``mdopt.mps.explicit.py``:
        each tensor in the MPO list has legs (vL, vR, pU, pD),
        where v stands for "virtual", p -- for "physical",
        and L, R, U, D stand for "left", "right", "up", "down".
        For explanation of what `v_right` and `v_left` are,
        see https://arxiv.org/abs/1603.03039 page 22.
        """
        v_left = np.array([0.0, 0.0, 1.0])
        v_right = np.array([1.0, 0.0, 0.0])

        mpo_bulk = np.zeros((3, 3, 2, 2))
        mpo_bulk[0, 0] = self.identity
        mpo_bulk[1, 0] = self.pauli_z
        mpo_bulk[2, 0] = -self.h_magnetic * self.pauli_x
        mpo_bulk[2, 1] = -1.0 * self.pauli_z
        mpo_bulk[2, 2] = self.identity

        mpo_left = np.tensordot(v_left, mpo_bulk, [0, 0]).reshape((1, 3, 2, 2))
        mpo_right = np.tensordot(mpo_bulk, v_right, [1, 0]).reshape((3, 1, 2, 2))

        mpo_list = []
        mpo_list.append(mpo_left)
        for _ in range(self.num_sites - 2):
            mpo_list.append(mpo_bulk)
        mpo_list.append(mpo_right)

        return mpo_list

    def z_magnetisation(
        self, i: int, mps: Union[CanonicalMPS, ExplicitMPS]
    ) -> np.float32:
        """
        Computes the z-magnetisation value
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return compute_one_site_expectation_value(mps, self.pauli_z, i)

    def x_magnetisation(
        self, i: int, mps: Union[CanonicalMPS, ExplicitMPS]
    ) -> np.float32:
        """
        Computes the x-magnetisation value
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return compute_one_site_expectation_value(mps, self.pauli_x, i)

    def average_chain_z_magnetisation(
        self, mps: Union[CanonicalMPS, ExplicitMPS]
    ) -> np.float32:
        """
        Computes the average z-magnetisation
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return (
            sum([self.z_magnetisation(i, mps) for i in range(self.num_sites)])
            / self.num_sites
        )

    def average_chain_x_magnetisation(
        self, mps: Union[CanonicalMPS, ExplicitMPS]
    ) -> np.float32:
        """
        Computes the average x-magnetisation
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return (
            sum([self.x_magnetisation(i, mps) for i in range(self.num_sites)])
            / self.num_sites
        )
