"""
In this experiment we will use our DMRG optimiser to find the ground state
of an open-bounded transverse field Ising chain. The Hamiltonian reads:
$H = - sum_{i=1}^{N-1} Z_i Z_{i+1} - h * sum_{i=1}^{N} X_i$.
Here, the magnetic field is in the units of the pairwise Z-interaction.
We find the ground state of this Hamiltonian and compute observables.
"""


import sys

import matplotlib.pyplot as plt
import numpy as np
from opt_einsum import contract
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh

sys.path[0] += "/.."

from mpopt.contractor.contractor import apply_one_site_operator, apply_two_site_unitary
from mpopt.mps.canonical import inner_product
from mpopt.mps.explicit import create_simple_product_state
from mpopt.optimiser import DMRG as dmrg


def compute_one_site_expectation_value(mps, unitary, site):
    """
    Computes one-site expectation value for an MPS in the explicit form.
    """
    assert site < len(mps)

    mps_old = mps.to_right_canonical()
    mps_new = mps_old.copy()

    mps_new[site] = apply_one_site_operator(
        t_1=mps.single_site_right_iso(site), operator=unitary
    )

    return inner_product(mps_old, mps_new)


def compute_two_site_expectation_value(mps, unitary, site):
    """
    Computes two-site expectation value for an MPS in the explicit form.
    """
    assert site < len(mps)

    mps_old = mps.to_right_canonical()
    mps_new = mps_old.copy()

    mps_new[site], mps_new[site + 1] = apply_two_site_unitary(
        lambda_0=mps.schmidt_values[site],
        b_1=mps.single_site_right_iso(site),
        b_2=mps.single_site_right_iso(site + 1),
        unitary=unitary,
    )

    return inner_product(mps_old, mps_new)


class IsingExact:
    """
    Class for an exact representation of a transverse field Ising model in 1D
    and computing relevant physical observables.

    Attributes:
        num_sites: int
            Number of spins in the chain.
        h_magnetic: float
            Value of the transverse magnetic field scaled by the ZZ-interaction.
    """

    def __init__(self, num_sites, h_magnetic):
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

    def hamiltonian_sparse(self):
        """
        Returns a `scipy.sparse` representation of the Hamiltonian.
        """
        if self.num_sites == 2:
            return self.two_qubit_hamiltonian
        ising_recursive = IsingExact(self.num_sites - 1, self.h_magnetic)
        return csr_matrix(
            kron(ising_recursive.hamiltonian_sparse(), self.identity)
            - kron(eye(2 ** (self.num_sites - 2)), kron(self.pauli_z, self.pauli_z))
            - self.h_magnetic * kron(eye(2 ** (self.num_sites - 1)), self.pauli_x)
        )

    def hamiltonian_dense(self):
        """
        Returns a dense (`np.array`) representation of the Hamiltonian.
        Warning: memory-intensive! Do not use for `num_sites` > 20.
        """
        return self.hamiltonian_sparse().todense()

    def energy(self, state):
        """
        Computes the energy corresponding to a quantum state `state`.
        """
        return np.conj(state.T) @ self.hamiltonian_sparse @ state

    def z_magnetisation(self, i, state):
        """
        Computes the z-magnetisation value
        corresponding to a quantum state `state`
        at site `i`.
        """
        if i == 0:
            return (
                np.conj(state.T)
                @ kron(self.pauli_z, eye(2 ** (self.num_sites - 1))).toarray()
                @ state
            )
        if i == self.num_sites - 1:
            return (
                np.conj(state.T)
                @ kron(eye(2 ** (self.num_sites - 1)), self.pauli_z).toarray()
                @ state
            )
        return (
            np.conj(state.T)
            @ kron(
                kron(eye(2**i), self.pauli_z), eye(2 ** (self.num_sites - i - 1))
            ).toarray()
            @ state
        )

    def x_magnetisation(self, i, state):
        """
        Computes the x-magnetisation value
        corresponding to a quantum state `state`
        at site `i`.
        """
        if i == 0:
            return (
                np.conj(state.T)
                @ kron(self.pauli_x, eye(2 ** (self.num_sites - 1))).toarray()
                @ state
            )
        if i == self.num_sites - 1:
            return (
                np.conj(state.T)
                @ kron(eye(2 ** (self.num_sites - 1)), self.pauli_x).toarray()
                @ state
            )
        return (
            np.conj(state.T)
            @ kron(
                kron(eye(2**i), self.pauli_x), eye(2 ** (self.num_sites - i - 1))
            ).toarray()
            @ state
        )

    def average_chain_z_magnetisation(self, state):
        """
        Computes the average z-magnetisation
        corresponding to a quantum state `state`
        of the whole chain.
        """
        return (
            sum([self.z_magnetisation(i, state) for i in range(self.num_sites)])
            / self.num_sites
        )

    def average_chain_x_magnetisation(self, state):
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
        num_sites: int
            Number of spins in the chain.
        h_magnetic: float
            Value of the transverse magnetic field scaled by the ZZ-interaction.
    """

    def __init__(self, num_sites, h_magnetic):
        self.num_sites = num_sites
        self.h_magnetic = h_magnetic
        self.identity = np.identity(2)
        self.pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
        if num_sites < 2:
            raise ValueError(f"Number of sites should be >=2, given {num_sites}")

    def hamiltonian_mpo(self):
        """
        Returns a Matrix Product Operator representation of the Hamiltonian.
        Follows the convention of indices from ::module:: `mpopt.mps.explicit.py`:
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

    def z_magnetisation(self, i, mps):
        """
        Computes the z-magnetisation value
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return compute_one_site_expectation_value(mps, self.pauli_z, i)

    def x_magnetisation(self, i, mps):
        """
        Computes the x-magnetisation value
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return compute_one_site_expectation_value(mps, self.pauli_x, i)

    def average_chain_z_magnetisation(self, mps):
        """
        Computes the average z-magnetisation
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return (
            sum([self.z_magnetisation(i, mps) for i in range(self.num_sites)])
            / self.num_sites
        )

    def average_chain_x_magnetisation(self, mps):
        """
        Computes the average x-magnetisation
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return (
            sum([self.x_magnetisation(i, mps) for i in range(self.num_sites)])
            / self.num_sites
        )


if __name__ == "__main__":

    print(
        "__________________________________________________________________________________________"
    )
    print("")
    ising_exact = IsingExact(3, 1.0)
    ising_mpo = IsingMPO(3, 1.0)
    ham_mpo = ising_mpo.hamiltonian_mpo()
    m = contract(
        "zabc, adef, dygh -> begcfh",
        ham_mpo[0],
        ham_mpo[1],
        ham_mpo[2],
        optimize=[(0, 1), (0, 1)],
    ).reshape((8, 8))
    print(
        "Checking the exact and the MPO Hamiltonians being the same: ",
        (ising_exact.hamiltonian_dense() == m).all(),
    )

    print(
        "__________________________________________________________________________________________"
    )
    print("")
    print(
        "Checking the ground states from exact diagonalisation and DMRG being the same (up to a phase): "
    )
    print("")
    NUM_SITES = 20

    ising_exact = IsingExact(NUM_SITES, h_magnetic=1.0)
    ising_mpo = IsingMPO(NUM_SITES, h_magnetic=1.0)
    ham_mpo = ising_mpo.hamiltonian_mpo()
    ham_exact = ising_exact.hamiltonian_sparse()

    mps_start = create_simple_product_state(NUM_SITES, which="0")

    print("DMRG running")
    print("")
    engine = dmrg(mps_start, ham_mpo, chi_max=64, cut=1e-14, mode="SA")
    engine.run(10)
    print("")
    ground_state_mps = engine.mps
    print("Eigensolver running")
    ground_state_exact = eigsh(ham_exact, k=6)[1][:, 0]
    print(np.isclose(abs(ground_state_mps.to_dense()), abs(ground_state_exact)).all())

    print(
        "__________________________________________________________________________________________"
    )
    print("")
    print(
        "Let us compare the magnetisation plots from exact diagonalisation and DMRG (the plots should coincide exactly)"
    )
    print("")

    transverse_magnetic_field_space = np.linspace(0.1, 2.0, 20)
    mag_z_exact = []
    mag_x_exact = []
    mag_z_dmrg = []
    mag_x_dmrg = []
    for magnetic_field in transverse_magnetic_field_space:
        ising_exact = IsingExact(NUM_SITES, magnetic_field)
        ising_mpo = IsingMPO(NUM_SITES, magnetic_field)
        ham_mpo = ising_mpo.hamiltonian_mpo()
        ham_exact = ising_exact.hamiltonian_dense()
        mps_start = create_simple_product_state(NUM_SITES, which="0")
        engine = dmrg(mps_start, ham_mpo, chi_max=64, cut=1e-14, mode="SA")
        engine.run(10)
        ground_state_mps = engine.mps
        ground_state_exact = eigsh(ham_exact, k=6)[1][:, 0]

        mag_z_exact.append(
            ising_exact.average_chain_z_magnetisation(ground_state_exact)
        )
        mag_x_exact.append(
            ising_exact.average_chain_x_magnetisation(ground_state_exact)
        )

        mag_z_dmrg.append(ising_mpo.average_chain_z_magnetisation(ground_state_mps))
        mag_x_dmrg.append(ising_mpo.average_chain_x_magnetisation(ground_state_mps))

    plt.figure(figsize=(9, 4.5))
    plt.plot(transverse_magnetic_field_space, mag_z_exact, label="Exact")
    plt.plot(
        transverse_magnetic_field_space, mag_z_dmrg, label="DMRG", linestyle="dashed"
    )
    plt.xlabel("Transverse magnetic field $h$")
    plt.ylabel("Longitudinal magnetisation $m_z$", rotation=90, labelpad=10)
    plt.xlim((0, 2))
    plt.ylim((-1, 1))
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4.5))
    plt.plot(transverse_magnetic_field_space, mag_x_exact, label="Exact")
    plt.plot(
        transverse_magnetic_field_space, mag_x_dmrg, label="DMRG", linestyle="dashed"
    )
    plt.xlabel("Transverse magnetic field $h$")
    plt.ylabel("Transverse magnetisation $m_x$", rotation=90, labelpad=10)
    plt.xlim((0, 2))
    plt.ylim((0, 1))
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
