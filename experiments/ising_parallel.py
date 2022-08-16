"""
In this experiment we will use our DMRG optimiser to find the ground state
of an open-bounded transverse field Ising chain. The Hamiltonian reads:
$H = - sum_{i=1}^{N-1} Z_i Z_{i+1} - h * sum_{i=1}^{N} X_i$.
Here, the magnetic field is in the units of the pairwise Z-interaction.
We find the ground state of this Hamiltonian and compute observables.
Beware, this script will take a while to run (because of the eigensolver solving large matrices.)

Similar to the experiment `ising.py` but computed in parallel.
Co-authored by Moïse Rousseau.
"""


import sys
from typing import Union
from multiprocessing import Pool, cpu_count
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh

try:
    from mpopt.mps.explicit import ExplicitMPS
    from mpopt.mps.canonical import CanonicalMPS
    from mpopt.optimiser.dmrg import DMRG as dmrg_optimiser
    from mpopt.contractor.contractor import (
        apply_one_site_operator,
        apply_two_site_unitary,
    )
    from mpopt.mps.utils import inner_product, create_simple_product_state
except ImportError:
    sys.path[0] += "/.."
    from mpopt.mps.explicit import ExplicitMPS
    from mpopt.mps.canonical import CanonicalMPS
    from mpopt.optimiser.dmrg import DMRG as dmrg_optimiser
    from mpopt.contractor.contractor import (
        apply_one_site_operator,
        apply_two_site_unitary,
    )
    from mpopt.mps.utils import inner_product, create_simple_product_state


def compute_one_site_expectation_value(
    mps: Union[CanonicalMPS, ExplicitMPS], operator: np.ndarray, site: np.int32
) -> Union[np.float64, np.complex128]:
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
    mps: Union[CanonicalMPS, ExplicitMPS], unitary: np.ndarray, site: np.int32
) -> np.float64:
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

    def __init__(self, num_sites: np.int32 = 2, h_magnetic: np.float64 = 0):
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

    def energy(self, state: np.ndarray) -> np.float64:
        """
        Computes the energy corresponding to a quantum state `state`.
        """
        return np.conjugate(state.T) @ self.hamiltonian_sparse @ state

    def z_magnetisation(self, i: np.int32, state: np.ndarray) -> np.float64:
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

    def x_magnetisation(self, i: np.int32, state: np.ndarray) -> np.float64:
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

    def average_chain_z_magnetisation(self, state: np.ndarray) -> np.float64:
        """
        Computes the average z-magnetisation
        corresponding to a quantum state `state`
        of the whole chain.
        """
        return (
            sum([self.z_magnetisation(i, state) for i in range(self.num_sites)])
            / self.num_sites
        )

    def average_chain_x_magnetisation(self, state: np.ndarray) -> np.float64:
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
        num_sites :
            Number of spins in the chain.
        h_magnetic :
            Value of the transverse magnetic field scaled by the ZZ-interaction.
    """

    def __init__(self, num_sites: np.int32 = 2, h_magnetic: np.float64 = 0):
        self.num_sites = num_sites
        self.h_magnetic = h_magnetic
        self.identity = np.identity(2)
        self.pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
        if num_sites < 2:
            raise ValueError(f"Number of sites should be >=2, given {num_sites}")

    def hamiltonian_mpo(self) -> list[np.ndarray]:
        """Returns a Matrix Product Operator representation of the Hamiltonian.

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

    def z_magnetisation(
        self, i: np.int32, mps: Union[CanonicalMPS, ExplicitMPS]
    ) -> np.float64:
        """
        Computes the z-magnetisation value
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return compute_one_site_expectation_value(mps, self.pauli_z, i)

    def x_magnetisation(
        self, i: np.int32, mps: Union[CanonicalMPS, ExplicitMPS]
    ) -> np.float64:
        """
        Computes the x-magnetisation value
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return compute_one_site_expectation_value(mps, self.pauli_x, i)

    def average_chain_z_magnetisation(
        self, mps: Union[CanonicalMPS, ExplicitMPS]
    ) -> np.float64:
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
    ) -> np.float64:
        """
        Computes the average x-magnetisation
        corresponding to a quantum state
        in the form of an MPS at site `i`.
        """
        return (
            sum([self.x_magnetisation(i, mps) for i in range(self.num_sites)])
            / self.num_sites
        )


NUM_SITES = 15


def exact_simulation(magnetic_field: np.float64 = 0):
    """
    Exact simulation function.
    """

    exact = IsingExact(num_sites=NUM_SITES, h_magnetic=magnetic_field)
    ham_sp = exact.hamiltonian_sparse()
    ground_state_ex = eigsh(ham_sp, k=2, tol=1e-9)[1][:, 0]
    return (
        exact.average_chain_x_magnetisation(ground_state_ex),
        exact.average_chain_z_magnetisation(ground_state_ex),
    )


def dmrg_simulation(magnetic_field: np.float64 = 0):
    """
    DMRG simulation function.
    """

    mpo = IsingMPO(num_sites=NUM_SITES, h_magnetic=magnetic_field)
    hamiltonian_mpo = mpo.hamiltonian_mpo()
    start_mps = create_simple_product_state(num_sites=NUM_SITES, which="+")
    dmrg_container = dmrg_optimiser(
        start_mps,
        hamiltonian_mpo,
        chi_max=128,
        cut=1e-9,
        mode="SA",
        silent=False,
        copy=True,
    )
    dmrg_container.run(10)
    dmrg_final_mps = dmrg_container.mps
    return (
        mpo.average_chain_x_magnetisation(dmrg_final_mps),
        mpo.average_chain_z_magnetisation(dmrg_final_mps),
    )


if __name__ == "__main__":

    print(
        "__________________________________________________________________________________________"
    )
    print("")
    print(
        "Checking the ground states from exact diagonalisation and DMRG are the same (up to phase):"
    )
    print("")
    NUM_SITES = 15
    NUM_DMRG_RUNS = 10

    ising_exact = IsingExact(num_sites=NUM_SITES, h_magnetic=1.0)
    ising_mpo = IsingMPO(num_sites=NUM_SITES, h_magnetic=1.0)
    ham_mpo = ising_mpo.hamiltonian_mpo()
    ham_sparse = ising_exact.hamiltonian_sparse()

    mps_start = create_simple_product_state(num_sites=NUM_SITES, which="+")

    print("DMRG running:")
    print("")
    engine = dmrg_optimiser(mps_start, ham_mpo, chi_max=64, cut=1e-9, mode="SA")
    engine.run(NUM_DMRG_RUNS)
    print("")
    ground_state_mps = engine.mps
    print("Eigensolver running.")
    ground_state_exact = eigsh(ham_sparse)[1][:, 0]
    print(
        "The ground states are the same:",
        np.isclose(abs(ground_state_mps.dense()), abs(ground_state_exact)).all(),
    )

    print(
        "__________________________________________________________________________________________"
    )
    print("")
    print("Let us compare the magnetisation plots from exact diagonalisation and DMRG.")
    print("The plots should coincide exactly.")
    print("")

    transverse_magnetic_field_space = np.linspace(0.01, 2.0, 20)
    with threadpool_limits(limits=cpu_count(), user_api="blas"):
        with Pool() as pool:
            print("DMRGs running:")
            print("")
            mag_dmrg = pool.map(
                func=dmrg_simulation, iterable=transverse_magnetic_field_space
            )
            print("")
            print("Eigensolvers running.")
            print("")
            mag_exact = pool.map(
                func=exact_simulation, iterable=transverse_magnetic_field_space
            )
            mag_x_exact = [x[0] for x in mag_exact]
            mag_z_exact = [z[1] for z in mag_exact]
            mag_x_dmrg = [x[0] for x in mag_dmrg]
            mag_z_dmrg = [z[1] for z in mag_dmrg]
            pool.close()
            pool.join()

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
