"""
In this experiment we will use our DMRG optimiser to find the ground state
of an open-bounded transverse field Ising chain. The Hamiltonian reads:
$H = - sum_{i=1}^{N-1} Z_i Z_{i+1} - h * sum_{i=1}^{N} X_i$.
Here, the magnetic field is in the units of the pairwise Z-interaction.
We find the ground state of this Hamiltonian and compute observables.
The script should be launched from the root of the project directory.

Similar to the experiment ising.py but computed in parallel.
Co-authored by Moïse Rousseau.
"""


import sys
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from opt_einsum import contract
from threadpoolctl import threadpool_limits

sys.path[0] += "/.."

from mpopt.experiments.ising import IsingExact, IsingMPO
from mpopt.mps.explicit import create_simple_product_state
from mpopt.mps.canonical import inner_product
from mpopt.contractor.contractor import apply_one_site_operator, apply_two_site_unitary
from mpopt.optimiser import DMRG as dmrg


class Simulation:
    """
    A container class to deal with parallelisation of multiple DMRG engines.
    """

    def __init__(self, num_sites):
        self.num_sites = num_sites

    def exact_simulation(self, magnetic_field):
        """
        Exact simulation container.
        """

        ising_exact = IsingExact(self.num_sites, magnetic_field)
        ham_exact = ising_exact.hamiltonian_dense()
        np.save("./exact_ham.npy", ham_exact, allow_pickle=False)
        ground_state_exact = eigsh(ham_exact, k=6)[1][:, 0]
        mag_z_exact = ising_exact.average_chain_z_magnetisation(ground_state_exact)
        mag_x_exact = ising_exact.average_chain_x_magnetisation(ground_state_exact)
        return mag_x_exact, mag_z_exact

    def drmg_simulation(self, magnetic_field):
        """
        DMRG simulation container.
        """

        ising_mpo = IsingMPO(self.num_sites, magnetic_field)
        ham_mpo = ising_mpo.hamiltonian_mpo()
        mps_start = create_simple_product_state(self.num_sites, which="0")
        engine = dmrg(mps_start.copy(), ham_mpo, chi_max=64, cut=1e-14, mode="SA")
        engine.run(10)
        ground_state_mps = engine.mps
        mag_z_dmrg = ising_mpo.average_chain_z_magnetisation(ground_state_mps)
        mag_x_dmrg = ising_mpo.average_chain_x_magnetisation(ground_state_mps)
        return mag_x_dmrg, mag_z_dmrg


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
    NUM_SITES = 10

    ising_exact = IsingExact(NUM_SITES, h=1.0)
    ising_mpo = IsingMPO(NUM_SITES, h=1.0)
    ham_mpo = ising_mpo.hamiltonian_mpo()
    ham_exact = ising_exact.hamiltonian_dense()

    mps_start = create_simple_product_state(NUM_SITES, which="0")

    print("DMRG running")
    print("")
    engine = dmrg(mps_start, ham_mpo, chi_max=64, cut=1e-14, mode="SA")
    engine.run(20)
    print("")
    ground_state_mps = engine.mps
    ground_state_exact = eigsh(ham_exact, k=6)[1][:, 0]
    print(np.isclose(abs(ground_state_mps.to_dense()), abs(ground_state_exact)).all())

    print(
        "__________________________________________________________________________________________"
    )
    print("")
    print("Let us compare the magnetisation plots from exact diagonalisation and DMRG")
    print("")

    transverse_magnetic_field_space = np.linspace(0.1, 2.0, 20)
    with threadpool_limits(
        limits=8, user_api="blas"
    ):
        s = Simulation(NUM_SITES)
        with Pool() as pool:
            mag_exact = pool.map(s.exact_simulation, transverse_magnetic_field_space)
            mag_drmg = pool.map(s.drmg_simulation, transverse_magnetic_field_space)
    mag_x_exact = [x[0] for x in mag_exact]
    mag_z_exact = [x[1] for x in mag_exact]
    mag_x_dmrg = [x[0] for x in mag_drmg]
    mag_z_dmrg = [x[1] for x in mag_drmg]

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

