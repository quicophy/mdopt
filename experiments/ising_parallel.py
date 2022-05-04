"""
In this experiment we will use our DMRG optimiser to find the ground state
of an open-bounded transverse field Ising chain. The Hamiltonian reads:
$H = - sum_{i=1}^{N-1} Z_i Z_{i+1} - h * sum_{i=1}^{N} X_i$.
Here, the magnetic field is in the units of the pairwise Z-interaction.
We find the ground state of this Hamiltonian and compute observables.

Similar to the experiment `ising.py` but computed in parallel.
Co-authored by Moïse Rousseau.
"""


import sys
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits

sys.path[0] += "/.."

from experiments.ising import IsingExact, IsingMPO
from mpopt.mps.explicit import create_simple_product_state
from mpopt.optimiser import DMRG as dmrg


class Simulation:
    """
    A container class to deal with parallelisation of multiple DMRG entities.
    """

    def __init__(self, num_sites):
        self.num_sites = num_sites

    def exact_simulation(self, magnetic_field):
        """
        Exact simulation container.
        """

        exact = IsingExact(self.num_sites, magnetic_field)
        hamiltonian = exact.hamiltonian_sparse()
        ground_state = eigsh(hamiltonian, k=6)[1][:, 0]
        return ising_exact.average_chain_x_magnetisation(
            ground_state
        ), ising_exact.average_chain_z_magnetisation(ground_state)

    def drmg_simulation(self, magnetic_field):
        """
        DMRG simulation container.
        """

        mpo = IsingMPO(self.num_sites, magnetic_field)
        hamiltonian_mpo = mpo.hamiltonian_mpo()
        start_mps = create_simple_product_state(self.num_sites, which="0")
        dmrg_container = dmrg(
            start_mps,
            hamiltonian_mpo,
            chi_max=64,
            cut=1e-14,
            mode="SA",
            silent=False,
            copy=True,
        )
        dmrg_container.run(10)
        dmrg_final_mps = engine.mps
        return ising_mpo.average_chain_x_magnetisation(
            dmrg_final_mps
        ), ising_mpo.average_chain_z_magnetisation(dmrg_final_mps)


if __name__ == "__main__":

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
    print("Let us compare the magnetisation plots from exact diagonalisation and DMRG")
    print("")

    transverse_magnetic_field_space = np.linspace(0.1, 2.0, 20)
    with threadpool_limits(limits=4, user_api="blas"):
        do = Simulation(NUM_SITES)
        with Pool() as pool:
            mag_exact = pool.map(do.exact_simulation, transverse_magnetic_field_space)
            mag_drmg = pool.map(do.drmg_simulation, transverse_magnetic_field_space)
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
