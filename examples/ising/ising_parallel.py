"""
Here we parallelise our ground-state example. Co-authored by Moïse Russeau.
"""

from multiprocessing import cpu_count, Pool
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.sparse.linalg import eigsh

from ising import IsingExact, IsingMPO
from mdopt.mps.utils import create_simple_product_state
from mdopt.optimiser.dmrg import DMRG as dmrg_optimiser

NUM_SITES = 15


def exact_simulation(magnetic_field: float = 0):
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


def dmrg_simulation(magnetic_field: float = 0):
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
    NUM_SITES = 15

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
    plt.xlim((0.2, 2))
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
    plt.xlim((0.2, 2))
    plt.ylim((0, 1))
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
