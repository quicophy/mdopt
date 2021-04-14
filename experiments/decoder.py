import numpy as np
from mpopt.contractor.mps import MPS

"""
Parameters
----------
dephased: bool
    Whether to apply dephasing channel to the MPO, i.e.,
    rho' = 0.5 * (rho + Z*rho*Z).

if dephased:
            pauli_z = np.diag([1.0, -1.0])
            mpo = self.dephase_mpo(0.5, pauli_z)
"""


def ferro_mps(nsites, dim):
    """
    Return a ferromagnetic MPS (a product state with all spins up).

    Arguments:
        nsites: int
            Number of sites of the MPS.
        dim: int
            Dimensionality of a local Hilbert space at each site.

    Returns:
        MPS: an instance of the MPS claschmidt_values
    """

    tensor = np.zeros([1, dim, 1], np.float64)
    tensor[0, 0, 0] = 1.0
    schmidt_value = np.ones([1], np.float64)
    tensors = [tensor.copy() for i in range(nsites)]
    schmidt_values = [schmidt_value.copy() for i in range(nsites)]
    return MPS(tensors, schmidt_values)
