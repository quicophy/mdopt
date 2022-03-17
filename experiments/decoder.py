import numpy as np
from mpopt.mps import ExplicitMPS

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

# def dephase_mpo(self, prob, tensor_to_apply):
#        """
#        Transforms the density MPO of a given MPS as follows:
#        t = tensor_to_apply
#        rho' = (1-prob) * rho + prob * t*rho*t_dag
#        """
#
#       return dephase_tensor(self.density_mpo(dephased=False), prob, tensor_to_apply)

# def dephase_tensor(tensor, prob, tensor_to_apply):
#    """
#    T = tensor
#    t = tensor_to_apply
#    Returns (1-prob)*T + prob * t*T*t_dag.
#    """
#
#    t_tensor_t_dag = np.einsum("ab, iacl, cd -> ibdl", tensor_to_apply, tensor, dagger(tensor_to_apply))
#    return (1 - prob) * tensor + prob * t_tensor_t_dag


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
    schmidt_values = [schmidt_value.copy() for i in range(nsites + 1)]
    return ExplicitMPS(tensors, schmidt_values)


def antiferro_mps(nsites, dim):
    """
    Return an antiferromagnetic MPS (a product state with all spins down).

    Arguments:
        nsites: int
            Number of sites of the MPS.
        dim: int
            Dimensionality of a local Hilbert space at each site.

    Returns:
        MPS: an instance of the MPS claschmidt_values
    """

    tensor = np.zeros([1, dim, 1], np.float64)
    tensor[0, 1, 0] = 1.0
    schmidt_value = np.ones([1], np.float64)
    tensors = [tensor.copy() for _ in range(nsites)]
    schmidt_values = [schmidt_value.copy() for _ in range(nsites + 1)]
    return ExplicitMPS(tensors, schmidt_values)
