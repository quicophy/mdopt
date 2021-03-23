# This code was heavily adopted from TeNPy by Alex Berezutskii in 2020-2021.

import numpy as np
from functools import reduce

class MPS:
    """
    Class for a finite-size matrix product state.

    We index sites with i from 0 to L-1, this means that bond i is left of site i.
    We assume that the state is in right-canonical form.

    Parameters
    ----------
    Bs, Ss:
        Same as attributes.

    Attributes
    ----------
    Bs : list of np.arrays[ndim=3]
        The tensors in right-canonical form, one for each physical site
        (within the unit-cell for an infinite MPS).
        Each B[i] has legs (virtual left, physical, virtual right), in short (vL, i, vR)
    Ss : list of np.arrays[ndim=1]
        The Schmidt values at each of the bonds, Ss[i] is left of Bs[i].
    L : int
        Number of sites (in the unit-cell for an infinite MPS).
    nbonds : int
        Number of (non-trivial) bonds: L-1 for finite boundary conditions
    """
    def __init__(self, Bs, Ss):
        self.Bs = Bs
        self.Ss = Ss
        self.L = len(Bs)
        self.nbonds = self.L - 1
        assert len(Bs) == len(Ss)

    def copy(self):
        return MPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss])

    def get_theta1(self, i):
        """
        Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs (vL, i, vR), as one of the Bs.
        """
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])  # vL [vL'], [vL] i vR

    def get_theta2(self, i):
        """
        Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs (vL, i, j, vR).
        """
        j = (i + 1) % self.L
        return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  # vL i [vR], [vL] j vR

    def get_chi(self):
        """
        Return bond dimensions.
        """
        return [self.Bs[i].shape[2] for i in range(self.nbonds)]

    def site_expectation_value(self, op):
        """
        Calculate expectation values of a local operator at each site.
        """
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = np.tensordot(op, theta, axes=[1, 1])  # i [i*], vL [i] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2], [1, 0, 2]]))
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """
        Calculate expectation values of a local operator at each bond.
        """
        result = []
        for i in range(self.nbonds):
            theta = self.get_theta2(i)  # vL i j vR
            op_theta = np.tensordot(op[i], theta, axes=[[2, 3], [1, 2]])
            # i j [i*] [j*], vL [i] [j] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
        return np.real_if_close(result)

    def entanglement_entropy(self):
        """
        Return the (von Neumann) entanglement entropy for a bipartition at any of the bonds.
        """
        bonds = range(1, self.L)
        result = []
        for i in bonds:
            S = self.Ss[i].copy()
            S[S < 1.e-20] = 0.  # 0*log(0) should give 0; avoid warning or NaN.
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-14
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)
    
    def to_dense(self):
        """
        Return the dense representation of the MPS.
        """
        tensors = [self.get_theta1(i) for i in range(self.L)]
        return reduce(lambda a,b: np.tensordot(a, b, axes=(-1,0)), tensors)
    
    def density_mpo(self, dephased):
        """
        Return the MPO representation of the (dephased) density matrix defined by a given MPS.
        Each tensor in the MPO list has legs (virtual left, physical up, physical down, virtual right), 
        in short (vL, i_u, i_d, vR).
        
        Parameters
        ----------
        dephased: bool
        Whether to apply dephasing channel to the MPO, i.e., rho = 0.5*(rho + Z*rho*Z.T).
        """
    
        # Pauli Z
        Z = np.asarray([[1.,0.],
                        [0.,-1.]])

        mps = [self.get_theta1(i) for i in range(self.L)]

        mpo_tensors_before_dephasing = []
        mpo_tensors_after_dephasing = []

        for i in range(self.L):

            # Tensor product of MPS and dag(MPS)
            tmp = np.kron(mps[i],
                          np.conjugate(mps[i]).T)
            tmp = tmp.reshape((mps[i].shape[0]**2, 
                               mps[i].shape[1], 
                               mps[i].shape[1], 
                               mps[i].shape[2]**2))
            mpo_tensors_before_dephasing.append(tmp)

        # Applying the dephasing channel
        if dephased:
            for i in range(self.L):
                tmp = np.einsum('ab, iacl, cd -> ibdl', 
                                Z, 
                                mpo_tensors_before_dephasing[i],
                                np.conjugate(Z).T)
                mpo_tensors_after_dephasing.append(tmp)

        mpo = []
        if dephased:
            for i in range(self.L):
                f = 0.5 * (mpo_tensors_before_dephasing[i] + mpo_tensors_after_dephasing[i])
                mpo.append(f.reshape((f.shape[0],
                                      f.shape[1],
                                      f.shape[2],
                                      f.shape[3])))
           

        if not dephased:
            for i in range(self.L):
                f = mpo_tensors_before_dephasing[i]
                mpo.append(f.reshape((f.shape[0],
                                      f.shape[1],
                                      f.shape[2],
                                      f.shape[3])))

        return mpo
    
    def mixed_canonical(self):
        """
        Return the MPS in the mixed canonical form.
        """
        return [get_theta1(self, i) for i in range(self.L)]