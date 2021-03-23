# This code was written by Alex Berezutskii inspired by TenPy in 2020-2021.
# See MPS.py for documentation

import numpy as np
from scipy.linalg import svd
import scipy.sparse.linalg.eigen.arpack as arp


class DMRG:
    """
    DMRG algorithm for a finite-size system, implemented as class holding the necessary data.

    Parameters
    ----------
    mps, model, chi_max, eps:
        See attributes

    mps : MPS
        The current ground-state (approximation).
    mpo : MPO, list of ndarrays[ndim=4]
        The mpo of which the groundstate is to be computed.
    chi_max, eps : float, float
        Truncation parameters.
    LPs, RPs : lists of ndarrays[ndim=3]
        Left and right parts ("environments") of the effective Hamiltonian.
        LPs[i] is the contraction of all parts left of site i in the network <psi|H|psi>,
        and similar RPs[i] for all parts right of site i.
        Each LPs[i] has legs (vL wL* vL*), RPS[i] has legs (vR* wR* vR).
    mode : str, which mode of the eigensolver to use
        Available options:
            'LM' : Largest (in magnitude) eigenvalues.
            'SM' : Smallest (in magnitude) eigenvalues.
            'LA' : Largest (algebraic) eigenvalues.
            'SA' : Smallest (algebraic) eigenvalues.
    """

    def __init__(self, mps, mpo, chi_max, eps, mode):
        assert mps.L == len(mpo)
        self.mps = mps
        self.LPs = [None] * mps.L
        self.RPs = [None] * mps.L
        self.mpo = mpo
        self.chi_max = chi_max
        self.eps = eps
        self.mode = mode
        # initialize left and right environments
        D = self.mpo[0].shape[0]
        chi = mps.Bs[0].shape[0]
        LP = np.zeros([chi, D, chi], dtype=np.float)  # vL wL* vL*
        RP = np.zeros([chi, D, chi], dtype=np.float)  # vR* wR* vR
        LP[:, 0, :] = np.eye(chi)
        RP[:, D - 1, :] = np.eye(chi)
        self.LPs[0] = LP
        self.RPs[-1] = RP
        # initialize necessary RPs
        for i in range(mps.L - 1, 1, -1):
            self.update_RP(i)

    def sweep(self):
        # sweep from left to right
        for i in range(self.mps.nbonds - 1):
            self.update_bond(i)

        # sweep from right to left
        for i in range(self.mps.nbonds - 1, 0, -1):
            self.update_bond(i)

    def update_bond(self, i):
        # Updating the bond
        j = (i + 1) % self.mps.L

        # get the effective Hamiltonian, which will be diagonalized during the update bond step, looks like this:
        #
        #    .--vL*           vR*--.
        #    |       i*    j*      |
        #    |       |     |       |
        #    (LP)---(W1)--(W2)----(RP)
        #    |       |     |       |
        #    |       i     j       |
        #    .--vL             vR--.
        # LP: vL wL* vL*
        # RP: vR* wR* vR
        # W1(mpo[i]): wL wC i i*
        # W2(mpo[j]): wC wR j j*
        Heff_l = np.tensordot(self.LPs[i], self.mpo[i], axes=[1, 0])
        Heff_r = np.tensordot(self.RPs[j], self.mpo[j], axes=[1, 1])
        Heff = np.tensordot(Heff_l, Heff_r, axes=[2, 2])
        chi1, chi2 = self.LPs[i].shape[0], self.RPs[j].shape[2]
        d1, d2 = self.mpo[i].shape[2], self.mpo[j].shape[2]
        Heff = Heff.reshape(chi1 * d1 * d2 * chi2, chi1 * d1 * d2 * chi2)

        # Let's calculate effective single-site wave function on site i in mixed canonical form.
        # The returned array has legs ``vL, i, vR`` (as one of the Bs).
        get_theta1 = np.tensordot(
            np.diag(self.mps.Ss[i]), self.mps.Bs[i], [1, 0]
        )  # vL [vL'], [vL] i vR

        # Let's calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form
        # The returned array has legs ``vL, i, j, vR``.
        get_theta2 = np.tensordot(
            get_theta1, self.mps.Bs[(i + 1) % self.mps.L], [2, 0]
        )  # vL i [vR], [vL] j vR

        # Diagonalize the effective Hamiltonian, find the "ground" state `theta`
        theta0 = np.reshape(get_theta2, [chi1 * d1 * d2 * chi2])  # initial guess
        _, v = arp.eigsh(
            Heff, k=1, which=self.mode, return_eigenvectors=True, v0=theta0
        )
        theta = np.reshape(v[:, 0], (chi1, d1, d2, chi2))

        # Split and truncate a two-site wave function in mixed canonical form.
        # Split a two-site wave function as follows::
        #      vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
        #            |   |                       |             |
        #            i   j                       i             j
        #
        # Afterwards, truncate in the new leg (labeled ``vC``).
        # A : ndarray[ndim=3], left-canonical matrix on site i, with legs ``vL, i, vC``
        # S : ndarray[ndim=1], singular values.
        # B : ndarray[ndim=3], right-canonical matrix on site j, with legs ``vC, j, vR``
        chivL, dL, dR, chivR = theta.shape
        theta = np.reshape(theta, [chivL * dL, dR * chivR])
        X, Y, Z = svd(theta, full_matrices=False)
        # truncate
        chivC = min(self.chi_max, np.sum(Y > self.eps))
        piv = np.argsort(Y)[::-1][:chivC]  # keep the largest `chivC` singular values
        X, Y, Z = X[:, piv], Y[piv], Z[piv, :]
        # renormalise
        Sj = Y / np.linalg.norm(Y)
        # split legs of X and Z
        Ai = np.reshape(X, [chivL, dL, chivC])
        Bj = np.reshape(Z, [chivC, dR, chivR])

        # Put back into MPS
        Gi = np.tensordot(
            np.diag(self.mps.Ss[i] ** (-1)), Ai, axes=[1, 0]
        )  # vL [vL*], [vL] i vC
        self.mps.Bs[i] = np.tensordot(
            Gi, np.diag(Sj), axes=[2, 0]
        )  # vL i [vC], [vC*] vC
        self.mps.Ss[j] = Sj  # vC
        self.mps.Bs[j] = Bj  # vC j vR
        self.update_LP(i)
        self.update_RP(j)

    def update_RP(self, i):
        # Calculate RP right of site `i-1` from RP right of site `i`
        j = (i - 1) % self.mps.L
        RP = self.RPs[i]  # vR* wR* vR
        B = self.mps.Bs[i]  # vL i vR
        Bc = B.conj()  # vL* i* vR*
        W = self.mpo[i]  # wL wR i i*
        RP = np.tensordot(B, RP, axes=[2, 0])  # vL i [vR], [vR*] wR* vR
        RP = np.tensordot(
            RP, W, axes=[[1, 2], [3, 1]]
        )  # vL [i] [wR*] vR, wL [wR] i [i*]
        RP = np.tensordot(
            RP, Bc, axes=[[1, 3], [2, 1]]
        )  # vL [vR] wL [i], vL* [i*] [vR*]
        self.RPs[j] = RP  # vL wL vL* (== vR* wR* vR on site i-1)

    def update_LP(self, i):
        # Calculate LP left of site `i+1` from LP left of site `i`
        j = (i + 1) % self.mps.L
        LP = self.LPs[i]  # vL wL vL*
        B = self.mps.Bs[i]  # vL i vR
        G = np.tensordot(np.diag(self.mps.Ss[i]), B, axes=[1, 0])  # vL [vL*], [vL] i vR
        A = np.tensordot(
            G, np.diag(self.mps.Ss[j] ** -1), axes=[2, 0]
        )  # vL i [vR], [vR*] vR
        Ac = np.conjugate(A)  # vL* i* vR*
        W = self.mpo[i]  # wL wR i i*
        LP = np.tensordot(LP, A, axes=[2, 0])  # vL wL* [vL*], [vL] i vR
        LP = np.tensordot(
            W, LP, axes=[[0, 3], [1, 2]]
        )  # [wL] wR i [i*], vL [wL*] [i] vR
        LP = np.tensordot(
            Ac, LP, axes=[[0, 1], [2, 1]]
        )  # [vL*] [i*] vR*, wR [i] [vL] vR
        self.LPs[j] = LP  # vR* wR vR (== vL wL* vL* on site i+1)
