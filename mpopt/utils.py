# Some tensor-network utilities.
# This code was written by Alex Berezutskii inspired by TenPy in 2020-2021.
# See MPS.py for more documentation.

import numpy as np
from .contractor import MPS
from scipy.linalg import svd


# Function courtesy of Samuel Desrosiers.
def trimmed_SVD(
    M,
    cut=1e-12,
    max_num=int(1e6),
    normalize=True,
    init_norm=True,
    norm_ord=2,
    limit_max=False,
    err_th=1e-10,
):
    """
    Return the Singular Value Decomposition of a matrix M.

    Arguments:
        M: ndarray[ndim=2]
            Matrix given as a ndarray with 2 dimensions.
        cut: float
            Norm value cut for lower singular values.
        max_num: int
            Maximum number of singular values to keep.
        normalize: bool
            Activates normalization of the final singular value spectrum.
        init_norm: bool
            Activates the use of relative norm for unormalized tensor's decomposition.
        norm_ord: int
            choose the vector normalization order.
        limit_max: bool
            Activate an upper limit to the spectrum's size.
        err_th: float
            Singular value spectrum norm error.

    Returns:
        U: ndarray
            Unitary matrix having left singular vectors as columns.
        S: ndarray
            The singular values, sorted in non-increasing order.
        Vh: ndarray
            Unitary matrix having right singular vectors as rows.
    """

    try:
        U, S, Vh = svd(M, full_matrices=False, lapack_driver="gesdd")
    except:
        try:
            Vh, S, U = svd(np.transpose(M), full_matrices=False,
                           lapack_driver="gesdd")
            U = np.transpose(U)
            Vh = np.transpose(Vh)
        except:
            U, S, Vh = svd(M, full_matrices=False, lapack_driver="gesvd")

    # Relative norm calculated for cut evaluation
    if init_norm == True:
        norm_S = S / np.linalg.norm(S.reshape(-1, 1), norm_ord)
    else:
        norm_S = S

    norm_sum = 0
    i = 0  # vfor last svd value kept index
    one_norm = 1
    one_norms = []

    # Evaluating final SVD value kept (index), for size limit fixed or not
    if limit_max == True:
        while (
            (norm_sum < (1 - cut))
            and (i < max_num)
            and (i < S.size)
            and (one_norm > err_th)
        ):
            one_norm = np.power(norm_S[i], norm_ord)
            norm_sum += one_norm
            one_norms.append(one_norm)
            i += 1
    else:
        while norm_sum < (1 - cut) and i < S.size and one_norm > err_th:
            one_norm = np.power(norm_S[i], norm_ord)
            norm_sum += one_norm
            one_norms.append(one_norm)
            i += 1

    if normalize == True:
        # Final renormalisation of SVD values kept or not, returning the correct dimensions
        S = S[:i] / np.linalg.norm(S[:i].reshape(-1, 1), norm_ord)
        return U[:, :i], S, Vh[:i, :]
    else:
        return U[:, :i], S[:i], Vh[:i, :]


def MPS_from_dense(psi, d=2, limit_max=False, max_num=100):
    """
    Return the Matrix Product State given a state in the dense (statevector) form.

    Arguments:
        psi: ndarray
            State vector.
        d: int
            Dimensionality of the local Hilbert space, d=2 for qubits.
        limit_max: bool
            Activate an upper limit to the spectrum's size.
        max_num: int
            Maximum number of the singular values to keep.

    Returns:
        MPS(Bs, Ss):
            See the class defined in MPS.py.
    """

    Bs = []
    Ss = []

    # reshape into (..., d)
    psi = psi.reshape(-1, d)

    # Getting the first B and S tensors
    psi, S, Vh = trimmed_SVD(psi, normalize=False,
                             limit_max=limit_max, max_num=max_num)

    # Adding the first B and S tensors to the corresponding lists
    # Note adding the ghost dimension to the first B tensor
    Bs.insert(0, np.expand_dims(Vh, -1))
    Ss.insert(0, S)

    while psi.shape[0] >= d:

        bond_dim = psi.shape[-1]
        psi = psi.reshape(-1, d * bond_dim)
        psi, S, Vh = trimmed_SVD(
            psi, normalize=False, limit_max=limit_max, max_num=max_num
        )
        Vh = Vh.reshape(-1, d, bond_dim)

        # Adding the B and S tensors to the corresponding lists
        Bs.insert(0, Vh)
        Ss.insert(0, S)

    return MPS.MPS(Bs, Ss)


def FM_MPS(L, d):
    """
    Return a ferromagnetic MPS (a product state with all spins up).

    Arguments:
        L: int
            Number of sites of the MPS.
        d: int
            Dimensionality of a local Hilbert space at each site.

    Returns:
        MPS: an instance of the MPS class
    """

    B = np.zeros([1, d, 1], np.float64)
    B[0, 0, 0] = 1.0
    S = np.ones([1], np.float64)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return MPS.MPS(Bs, Ss)


def AFM_MPS(L, d):
    """
    Return an antiferromagnetic MPS (a product state with all spins down).

    Arguments:
        L: int
            Number of sites of the MPS.
        d: int
            Dimensionality of a local Hilbert space at each site.
    Returns:
        MPS: an instance of the MPS class
    """

    B = np.zeros([1, d, 1], np.float64)
    B[0, 1, 0] = 1.0
    S = np.ones([1], np.float64)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return MPS.MPS(Bs, Ss)


def split_truncate_theta(theta, chi_max, eps):
    """
    Split and truncate a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled ``vC``).

    Parameters
    ----------
    theta : ndarray[ndim=4]
        Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
    chi_max : int
        Maximum number of singular values to keep
    eps : float
        Discard any singular values smaller than that.

    Returns
    -------
    A : ndarray[ndim=3]
        Left-canonical matrix on site i, with legs ``vL, i, vC``
    S : ndarray[ndim=1]
        Singular/Schmidt values.
    B : ndarray[ndim=3]
        Right-canonical matrix on site j, with legs ``vC, j, vR``
    """

    chivL, dL, dR, chivR = theta.shape
    theta = np.reshape(theta, [chivL * dL, dR * chivR])
    X, Y, Z = svd(theta, full_matrices=False)
    # truncate
    chivC = min(chi_max, np.sum(Y > eps))
    # keep the largest `chivC` singular values
    piv = np.argsort(Y)[::-1][:chivC]
    X, Y, Z = X[:, piv], Y[piv], Z[piv, :]
    # renormalize
    S = Y / np.linalg.norm(Y)  # == Y/sqrt(sum(Y**2))
    # split legs of X and Z
    A = np.reshape(X, [chivL, dL, chivC])
    B = np.reshape(Z, [chivC, dR, chivR])
    return A, S, B


def to_right_canonical(mps, chi_max, eps):
    """
    Return the right canonical MPS, i.e., B-tensors and S-tensors given the MPS as just a list of tensors with the last one being     the orthogonality centre.

    Arguments:
    ----------
        tensors: list
            A list of tensors, where each tensor has dimenstions
            (virtual left, physical, virtual right), in short (vL, i, vR).
        chi_max: int
            Maximum bond dimension.
        eps: float
            Minimum singular values to keep.
    """

    # checking the "ghost" dimensions for the boundary tensors
    for i in [0, -1]:
        if len(mps.Bs[i].shape) == 2:
            mps.Bs[i] = np.expand_dims(mps.Bs[i], i)  # convention

    ###############################################################
    # reversing the mps so that the orthogonality centre goes first
    # given that it's the last
    # this is not the best solution but at least it works
    mps_Bs = []
    for i in range(mps.L):
        tmp = mps.Bs[-i].transpose((2, 1, 0))
        mps_Bs.append(tmp)
    mps.Bs = mps_Bs
    ###############################################################

    # initialising the initial Ss tensors
    mps.Ss = [np.ones((mps.Bs[i].shape[0]), dtype=float) for i in range(mps.L)]
    mps.Ss[0] = np.diag(mps.Bs[0].squeeze())

    for i in range(0, mps.nbonds, 2):
        j = (i + 1) % mps.L
        theta_2 = mps.get_theta2(i)
        Ai, Sj, Bj = split_truncate_theta(theta_2, chi_max, eps)
        Gi = np.tensordot(mps.Ss[i] ** (-1), Ai, axes=[0, 0])
        mps.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=[1, 0])
        mps.Ss[j] = Sj
        mps.Bs[j] = Bj

    return mps
