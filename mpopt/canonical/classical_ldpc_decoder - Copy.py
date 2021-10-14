import numpy as np
import random
import warnings
import TN_libraries.TNTools as tnt
import os
import qecstruct as qs
import qeclab as ql

'''
Classical LDPC decoder using TNTools MPS-MPO formalism. 

'''


def rand_bin_array(K, N):
    arr = np.zeros(N, dtype=int)
    arr[:K] = 1
    random.shuffle(arr)
    return arr


'''
Theses are functions for creating mpo xor checks following TNTools format
'''


def xor_4t2():
    # creating initial 3 legs xor tensor
    reverse_id = np.array([[0., 1.], [1., 0.]])
    xor_3t2 = np.zeros((2, 2, 2))
    xor_3t2[0, :, :] = reverse_id
    xor_3t2[1, :, :] = np.identity(2)

    # Creating initial delta tensor
    delta = np.zeros((2, 2, 2))
    delta[0, :, :] = np.array([[1., 0.], [0., 0.]])
    delta[1, :, :] = np.array([[0., 0.], [0., 1.]])

    # contracting the two to generate generic xor MPO tensor in correct format
    tens = np.tensordot(delta, xor_3t2, axes=([2], [0]))

    return tens


def parity_TN(check, par_tens=xor_4t2(), adjacency=False):
    # Creating simple cross tensor
    cross = np.tensordot(np.identity(2), np.identity(2), axes=0)
    # adapting receiving of input, adjacency matrix column or check positions
    if adjacency == True:
        nonzeros = np.nonzero(check)[1]
    else:
        nonzeros = np.array([c for c in check])

    # init list and tens dimensions
    mpo = []
    tens_shape = par_tens.shape

    # We save the first element of the MPO, index fixed linked to #tens. parity
    if nonzeros.size % 2 == 0:
        mpo.append(par_tens[:, :, 0, :].reshape(
            (tens_shape[0], tens_shape[1], 1, tens_shape[3])))
    else:
        mpo.append(par_tens[:, :, 1, :].reshape(
            (tens_shape[0], tens_shape[1], 1, tens_shape[3])))

    # we assign a tensor at each position of MPO with check or not
    for i in range(nonzeros[0]+1, nonzeros[-1]):
        if i in nonzeros:
            mpo.append(par_tens)
        else:
            mpo.append(cross)
    # We save the last tensor
    mpo.append(par_tens[:, :, :, 0].reshape(
        (tens_shape[0], tens_shape[1], tens_shape[2], 1)))

    # We return the list of tensors, the positions of the mps that are assigned
    # to the first and last mpo tensors.

    return mpo, nonzeros[0], nonzeros[-1]


def classical_ldpc_decoding(entry, checks, extract='exact', chi_max=False, b_prob=0.1, max_len=False, cut=0.0, err_th=1E-30):
    # Initiating svd function
    def svd_func(_m):
        return tnt.reduced_svd(_m, cut=cut, max_len=max_len, normalize=True, init_norm=True,
                               norm_ord=2, err_th=err_th)

    # Creating the initial mps state class
    entry_mps = tnt.binary_mps_from_sparse(entry)
    state_mps = tnt.MpsStateCanon(entry_mps, orth_pos=None, svd_func=svd_func)
    state_mps.create_orth()  # an orth center is created at the last site by default

    # Inputing boltzmann weight
    boltz_mpo = tnt.boltz_mpo(state_mps.length, prob=b_prob)
    state_mps.mpo_contract(boltz_mpo)

    # Procedure to protect from complete state kill
    try:
        # Filtering loop
        for check in checks.rows():
            c_mpo, begin, _ = parity_TN(check)
            state_mps.mpo_contract(c_mpo, begin)

        # check for nans and infs in array
        NanInf, _ = state_mps.nans_infs_find()
        if NanInf is True:
            warnings.warn(
                'Final mps is invalid (NANs or infs). Try again with less approximative svd function.')
            main_comp = None
        else:
            # Main component extraction
            main_comp = state_mps.main_component(
                method=extract, chi_max=chi_max)
    except ValueError:
        warnings.warn(
            'Total norm elimination of the state. Try again with less approximative svd function.')
        main_comp = None

    return main_comp


'''
def single_test(n=2, error_size=1):
    entry=rand_bin_array(error_size,n*8)
    checks=ldpc_34_gen(n)
    print(classical_ldpc_decoding(entry, checks))
'''


class TN_LDPC_Decoder:
    def __init__(self, code, **kwargs):
        self.checks = code.par_mat()
        self.kwargs = kwargs

    def decode(self, entry):
        output = classical_ldpc_decoding(entry, self.checks)

        return qs.BinaryVector(len(entry), np.nonzero(output)[0])


def new_run_func():
    rng = qs.Rng()
    print(rng)
    #ldpc_code = qs.random_regular_code(12,15,3,4,rng)
    ldpc_code = qs.repetition_code(5)
    noise = qs.BinarySymmetricChannel(0.1)
    decoder = TN_LDPC_Decoder(ldpc_code)

    experiment = ql.LinearDecodingExperiment(ldpc_code, decoder, noise)
    stat = experiment.run_num_times(10, rng)
    df = ql.convert_to_dataframe(experiment, stat)
    print(df)


new_run_func()
