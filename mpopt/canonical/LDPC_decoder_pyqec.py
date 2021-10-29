import numpy as np
import warnings
import TN_libraries.TNTools as tnt
import qecstruct as qs
import qeclab as ql
import time
import json


'''
Classical LDPC decoder using TNTools MPS-MPO formalism. 

'''


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


def classical_ldpc_decoding(entry, checks, decoder_par, svd_function_par, main_comp_finder_par):
    # Initiating svd function
    def svd_func(_m):
        return tnt.reduced_svd(_m, **svd_function_par)

    # Creating the initial mps state class
    entry_mps = tnt.binary_mps_from_sparse(entry)
    state_mps = tnt.MpsStateCanon(entry_mps, orth_pos=None, svd_func=svd_func)
    state_mps.create_orth()  # an orth center is created at the last site by default

    # Inputing boltzmann weight
    boltz_mpo = tnt.boltz_mpo(state_mps.length, **decoder_par)
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
            main_comp = state_mps.main_component(**main_comp_finder_par)
    except ValueError:
        warnings.warn(
            'Total norm elimination of the state. Try again with less approximative svd function.')
        main_comp = None

    return main_comp


'''
Function for data saving
'''


def save_to_file(parameters, filename, header=False, path='./', buffer=20):

    try:
        with open(path+filename) as _:
            pass

    except FileNotFoundError:
        if header is False:
            print('The file doesn\'t exist. Creating one for data.')

        else:
            print('File doesn\'t exist, Creating one with given header.')
            header_line = ''
            for i, param in enumerate(header):
                if i == len(parameters)-1:
                    param_add = str(param)
                else:
                    param_add = str(param)+','
                skip = buffer-len(param_add)
                if skip <= 0:
                    param_add += ' '*buffer
                else:
                    param_add += ' '*skip
                header_line += param_add
            save = open(path+filename, 'a')
            print(header_line, file=save)
            save.close()

    step_line = ''
    for i, param in enumerate(parameters):
        if i == len(parameters)-1:
            param_add = str(param)
        else:
            param_add = str(param)+','
        skip = buffer-len(param_add)
        if skip <= 0:
            param_add += ' '*buffer
        else:
            param_add += ' '*skip
        step_line += param_add

    save = open(path+filename, 'a')
    print(step_line, file=save)
    save.close()


'''
Decoder class + schedule
'''


class TN_LDPC_Decoder:
    def __init__(self, code, decoder_par, svd_function_par, main_comp_finder_par):
        self.checks = code.par_mat()
        self.decoder_par = decoder_par
        self.svd_function_par = svd_function_par
        self.main_comp_finder_par = main_comp_finder_par

    def decode(self, entry):
        output = classical_ldpc_decoding(
            entry, self.checks, self.decoder_par, self.svd_function_par, self.main_comp_finder_par)
        return qs.BinaryVector(len(entry), np.nonzero(output)[0])


class TimedDecoder:
    def __init__(self, decoder):
        self.decoder = decoder
        self.times = []

    def decode(self, entry):
        before = time.time()
        output = self.decoder.decode(entry)
        after = time.time()

        self.times.append(after-before)

        return output


def code_noise_select(bit_degree, check_degree, code_size_mult, phys_err_rt, num_times, rng):
    ldpc_code = qs.random_regular_code(code_size_mult*check_degree,
                                       code_size_mult*bit_degree, bit_degree, check_degree, rng)

    noise = qs.BinarySymmetricChannel(phys_err_rt)

    return ldpc_code, noise, num_times


def new_run_func(code_selector, decoder_par, svd_function_par, main_comp_finder_par):
    rng = qs.Rng()

    ldpc_code, noise, num_times = code_noise_select(**code_selector, rng=rng)

    decoder = TimedDecoder(TN_LDPC_Decoder(
        ldpc_code, decoder_par, svd_function_par, main_comp_finder_par))

    experiment = ql.LinearDecodingExperiment(ldpc_code, decoder, noise)
    stat = experiment.run_num_times(num_times, rng)

    print(stat.failure_rate())
    print(stat.std())
    print(np.mean(decoder.times))
    print(np.std(decoder.times))


# TODO: - Create new_run_func that includes all params
#      - Data handling and plot savings
#      - json files for parameters? Easier batch throwing.


# read file
with open('decoder_params.json') as myfile:
    params = json.load(myfile)

new_run_func(**params)
