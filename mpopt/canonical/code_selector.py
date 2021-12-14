import numpy as np
import warnings
import TN_libraries.TNTools as tnt
import qecstruct as qs
import qeclab as ql
import pickle
import json
import sys
import tqdm


'''
This is a code generator for selecting the best LDPC codes for the the decoder
testing. it uses the decoder itself to evaluate the best. Tests are usually done
at a physical error rate of 0.15, as it is a value around threshold for
the 3-4 ldpc code class (the one tested in my thesis)

-Samuel Desrosiers,
23/11/2021

'''


'''
Theses are functions for creating mpo xor checks following TNTools format
'''


def xor_4t2():
    '''
    This function returns the four legged tensor fusion of a 3 variables xor
    tensor and a 3 legged copy tensor. Basic tensor of the decomposition of an
    xor gate into MPO format.  
    '''
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
    '''
    Returns the MPO decomposition of a parity check gates plus the position of
    the first and last tensors in the chain. 
    '''
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
    '''
    Full classical ldpc schedule:
        -entry: received message in np array;
        -checks: list of parity checks positions;
        -decoder_par: dictionnary of the decoder params;
        -svd_function_par: dict. of the svd function params used in the decoder; 
        -main_comp_finder_par: dict of main component finder params.

    '''
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
Decoder class + schedule
'''


class TN_LDPC_Decoder:
    '''
    Decoder class fitting for qeclab/qecstruct format. 
    '''

    def __init__(self, code, decoder_par, svd_function_par, main_comp_finder_par):
        self.checks = code.par_mat()
        self.decoder_par = decoder_par
        self.svd_function_par = svd_function_par
        self.main_comp_finder_par = main_comp_finder_par

    def decode(self, entry):
        #print('running decoder!')
        output = classical_ldpc_decoding(
            entry, self.checks, self.decoder_par, self.svd_function_par, self.main_comp_finder_par)
        #print('Decoder run worked!')
        return qs.BinaryVector(len(entry), np.nonzero(output)[0])


def code_noise_select(bit_degree, check_degree, code_size_mult, phys_err_rt, num_times, rng):
    '''
    Builds a code, noise model, and the number of successive experiences from
    given parameters.
    '''
    ldpc_code = qs.random_regular_code(code_size_mult*check_degree,
                                       code_size_mult*bit_degree, bit_degree, check_degree, rng)

    noise = qs.BinarySymmetricChannel(phys_err_rt)

    return ldpc_code, noise, num_times


def new_run_func(code_selector, decoder_par, svd_function_par, main_comp_finder_par):
    '''
    Run one serie of experience for a set of parameters and returns the required
    parameters results in a dictionnary format.
    '''

    #print('Noise selection and decoder implement.')
    rng = qs.Rng()

    ldpc_code, noise, num_times = code_noise_select(**code_selector, rng=rng)

    # Default case for minimal noise cutting in svd function
    if svd_function_par['err_th'] == 'default':
        # minimum possible probability value
        minimal_val = decoder_par['b_prob']**(
            code_selector['check_degree']*code_selector['code_size_mult'])
        # Going one order below for safety
        svd_function_par['err_th'] = minimal_val/10

    # Default max bond for dephased dmrg is the same as for the whole schedule
    if main_comp_finder_par['chi_max'] == 'default':
        main_comp_finder_par['chi_max'] = svd_function_par['max_len']

    decoder = TN_LDPC_Decoder(
        ldpc_code, decoder_par, svd_function_par, main_comp_finder_par)

    #print('Defining experiment and running it.')

    experiment = ql.LinearDecodingExperiment(ldpc_code, decoder, noise)
    stat = experiment.run_num_times(num_times, rng)

    #print('Succesful experiment results.')

    return stat.failure_rate(), ldpc_code


def best_code(best_of, code_selector, decoder_par, svd_function_par, main_comp_finder_par, max_zeros=5):
    '''
    Finds the best code of few generated randomly. Makes sure the physical error
    rate chosen is not completely off-threshold.
    '''
    best_fail_rate = 1
    best_code = None
    iter = 0

    # Variables for threshold detection.
    fail_rt_legacy = []
    adjust_step = 0.01
    adjust_under = False
    adjust_over = False

    print('Starting loop of codes testing.')

    pb = tqdm.tqdm(total=best_of["best_of"])

    while iter < best_of["best_of"]:
        #print('Generating one code in the loop')
        fail_rate, code = new_run_func(
            code_selector, decoder_par, svd_function_par, main_comp_finder_par)
        #print('One generated code successfully.')
        fail_rt_legacy.append(fail_rate)
        if fail_rate < best_fail_rate:
            best_fail_rate = fail_rate
            best_code = code

        iter += 1
        pb.update(1)

        # Making sure that the phys err rate is not completely below threshold
        if iter == max_zeros and max(fail_rt_legacy) <= 0.1:
            warnings.warn('Error rate below threshold. Adjusting!')
            iter = 0
            pb.update(-1*max_zeros)
            adjust_under = True
            # Adapting adjust step if back and forth
            if adjust_over == True:
                warnings.warn('Reducung step size.')
                adjust_step = adjust_step/10
                adjust_over = False
            code_selector["phys_err_rt"] += adjust_step
            fail_rt_legacy = []

        # Making sure that the phys err rate is not over threshold
        if iter == max_zeros and min(fail_rt_legacy) >= 0.4:
            warnings.warn('Error rate over/too close to threshold. Adjusting!')
            iter = 0
            pb.update(-1*max_zeros)
            adjust_over = True
            # Adapting adjust step if back and forth
            if adjust_under == True:
                warnings.warn('Reducing step size.')
                adjust_step = adjust_step/10
                adjust_under = False
            code_selector["phys_err_rt"] -= adjust_step
            fail_rt_legacy = []

    pb.close()

    return best_fail_rate, code_selector["phys_err_rt"], best_code


def make_codes_frm_file(start_from='code_selector_dict_list.json', send_to='./'):
    '''
    Runs a batch of decoding procedures studies for a set of parameters from a
    list of dictionnaries in a file.
    '''
    # read file
    with open(start_from) as myfile:
        params_list = json.load(myfile)

    # list of parameters studies
    iter = 1
    for param_set in params_list:
        print('[Code '+str(iter)+' out of '+str(len(params_list))+']')
        # run decoding procedure
        fail_rate, phys_rate, selected_code = best_code(**param_set)
        print('Generated code, now saving it!')

        # save code
        decoder_dict = param_set["code_selector"]
        bit_degree = decoder_dict["bit_degree"]
        check_degree = decoder_dict["check_degree"]
        n_mult = decoder_dict["code_size_mult"]
        iter += 1

        # Saving code or comparing ith already existing one.
        try:
            with open(send_to+'/'+str(bit_degree)+'_' + str(check_degree)+'_'+str(n_mult)+'.pkl', "rb") as file:
                data = pickle.load(file)
                legacy_fail_rate = data[0]
                legacy_phys_rate = data[1]
            if legacy_fail_rate <= fail_rate and legacy_phys_rate >= phys_rate:
                print('Already better existing code. Not replacing it.')
            else:
                print('Already existing inferior code. Replacing it.')
                with open(send_to+'/'+str(bit_degree)+'_' + str(check_degree)+'_'+str(n_mult)+'.pkl', "wb") as file:
                    pickle.dump([fail_rate, phys_rate, selected_code], file)
        except:
            print('No already existing code. Creating file.')
            with open(send_to+'/'+str(bit_degree)+'_' + str(check_degree)+'_'+str(n_mult)+'.pkl', "wb") as file:
                pickle.dump([fail_rate, phys_rate, selected_code], file)


if __name__ == "__main__":
    # Calling this file with a repo address with contained parameters sets.
    if len(sys.argv) > 1:
        path = str(sys.argv[1])  # retrieving path given
    else:
        path = 'LDPC_params_results'

    make_codes_frm_file(start_from=path+'/codes_dict_list.json',
                        send_to=path+'/selected_codes/')
