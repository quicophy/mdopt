import numpy as np
import TN_libraries.TNTools as tnt
import time
import tqdm


'''
A file made to evaluate the efficiency of the different main component finders.

TODO:-dmrg method for creating test mps. (Delayed to the future!)
     -different probability distribution testing.
     -add ansatz max bond parameter.
     -Test the validity of the states generated with that ansatz bond parameter.
     -Create a program to find right parameter value to get specific failure rate.
     -Create a program to generate the MPS states to evaluate the methods on. (IMP)


Parameters
    ----------
    mps : MPS tensor list with phantom legs (3 indices per tensor)
    method : - exact : Contracts the mps into a vector, then finds manually
                       the largest element.
             - t_prod_state : Creates a tensor product state approximation of
                              the received mps. Finds the lagest index value of
                              each site.
             - dephased_DMRG : Applies the dephased DMRG method, making an
                               ansatz mps converge to the basis product state
                               corresponding to the largest value element.
             - chi_max (optional) : Only used in dephased DMRG. The maximal
                                    number of singular values kept during the
                                    algorithm. If equal to False, no limit is
                                    applied to the bond dimension. By default,
                                    chi_max=False.

'''


def single_test_dmrg(size=8, max_bond=False, basis=2):

    # Creating a random mps with fixed bond dimension
    mps = tnt.ansatz_mps(mps_length=size, max_chi=max_bond, phys_ind=basis)
    mps = tnt.move_orthog(mps)

    # Calling the main component method used and timing it
    before_dmrg = time.time()

    dmrg_main = tnt.main_component(
        mps, method='dephased_DMRG', chi_max=max_bond, energy_var=0, maxsweep=10)

    after_dmrg = time.time()
    dmrg_time = after_dmrg-before_dmrg

    before_exact = time.time()

    exact_main = tnt.main_component(
        mps, method='exact')

    after_exact = time.time()
    exact_time = after_exact-before_exact

    # Checking if the main component found is correct
    result = np.allclose(exact_main, dmrg_main)

    return result, dmrg_time, exact_time


def single_success_rate_dmrg(size=8, max_bond=False, sample=10):
    failures = []
    dmrg_times = []
    exact_times = []

    pb = tqdm.tqdm(total=sample)
    for _ in range(sample):
        # Run one test
        result, dmrg_time, exact_time = single_test_dmrg(
            size=size, max_bond=max_bond)

        pb.update(1)

        if result:
            # Register NO failure
            failures.append(0)
        else:
            # Register failure
            failures.append(1)

        # Register time
        dmrg_times.append(dmrg_time)
        exact_times.append(exact_time)

    pb.close()

    # Calculate average values for that point.
    avg_dmrg_time = np.mean(dmrg_times)
    std_dmrg_time = np.std(dmrg_times)
    avg_exact_time = np.mean(exact_times)
    std_exact_time = np.std(exact_times)
    fail_rt = np.mean(failures)
    fail_std = np.std(failures)

    return fail_rt, fail_std, avg_dmrg_time, std_dmrg_time, avg_exact_time, std_exact_time,


def save_to_file(parameters, filename, header=False, path='./', buffer=20):

    # Check if result file already exist, and initiate it if not
    try:
        with open(path+filename) as _:
            pass

    except FileNotFoundError:
        if header is False:
            print('The file doesn\'t exist. Creating one for data.')

        else:
            print('File doesn\'t exist, Creating one with given header.')
            header_line = ''
            # enumerate all header elements and convert to a atring line
            for i, param in enumerate(header):
                # If parameter is last of list, do not separate w. comma & buffer
                if i == len(parameters)-1:
                    param_add = str(param)
                else:
                    param_add = str(param)+','

                # Put space for regular column look
                skip = buffer-len(param_add)

                # If value string larger than buffer. Just add full buffer value.
                if skip <= 0:
                    param_add += ' '*buffer
                else:
                    param_add += ' '*skip

                # Add parameter string to the whole lign
                header_line += param_add

            # Print string line on file and close it
            save = open(path+filename, 'a')
            print(header_line, file=save)
            save.close()

    step_line = ''
    # see header printing fo context on top
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


def full_study(sizes=[8], max_bonds=[False], sample=20):
    '''
    Dumb function for parameter study of main component finding methods.
    '''

    # Header list for saving file
    header = ['size', 'max_bond',
              'fail_rate', 'fail_std', 'avg_dmrg_time', 'std_dmrg_time', 'avg_exact_time', 'std_exact_time']

    # Looping over all possible parameters

    for _mb, max_bond in enumerate(max_bonds):
        print('For max. bond of '+str(max_bond) +
              ' ('+str(_mb+1)+'/'+str(len(max_bonds)) + ')')

        for _s, size in enumerate(sizes):
            print('[Run for n='+str(size) +
                  ' bits ('+str(_s+1)+'/' + str(len(sizes))+')]')

            # Value for that point
            f_rate, std_rate, avg_dmrg_time, std_dmrg_time, avg_exact_time, std_exact_time = single_success_rate_dmrg(
                size=size, max_bond=max_bond, sample=sample)
            print(f'Failure rate: {f_rate}')
            print(f'DMRG_time: {avg_dmrg_time}')
            print(f'Exact_time: {avg_exact_time}')

            # Writing parameters list to file
            step_params = [size, max_bond,
                           f_rate, std_rate, avg_dmrg_time, std_dmrg_time, avg_exact_time, std_exact_time]
            save_to_file(
                step_params, 'exact_v_dmrg.dat', path='./main_component_results/', header=header)


sizes = [5, 10, 15, 20, 21, 22, 23, 24, 25]
sample = 10
max_bonds = [4, 6, 8]

full_study(sizes=sizes, max_bonds=max_bonds, sample=sample)
