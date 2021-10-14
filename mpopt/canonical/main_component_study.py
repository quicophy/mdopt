import numpy as np
import TN_libraries.TNTools as tnt
import os


'''
A file made to evaluate the efficiency of the different main component finders.

TODO: - Optimize dephase mpo build function
      - More efficient way to build main component test vector
      



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


def single_test(size=8, noise=0.5, max_bond=False, basis=2, method='dephased_DMRG'):
    # Assigning a random main component position
    main_comp_vec = np.random.randint(0, basis**size-1)
    ind_main = tnt.integer_to_basis_pos(
        integer=main_comp_vec, width=size, basis=basis)

    # Creating an mps with that main component pos.
    mps = tnt.random_mps_gen(_n=size, noise_ratio=noise,
                             highest_value_index=main_comp_vec, max_bond=20)

    # Calling the main component method used
    found_main = tnt.main_component(
        mps, method=method, chi_max=max_bond)

    # Checking if the main component found is correct
    result = np.allclose(ind_main, found_main)
    return result


def single_success_rate(size=8, noise=0.5, max_bond=False, basis=2, method='dephased_DMRG', sample=10):
    success = 0
    for i in range(sample):
        if single_test(size=size, noise=noise, max_bond=max_bond, basis=basis, method=method):
            success += 1

    return success/sample


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


def full_study(sizes=[8], noises=[0.5], max_bonds=[False], bases=[2], methods=['t_prod_state'], sample=20):
    header = ['basis', 'method', 'max_bond', 'noise', 'size', 'rate']
    for _b, basis in enumerate(bases):
        for _met, method in enumerate(methods):
            for _s, size in enumerate(sizes):
                for _mb, max_bond in enumerate(max_bonds):
                    for _n, noise in enumerate(noises):
                        print(
                            f'[method={method}, size={size}({_s+1}/{len(sizes)}), max_bond={max_bond}({_mb+1}/{len(max_bonds)}), noise={noise}({_n+1}/{len(noises)})]')
                        rate = single_success_rate(
                            size=size, noise=noise, max_bond=max_bond, basis=basis, method=method, sample=sample)
                        print(f'Success rate: {rate}')
                        step_params = [basis, method,
                                       max_bond, noise, size, rate]
                        save_to_file(
                            step_params, 'data.dat', path='./main_component_results/', header=header)


noises = [0.8, 0.9, 0.99]
sizes = [8, 10, 12, 14]
methods = ['dephased_DMRG']
max_bonds = [8, 10, 12, 14, 16, False]

full_study(noises=noises, sizes=sizes, methods=methods, max_bonds=max_bonds)
