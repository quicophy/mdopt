import TN_libraries.TNTools as tnt
import numpy as np
import sys

'''
A file made to evaluate the efficiency of the different main component finders.

TODO: - printing results in json file



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
                             highest_value_index=main_comp_vec, max_bond=np.Inf)

    # Calling the main component method used
    found_main = tnt.main_component(
        mps, method=method, chi_max=max_bond)

    # Checking if the main component found is correct
    result = np.allclose(ind_main, found_main)
    return result


def single_success_rate(size=8, noise=0.5, max_bond=False, basis=2, method='dephased_DMRG', sample=10):
    success=0
    for i in range(sample):
        if single_test(size=size, noise=noise, max_bond=max_bond, basis=basis, method=method):
            success+=1
    
    return success/sample

def full_study(sizes=[8],noises=[0.5],max_bonds=[False], bases=[2], methods=['dephased_DMRG'], sample=20):
    rates = []
    for _s, size in enumerate(sizes):
        for _n, noise in enumerate(noises):
            for _mb, max_bond in enumerate(max_bonds):
                for _b, basis in enumerate(bases):
                    for _met, method in enumerate(methods):
                        print(f'[Success rate: size={size}, noise={noise}]')
                        rate = single_success_rate(size=size, noise=noise, max_bond=max_bond, basis=basis, method=method, sample=sample)
                        rates.append(rate)
    return rates

noises = [0.95,0.99,0.99999]

rates = full_study(noises=noises)






