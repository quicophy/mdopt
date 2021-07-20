import sys
sys.path.append("..")
import TNTools as tnt
import numpy as np

'''
A file made to evaluate the efficiency of the different main component finders.
'''



def single_test(size=12,noise=0.5,max_bond=20, basis=2):
    #Assigning a random main component position
    main_comp_vec = np.random.random_integers(0,basis**size)
    ind_main = tnt.integer_to_basis_pos(integer=main_comp_vec, width=size, basis=basis)

    #Creating an mps with that main component pos.
    mps = tnt.random_mps_gen(_n=size,noise_ratio=noise, highest_value_index=main_comp_vec, max_bond=8)

    #Calling the main component method used
    print('main_comp')
    found_main  = tnt.main_component(mps, method='dephased_DMRG', chi_max=max_bond)

    #Checking if the main component found is correct
    result = np.allclose(ind_main, found_main)
    return result



