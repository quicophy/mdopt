import numpy as np
import TN_libraries.TNTools as tnt
import pickle
import sys


'''
A file to generate mps with uniform distributions for testing main_component
finding methods.


TODO:-dmrg method for creating test mps. (Delayed to the future!)
     -different probability distribution testing.
     -add ansatz max bond parameter.
     -Test the validity of the states generated with that ansatz bond parameter.
     -Create a program to find right parameter value to get specific failure rate.
     -Create a program to generate the MPS states to evaluate the methods on. (IMP)

'''


def full_exact_uniform_mps(size=8, noise=0.5, max_bond=False, basis=2):
    '''
    Generates a mps with one defined maxima at given position 
    '''

    # Assigning a random main component position
    main_comp_vec = np.random.randint(0, basis**size-1)
    ind_main = tnt.integer_to_basis_pos(
        integer=main_comp_vec, width=size, basis=basis)

    # Creating an mps with that main component pos.
    mps = tnt.random_mps_gen(_n=size, noise_ratio=noise,
                             highest_value_index=main_comp_vec, max_bond=max_bond)

    return [mps, ind_main]


def full_random_mps(size=8, max_bond=32, basis=2, find_exact_max=False):
    '''
    Generates full random mps and finds exactly main value if asked to.
    '''
    ind_main = False

    mps = tnt.ansatz_mps(mps_length=size, max_chi=max_bond, phys_ind=basis)
    mps = tnt.move_orthog(mps)

    if find_exact_max:
        ind_main = tnt.main_component(mps, method='exact')

    return [mps, ind_main]


def save_to_pickle(list, file_name='mps_collection.pkl', send_to='main_component_results/stored_mps/'):
    '''
    The name says it all...
    '''
    with open(send_to+file_name, 'wb') as file:
        pickle.dump(list, file)


def generate_many_of(size=8, noise=0.5, max_bond=False, many=20, mode='known', job_id='', rand_exact=True):
    mps_list = []
    for _ in range(many):
        if mode == 'known':
            duo = full_exact_uniform_mps(
                size=size, noise=noise, max_bond=max_bond)
        elif mode == 'random':
            duo = full_random_mps(
                size=size, max_bond=max_bond, find_exact_max=rand_exact)
        mps_list.append(duo)

    if mode == 'known':
        p1, _, p2 = str(noise).partition('.')
        noise_str = p1+','+p2
        save_to_pickle(mps_list, file_name=mode+'_' +
                       str(size)+'_'+str(noise)+'_'+job_id+'.pkl')
    elif mode == 'random':
        save_to_pickle(mps_list, file_name=mode+'_' +
                       str(size)+'_'+str(max_bond)+'_'+job_id+'.pkl')


def run_loop(mode, many, sizes, extra_param, job_id, rand_exact=True):
    '''
    Again, the name says it all!
    '''

    for _s, size in enumerate(sizes):
        print('[Run for n='+str(size)+' bits ('+str(_s+1)+'/' +
              str(len(sizes))+'), for '+str(mode)+' generation]')
        for _p, param in enumerate(extra_param):
            print('Param of '+str(param) +
                  ' ('+str(_p+1)+'/'+str(len(extra_param))+')')
            if mode == 'known':
                generate_many_of(size=size, noise=param,
                                 many=many, job_id=job_id, mode=mode)
            elif mode == 'random':
                generate_many_of(size=size, max_bond=param,
                                 many=many, job_id=job_id, mode=mode, rand_exact=rand_exact)


if __name__ == "__main__":
    # Calling this file with a repo address with contained parameters sets.
    if len(sys.argv) > 1:
        job_id = str(sys.argv[1])  # retrieving path given
    else:
        job_id = '_'

    mode = 'known'
    many = 10
    sizes = [3]

    # only if 'known'
    noises = [0.5, 0.9, 0.99, 0.999]
    # Only if 'random'
    max_bonds = [16, 32, 64, 128]
    find_exact_max = True

    print('Passing on the loop.')

    if mode == 'known':
        run_loop(mode=mode, many=many, sizes=sizes,
                 extra_param=noises, job_id=job_id)
    elif mode == 'random':
        run_loop(mode=mode, many=many, sizes=sizes,
                 extra_param=max_bonds, job_id=job_id, rand_exact=find_exact_max)
