'''

 _______ __   _ _______  _____   _____         _______
    |    | \  |    |    |     | |     | |      |______
    |    |  \_|    |    |_____| |_____| |_____ ______|


Collection of Tn functions useful for the simple_canonical form solver.

The functions are made to input fantom legs. The format permits open boundary
conditions to the mps. Doesn't include
optimizers (DMRG/main_component/etc.).

TODO:
    - MPS-MPS contractor (?)
    - MPO-MPO contractor (?)
    - general gate to MPO
    - MPS-block gate contractor
    - Function to cancel canonicalization in class and related modifications
    - function to test if any tensor contains nans, infs or etc.
    - Adjust svd function to include state killing (norm=0):
    - return error in case svd results countains nans, infs, or 0 sing vals

'''

import warnings
import numpy as np
import scipy.linalg
from scipy.sparse.linalg import eigsh


'''
################################################################################
Basic Linear algebra operations
################################################################################
'''


def best_svd(matrix):
    '''    Returns the "best" svd given a possibly failing matrix.
    If failure, try transpose, if again, change lapack driver.
    '''
    try:
        _u, _s, _vh = scipy.linalg.svd(
            matrix, full_matrices=False, lapack_driver='gesdd')
    except scipy.linalg.LinAlgError:
        try:
            _u, _s, _vh = scipy.linalg.svd(np.transpose(
                matrix), full_matrices=False, lapack_driver='gesdd')
            _u = np.transpose(_u)
            _vh = np.transpose(_vh)
        except scipy.linalg.LinAlgError:
            _u, _s, _vh = scipy.linalg.svd(
                matrix, full_matrices=False, lapack_driver='gesvd')

    return _u, _s, _vh


def simple_reduced_svd(matrix, max_len=False, normalize=True, norm_ord=2):
    '''
    This is a simple reduced SVD function.
    normalize activates normalization of final svd spectrum;
    norm_ord choose the vector normalization order;
    max_len is the maximal length of kept sing. vals.
    '''

    _u, _s, _vh = best_svd(matrix)

    # We find the cutoff positon
    final_len = len(_s)
    if max_len is not False:
        final_len = min(final_len, max_len)

    if normalize:
        # Final renormalization of SVD values kept or not, returning the correct
        # matrices sizes
        _s = _s[:final_len]
        _s = _s/np.linalg.norm(_s, ord=norm_ord)
        return _u[:, :final_len], _s, _vh[:final_len, :], final_len
    else:
        return _u[:, :final_len], _s[:final_len], _vh[:final_len, :], final_len


def reduced_svd(matrix, cut=0.0, max_len=False, normalize=False, init_norm=True,
                norm_ord=2, err_th=1E-30):
    '''
    This is a reduced SVD function.
    cut is the norm value cut for lower svd values;
    limit_max activates an upper limit to the spectrum's size;
    normalize activates normalization of final svd spectrum;
    norm_ord choose the vector normalization order;
    init_norm make use of relative norm for unormalized tensor's decomposition.
    '''

    _u, _s, _vh = best_svd(matrix)

    # relative norm calculated for cut evaluation
    if init_norm:
        norm_s = _s / np.linalg.norm(_s, ord=norm_ord)
        norm_s = np.power(norm_s, norm_ord)
    else:
        norm_s = np.power(_s, norm_ord)

    # cumul norm evaluated
    norm_cumsum = np.cumsum(norm_s)

    # first fulfill cutoff criteria
    overhead = np.nonzero(norm_cumsum > 1-cut)[0]

    # first value below threshold
    first_thresh = np.nonzero(norm_s < err_th)[0]

    # We find the cutoff positon
    final_len = len(_s)
    if np.any(first_thresh):
        final_len = first_thresh[0]
    if np.any(overhead):
        final_len = min(final_len, overhead[0]+1)
    if type(max_len) == int:  # isinstance seems broken for bool and int
        final_len = min(final_len, max_len)

    if normalize:
        # Final renormalization of SVD values kept or not, returning the correct
        # matrices sizes
        _s = _s[:final_len]
        _s = _s/np.linalg.norm(_s, ord=norm_ord)
        return _u[:, :final_len], _s, _vh[:final_len, :], final_len
    else:
        return _u[:, :final_len], _s[:final_len], _vh[:final_len, :], final_len


'''
################################################################################
Matrix Product State (MPS) basic operations
################################################################################
'''


def state_to_mps_build(phi, qudit_level=2, normalize=True, max_bond=None):
    '''
    Builds a multi qudit state mps.
    '''
    # Takes a multi QUBIT state, outputs MPS with fantom legs
    mps = []
    # We make the first MPS tensor
    leftovers = phi.reshape(qudit_level, -1)  # correct reshape for qubit MPS
    _u, _s, _vh, _ = reduced_svd(
        leftovers, normalize=normalize, max_len=max_bond)
    mps.append(_u.reshape(1, qudit_level, -1))  # Adding first tensor
    # To arrive at an orthogonality center
    leftovers = np.dot(np.diag(_s), _vh)

    while leftovers.shape[1] > qudit_level:
        link = leftovers.shape[0]  # keep track of link index size
        # correct reshape for qubit MPS
        leftovers = leftovers.reshape((qudit_level*link, -1))
        _u, _s, _vh, _ = reduced_svd(
            leftovers, normalize=normalize, max_len=max)
        _u = _u.reshape(link, qudit_level, -1)  # Getting back bit index
        mps.append(_u)
        leftovers = np.dot(np.diag(_s), _vh)  # For orthogonality center
    # We save the last MPS tensor, the orthogonality center
    mps.append(leftovers.reshape((-1, qudit_level, 1)))

    return mps


def mps_contract(mps, renorm=False, norm_ord=2):
    '''
    Contracts an MPS with open boundary conditions (+ phantom legs).
    Very unefficient exact way.
    '''
    #
    dense = mps[0]

    # We form the dense representation of the mps
    for i in range(1, len(mps)):
        outside, _, _ = dense.shape
        _, _, bond = mps[i].shape
        dense = np.tensordot(dense, mps[i], axes=([2], [0]))
        dense = dense.reshape((outside, -1, bond))

    # Contraction on the extreme indices
    dense = np.trace(dense, axis1=0, axis2=2)

    if renorm:
        dense = dense/np.linalg.norm(dense, ord=norm_ord)
    return dense


def max_bond_size(mps):
    '''
    Finds the max bond dimension of an mps.
    '''
    max_bonds = [max(tens.shape[0], tens.shape[2])
                 for _, tens in enumerate(mps)]

    return max(max_bonds)


def find_left_noniso(mps, precision=1e-02):
    '''
    Finds the left non-unit tensors in a MPS.
    '''
    non_unit = []
    for i, site_tens in enumerate(mps):
        matrix = np.tensordot(site_tens, np.conj(
            site_tens), axes=([0, 1], [0, 1]))
        id_mat = np.identity(matrix.shape[0])
        error = np.allclose(matrix, id_mat, rtol=precision,  atol=precision)
        if error is False:
            non_unit.append(i)

    return non_unit


def find_right_noniso(mps, precision=1e-02):
    '''
    Finds the right non-unit tensors in a MPS.
    '''
    non_unit = []
    for i, site_tens in enumerate(mps):
        matrix = np.tensordot(site_tens, np.conj(
            site_tens), axes=([1, 2], [1, 2]))
        id_mat = np.identity(matrix.shape[0])
        error = np.allclose(matrix, id_mat, rtol=precision,  atol=precision)
        if error is False:
            non_unit.append(i)

    return non_unit


def find_strict_non_iso(mps, precision=1e-02):
    '''
    Returns positions of all tensors which are not either left or right
    isometries. Considering contractions resulting in scalar.
    '''
    non_unit1 = find_right_noniso(mps, precision=precision)
    non_unit2 = find_left_noniso(mps, precision=precision)

    return list(set(non_unit1) & set(non_unit2))


def find_orthogtoright_noniso(mps, precision=1e-02):
    '''
    Finds the non-unit tensors in a MPS considering the last tensor being the
    orthog center.
    '''
    non_unit = []
    last = len(mps)-1
    for i, site_tens in enumerate(mps):
        if i == last:
            matrix = np.tensordot(site_tens, np.conj(
                site_tens), axes=([1, 2], [1, 2]))
        else:
            matrix = np.tensordot(site_tens, np.conj(
                site_tens), axes=([0, 1], [0, 1]))
        id_mat = np.identity(matrix.shape[0])
        error = np.allclose(matrix, id_mat, rtol=precision,  atol=precision)
        if error is False:
            non_unit.append(i)

    return non_unit


def find_orthogtoleft_noniso(mps, precision=1e-02):
    '''
    Finds the non-unit tensors in a MPS considering the last tensor being the
    orthog center.
    '''
    non_unit = []
    for i, site_tens in enumerate(mps):
        if i == 0:
            matrix = np.tensordot(site_tens, np.conj(
                site_tens), axes=([0, 1], [0, 1]))
        else:
            matrix = np.tensordot(site_tens, np.conj(
                site_tens), axes=([1, 2], [1, 2]))
        id_mat = np.identity(matrix.shape[0])
        error = np.allclose(matrix, id_mat, rtol=precision,  atol=precision)
        if error is False:
            non_unit.append(i)

    return non_unit


def find_orthog_center(mps, precision=1e-02):
    '''
    Returns positions of all tensors which are not either left or right
    isometries. Avoiding scalar resulting cases.
    '''
    non_unit1 = find_orthogtoright_noniso(mps, precision=precision)
    non_unit2 = find_orthogtoleft_noniso(mps, precision=precision)

    return list(set(non_unit1) & set(non_unit2))


def _two_sites_mps_reduce(site1, site2, direction='right', **kwargs):
    '''
    Contracts two adjacent site tensors and returns them with a reduced bond
    dimension using the reduce_svd function.
    '''

    if 'svd_func' not in kwargs:
        svd_func = reduced_svd
    else:
        svd_func = kwargs.get('svd_func', None)

    bond_1, site_size_1, _ = site1.shape
    _, site_size_2, bond_3 = site2.shape
    _temp1 = site1.reshape((bond_1*site_size_1, -1))
    _temp2 = site2.reshape((-1, bond_3*site_size_2))
    _temp = np.dot(_temp1, _temp2)
    _u, _s, _vh, _ = svd_func(_temp)
    if direction == 'right':
        _vh = np.dot(np.diag(_s), _vh)
    elif direction == 'left':
        _u = np.dot(_u, np.diag(_s))
    elif direction == 'none':
        _u = np.dot(_u, np.diag(np.sqrt(_s)))
        _vh = np.dot(np.diag(np.sqrt(_s)), _vh)
    else:
        raise ValueError(
            "\'dir\' argument must be \'left\' or \'right\' strings")
    _u = _u.reshape((bond_1, site_size_1, -1))
    _vh = _vh.reshape((-1, site_size_1, bond_3))
    return _u, _vh, _s


def mpsrefresh_lefttoright(mps, begin=0, orth_pos=-1, **kwargs):
    '''
    Moves the orth center from one site to a site to its right. Can be used to
    put the mps in canonical form and normalize it.
    '''

    if 'svd_func' not in kwargs:
        svd_func = reduced_svd
    else:
        svd_func = kwargs.get('svd_func', None)

    length = len(mps)
    if orth_pos < 0:
        end = length+orth_pos
    else:
        end = orth_pos

    if begin < 0:
        begin = length+begin

    for i in range(begin, end):
        # Contract, svd and shape-back
        mps[i], mps[i+1], _ = _two_sites_mps_reduce(
            mps[i], mps[i+1], direction='right', svd_func=svd_func)

    return mps


def mpsrefresh_righttoleft(mps, begin=-1, orth_pos=0, **kwargs):
    '''
    Moves the orth center from one site to a site to its left. Can be used to
    put the mps in canonical form and normalize it.
    '''

    if 'svd_func' not in kwargs:
        svd_func = reduced_svd
    else:
        svd_func = kwargs.get('svd_func', None)

    length = len(mps)
    end = orth_pos
    while end < 0:
        end += length

    while begin < 0:
        begin += length

    cover = begin-end
    for i in range(0, cover):
        # Contract, svd and shape-back
        mps[begin-i-1], mps[begin-i], _ = _two_sites_mps_reduce(
            mps[begin-i-1], mps[begin-i], direction='left', svd_func=svd_func)

    return mps


def move_orthog(mps, begin=0, end=-1, **kwargs):
    '''
    This simply moves the orth center from one position to the other calling the
    refresh functions
    '''

    if 'svd_func' not in kwargs:
        svd_func = reduced_svd
    else:
        svd_func = kwargs.get('svd_func', None)

    while begin < 0:
        begin += len(mps)
    while end < 0:
        end += len(mps)

    if begin == end:
        return mps
    if begin < end:
        return mpsrefresh_lefttoright(
            mps, begin=begin, orth_pos=end, svd_func=svd_func)
    if begin > end:
        return mpsrefresh_righttoleft(
            mps, begin=begin, orth_pos=end, svd_func=svd_func)
    else:
        raise ValueError("\'begin\' and \'end\' values are not compatible")


'''
################################################################################
Specific MPSs and MPOs Builder 
################################################################################
'''


def ones_mps(size, qudit=2):
    '''
    Returns a full ones mps. Equivalent to a list of n reshaped copy tensors of
    size qudit. No orthogonality center and no normalization.
    '''

    # Create the ones tensor
    ones = np.ones(qudit).reshape((1, qudit, 1))
    # create list of ones of length n
    mps = [ones for _ in range(size)]

    return mps


def plus_state_mps(size, qudit=2):
    '''
    Returns a normalized plus tensor product state mps. No orthogonality center.
    '''
    # Create the ones tensor
    plus = np.ones(qudit).reshape((1, qudit, 1))
    plus /= np.sqrt(qudit)
    # create list of ones of length n
    mps = [plus for _ in range(size)]

    return mps


def binary_mps(binary):
    '''
    Turns a classical binary array into a an equivalent normalised MPS.
    '''
    mps = []
    # Initiating tensors
    zero = (np.array([1., 0.])).reshape((1, 2, 1))
    one = (np.array([0., 1.])).reshape((1, 2, 1))

    for _, j in enumerate(binary):
        if j == 0:
            mps.append(zero)
        elif j == 1:
            mps.append(one)
        else:
            raise ValueError(
                " \'binary\' entry must be numpy array with 0/1 values")
    return mps


def ansatz_mps(mps_length, max_chi=20, phys_ind=2):
    '''
    Building a random MPS ansatz of specific length, with maximal bond dim chi
    The physical index size can be adapted (default to qubit). NOT NORMALIZED!!!
    '''
    mps = []
    # first tensor
    if max_chi is False:
        max_chi = np.Inf
    mps.append(np.random.rand(1, phys_ind, min(phys_ind, max_chi)))
    for i in range(1, mps_length):
        bond_size = min(max_chi, phys_ind **
                        (mps_length-i-1), phys_ind**(i+1))

        mps.append(np.random.rand(mps[-1].shape[2], phys_ind, bond_size))

    return mps


def random_mps_gen(_n=8, noise_ratio=0.1, highest_value_index=1, max_bond=100, basis=2):
    '''
    Builds a MPS with a specific main component element and a uniformly random
    noise over other components. Can be used to test main component finding
    methods.
    '''
    # vector of the noise model
    vector = noise_ratio*np.random.rand(basis**_n)
    # Selecting the main component value
    vector[highest_value_index] = 1

    # We can refer to my MPS builder for qubit states
    mps = state_to_mps_build(vector, normalize=True, max_bond=max_bond)

    return mps


def boltz_mpo(size, prob=1/100):
    '''
    This returns an MPO in the form of boltzmann probability boxes.
    '''
    # Boltzmann Box
    boltz = np.array([[1-prob, prob], [prob, 1-prob]])
    boltz = boltz.reshape((2, 2, 1, 1))  # Reshape for indices ordering
    # Create mpo list
    mpo = [boltz for _ in range(size)]
    return mpo


def identity_mpo(size, qudit=2):
    '''
    This returns an MPO made from identity matrices.
    '''
    # identity mpo tensor
    ident = np.identity(n=qudit)
    ident = ident.reshape((qudit, qudit, 1, 1))
    # Create mpo list
    mpo = [ident for _ in range(size)]

    return mpo


'''
################################################################################
MPS-MPO contractor
################################################################################
'''


def _mps_mpo_contract_firstsite(mps_tens, mpo_tens, direction='right'):
    _temp = np.tensordot(mps_tens, mpo_tens, axes=([1], [0]))
    # getting rid of mpo fantom leg
    if direction == 'right':
        opened = _temp[:, :, :, 0, :]
    elif direction == 'left':
        opened = _temp[:, :, :, :, 0]

    return opened


def _mps_mpo_contract_opentoright(opened, mps_tens, mpo_tens, orthog=True, **kwargs):
    '''
    Contracts the 'opened' tensor to the next mps and mpo tensors.
    '''

    if 'svd_func' not in kwargs:
        svd_func = reduced_svd
    else:
        svd_func = kwargs.get('svd_func', None)

    # contraction with mps tens
    _temp = np.tensordot(opened, mps_tens, axes=([1], [0]))
    _temp = np.tensordot(_temp, mpo_tens, axes=([2, 3], [2, 0]))
    _shape = _temp.shape
    _temp = _temp.reshape((_shape[0]*_shape[1], -1))
    _u, _s, _vh, _ = svd_func(_temp)
    if orthog:
        _vh = np.dot(np.diag(_s), _vh)
    else:
        _u = np.dot(_u, np.diag(np.sqrt(_s)))
        _vh = np.dot(np.diag(np.sqrt(_s)), _vh)
    prev_mps = _u.reshape(_shape[:2]+(-1,))
    next_opened = _vh.reshape((-1,)+_shape[2:])

    return prev_mps, next_opened, _s


def mps_mpo_contract_fromlefttoright(mps, mpo, index=0, **kwargs):
    '''
    Partial-mpo to mps contractor. Assumes fantom legs on the mpo ends. Begins
    at start of mpo.
    '''

    if 'svd_func' not in kwargs:
        svd_func = reduced_svd
    else:
        svd_func = kwargs.get('svd_func', None)

    mpo_length = len(mpo)
    mps_length = len(mps)

    while index < 0:
        index += mps_length

    # first site contraction
    open_right = _mps_mpo_contract_firstsite(
        mps[index], mpo[0], direction='right')
    # Contract for all intermediate tensors
    for i in range(1, mpo_length):
        mps[i+index-1], open_right, _ = _mps_mpo_contract_opentoright(
            open_right, mps[i+index], mpo[i], svd_func=svd_func)
    # Form the last mps site
    mps[index+mpo_length-1] = np.transpose(open_right[:, :, :, 0], (0, 2, 1))

    return mps


def _mps_mpo_contract_opentoleft(opened, mps_tens, mpo_tens, orthog=True, **kwargs):
    '''
    Contracts the 'opened' tensor to the previous mps and mpo tensors.
    '''

    if 'svd_func' not in kwargs:
        svd_func = reduced_svd
    else:
        svd_func = kwargs.get('svd_func', None)

    # contraction with mps tens
    _temp = np.tensordot(opened, mps_tens, axes=([0], [2]))
    _temp = np.tensordot(_temp, mpo_tens, axes=([2, 4], [3, 0]))
    _shape = _temp.shape
    _temp = _temp.reshape((_shape[0]*_shape[1], -1))
    _u, _s, _vh, _ = svd_func(_temp)
    if orthog:
        _vh = np.dot(np.diag(_s), _vh)
    else:
        _u = np.dot(_u, np.diag(np.sqrt(_s)))
        _vh = np.dot(np.diag(np.sqrt(_s)), _vh)
    prev_mps = _u.reshape(_shape[:2]+(-1,))
    prev_mps = np.transpose(prev_mps, (2, 1, 0))
    next_opened = _vh.reshape((-1,)+_shape[2:])
    next_opened = np.transpose(next_opened, (1, 0, 2, 3))

    return prev_mps, next_opened, _s


def mps_mpo_contract_fromrighttoleft(mps, mpo, index=0, **kwargs):
    '''
    Partial-mpo to mps contractor. Assumes fantom legs on the mpo ends. Begins
    at end of mpo.
    '''

    if 'svd_func' not in kwargs:
        svd_func = reduced_svd
    else:
        svd_func = kwargs.get('svd_func', None)

    mpo_length = len(mpo)
    mps_length = len(mps)

    while index < 0:
        index += mps_length

    # final mpo contraction point
    start = index+mpo_length-1
    # first site contraction
    open_left = _mps_mpo_contract_firstsite(
        mps[start], mpo[0], direction='right')

    for i in range(1, mpo_length):
        mps[start-i+1], open_left, _ = _mps_mpo_contract_opentoleft(
            open_left, mps[start-i], mpo[-i-1], svd_func=svd_func)
    mps[index] = np.transpose(open_left[:, :, :, 0], (0, 2, 1))

    return mps


def mps_mpo_contract_shortest_moves(mps, mpo, current_orth=-1, index=0, **kwargs):
    '''
    Moves the orth. center of the mps at right position then does partial
    mps-mpo contraction.
    Decides if it is more strategic to contract the mpo from right of left.
    '''

    if 'svd_func' not in kwargs:
        svd_func = reduced_svd
    else:
        svd_func = kwargs.get('svd_func', None)

    mpo_length = len(mpo)
    mps_length = len(mps)

    while index < 0:
        index += mps_length
    while current_orth < 0:
        current_orth += mps_length
    # Finds best strategy
    dist_start = np.abs(index-current_orth)
    dist_end = np.abs(index+mpo_length-1-current_orth)
    if dist_start > dist_end:
        mps = move_orthog(mps, begin=current_orth, end=index +
                          mpo_length-1, svd_func=svd_func)
        mps = mps_mpo_contract_fromrighttoleft(
            mps, mpo, index=index, svd_func=svd_func)
        new_orth = index
    else:
        mps = move_orthog(mps, begin=current_orth,
                          end=index, svd_func=svd_func)
        mps = mps_mpo_contract_fromlefttoright(
            mps, mpo, index=index, svd_func=svd_func)
        new_orth = index+mpo_length-1

    return mps, new_orth


'''
################################################################################
Other basic functions
################################################################################
'''


def real_index(index, list_lenght):
    '''
    Translates negative indices to positive integers
    '''
    while index < 0:
        index += list_lenght
    return index


def integer_to_basis_pos(integer, width, basis=2):
    '''
    Transforms an integer into a numpy array of each digit value in another 
    basis. Important for main component finding.

    ex.: basis=2 -> translates into binary.
    '''
    bin_str = np.base_repr(
        integer, basis)  # Rewriting as string in right basis
    bin_str = bin_str.zfill(width)  # Giving string right len, fill with zeros
    arr = np.fromstring(bin_str, 'u1') - ord('0')  # convert to numpy array

    return arr


def vector_list_maxes(list):
    '''
    Receives a list of vectors (MPS in tensor product state) and return the
    position of the maximal value for each element in a numpy array. Important
    for main component finding.
    '''
    maxes = np.zeros(len(list))  # Creating array of right size

    # Assigning right value for each element of the list
    for _i, tens in enumerate(list):
        maxes[_i] = np.argmax(np.abs(tens))

    return maxes


'''
################################################################################
MPS main component finder
################################################################################
'''


def dephased_tensor_mpo_built(tensor):
    '''
    Builds the correct dephased density matrix MPO element for an mps tensor.
    Follows general TNTools indices ordering.
    '''
    # Build density matrix
    tens_shape = tensor.shape
    conj_tensor = np.conj(tensor)
    mpo_tens = np.tensordot(conj_tensor, tensor, axes=([], []))
    mpo_tens = np.transpose(mpo_tens, axes=(1, 4, 0, 3, 2, 5))
    mpo_tens = mpo_tens.reshape(
        tens_shape[1], tens_shape[1], tens_shape[0]**2, tens_shape[2]**2)

    # Eliminating off-diagonal elements
    for i in range(tens_shape[0]**2):
        for j in range(tens_shape[2]**2):
            mpo_tens[:, :, i, j] = np.diag(np.diag(mpo_tens[:, :, i, j]))

    return mpo_tens


def dephased_mpo_built(mps):
    '''
    Builds the correct dephased density matrix MPO a complete MPS.
    Follows general TNTools indices ordering.
    '''
    return [dephased_tensor_mpo_built(_t) for _, _t in enumerate(mps)]


def main_component(mps, method='exact', **kwargs):
    '''
    Various ways of finding the main component (largest value) of an MPS.
    Returns a numpy array with the index value of each tensor of the MPS.

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
    if method == 'dephased_DMRG':
        # Defining maximal bond dimension
        if 'chi_max' not in kwargs:
            warnings.warn(
                'No max bond dimension (\'chi_max\') detected. No bound by default, \'chi_max\'=False.')
            chi_max = False
        else:
            chi_max = kwargs.get('chi_max', None)
        # Dephased mpo built from mps
        print('building dephased mpo')
        mpo = dephased_mpo_built(mps)
        print('building mps ansatz')
        ans_mps = ansatz_mps(len(mps), chi_max)  # An ansatz mps
        print('initializing DMRG class')
        deph_dmrg = Dmrg(
            ans_mps, mpo, chi_max=chi_max, eig_method='LM')  # DMRG largest value
        print('starting the DMRG')
        res_mps, _ = deph_dmrg.run()

        # Simplify mps then get main component
        # Defining svd function
        def svd_func(_m):
            return simple_reduced_svd(_m, max_len=1)
        # Swiping to tensor prod state
        simple_mps = mpsrefresh_righttoleft(res_mps, svd_func=svd_func)
        main_comp = vector_list_maxes(simple_mps)  # Max pos of each tensor

    elif method == 'exact':
        vector = mps_contract(mps)  # Contract mps to vector
        max_pos = np.argmax(np.abs(vector))  # max pos in vector
        main_comp = integer_to_basis_pos(
            max_pos, len(mps), basis=(mps[0].shape[1]))  # numpy array equivalent

    elif method == 't_prod_state':
        # Defining svd function
        def svd_func(_m):
            return simple_reduced_svd(_m, max_len=1)
        # Swiping to tensor prod state
        simple_mps = mpsrefresh_righttoleft(mps, svd_func=svd_func)
        main_comp = vector_list_maxes(simple_mps)  # Max pos of each tensor

    return main_comp


'''
################################################################################
Canonical MPS class
################################################################################
'''


class MpsStateCanon:
    '''
    MPS class object.

    -mps_list: list of mps tensors (3 legs each)
    optional:
        -orth_pos: position at which the orthogonality center can be found.
                    finds this position by default
        -svd_func: custom selected svd function used. The cut parameters, max bond
                    dims and etc. are selected from there. By default, reduced_svd
                    func in this file.

    '''

    def __init__(self, mps_list, **kwargs):
        self.mps = mps_list.copy()
        self.length = len(mps_list)
        self.orth_pos = None
        # initialize the orthogonality center position
        if 'orth_pos' not in kwargs:
            orth_poses = find_orthog_center(self.mps)
            if len(orth_poses) != 1:
                warnings.warn(
                    f'MPS seems to be in non-valid canonical form (one orthog. center) orth. poses ={orth_poses} not size 1 !')
                self.orth_pos = None
            else:
                self.orth_pos = orth_poses[0]
        else:
            self.orth_pos = real_index(
                kwargs.get('orth_pos', None), self.length)
        # Default SVD function
        if 'svd_func' not in kwargs:
            warnings.warn(
                'No SVD function specified for MPS class. Using standard \'no-cut\' reduced_SVD!')
            self.svd_func = reduced_svd
        else:
            self.svd_func = kwargs.get('svd_func', None)

    def test_the_svd(self, test_size=10):
        '''
        Quick test to verify if svd_func has the right properties.
        '''
        _m = np.random.rand(test_size, test_size)
        _u, _s, _vh, _ = self.svd_func(_m)
        print(_s)

    def update_svd_func(self, svd_func):
        '''
        Changes the default svd function by which operations are done.
        '''
        self.svd_func = svd_func

    def create_orth(self, position=-1):
        '''
        Puts the mps in canonical form and puts the orthog center at specified
        position.
        '''
        if self.orth_pos is None:
            self.mps = move_orthog(
                self.mps, begin=0, end=-1, svd_func=self.svd_func)
            self.mps = move_orthog(
                self.mps, begin=-1, end=position, svd_func=self.svd_func)
            self.orth_pos = real_index(position, self.length)
        else:
            warnings.warn(
                f'MPS already in canonical form with orthog position ={self.orth_pos}! Moving orthog center at required position={position}')
            self.mps = move_orthog(
                self.mps, begin=self.orth_pos, end=position, svd_func=self.svd_func)
            self.orth_pos = real_index(position, self.length)

    def move_orth(self, position):
        '''
        Moves the orthogonality center at given position.
        '''
        if self.orth_pos is None:
            warnings.warn(
                f'MPS has no orth. center, so cannot be moved at position {position}!')
        else:
            self.mps = move_orthog(
                self.mps, begin=self.orth_pos, end=position, svd_func=self.svd_func)
            self.orth_pos = real_index(position, self.length)

    def adjust_orth(self):
        '''
        Verifies that the orthog center is at the right position. If there is one
        orthog not at right position, changes orth_pos to that position. If the
        mps is not in canonical form, orth_pos is put to none.
        '''
        orth_poses = find_orthog_center(self.mps)
        if len(orth_poses) != 1:
            warnings.warn(
                f'MPS seems to be in non-valid canonical form (one orthog. center) orth. poses ={orth_poses} not size 1 !')
            self.orth_pos = None
        elif self.orth_pos != orth_poses[0]:
            warnings.warn(
                f'MPS orth_pos center seems to be at wrong position, adjusted to be at position {orth_poses} now!')
            self.orth_pos = orth_poses[0]

    def mpo_contract(self, mpo, beginning=0):
        '''
        Contracts mpo to the current mps. The orthgonality center is automatically
        updated.
        -mpo: MPO tensor list
        -beginning: position at the mps to which the first mpo tensor is applied.
        '''
        self.mps, self.orth_pos = mps_mpo_contract_shortest_moves(
            self.mps, mpo=mpo, current_orth=self.orth_pos, index=beginning, svd_func=self.svd_func)

    def main_component(self, method='exact', **kwargs):
        '''
        Finds largest value element (main component) of mps equivalent vector.
        See 'main_component' function in main component finding section of
        TNTools. 
        '''
        # Getting chi_max value for dephased_DMRG case.
        chi_max = False
        if method == 'dephased_DMRG':
            if 'chi_max' not in kwargs:
                warnings.warn(
                    'No max bond dimension (\'chi_max\') detected. No bound by default, \'chi_max\'=False.')
                chi_max = False
            else:
                chi_max = kwargs.get('chi_max', None)
            return main_component(self.mps, method=method, chi_max=chi_max)
        return main_component(self.mps, method=method)


'''
################################################################################
Other General TN algorithms
################################################################################
'''


class Dmrg():
    """
    Parameters
    ----------
    ansatz : MPS tensor list with phantom legs
    mpo : The Hamiltonian in mpo format
    chi_max : Maximal bond dimension permitted. No bound with chi_max=False.
    nsweep : Maximum number of sweeps done
    energy_var : Minimum energy variation in one step to declare conversion
    ansatz_is_left_mps : If ansatz input is in left canonical form
    debug : - 'partial': prints energy at each sweep
            - 'full': prints at each step of each sweep
            - 'None': no printing
    eig_method : - 'LM' : largest magnitude
                     - 'SM' : smallest magnitude
                     - 'LR' : largest real part
                     - 'SR' : smallest real part
                     - 'LI' : largest imaginary part
                     - 'SI' : smallest imaginary part
                     (See scipy documentation)

    Notes
    -----
    A MPS in right form (abreviated right_mps) is in right canonical form.
    i.e. orthog center is the first tensor on the left.

    Similarly, a MPS is in left form if the orthogonality center is the
    right-most tensor.
    """

    def __init__(self, ansatz, mpo, chi_max, maxsweep=10, energy_var=1E-5, ansatz_is_left_mps=False, debug='partial', **kwargs):
        self.nsites = len(ansatz)
        self.mpo = mpo
        self.energy_var = energy_var
        self.nsweep = maxsweep
        self.debug = debug
        # Retrieving svd function given, or chi_max
        if 'svd_func' not in kwargs:
            # If none, selects svd func with chi_max parameter
            self.svd_func = lambda m: reduced_svd(
                m, max_len=chi_max, cut=0, normalize=True, err_th=0)
        else:
            warnings.warn(
                'SVD function specified for DMRG class. Parameter chi_max made obsolete.')
            self.svd_func = kwargs.get('svd_func', None)
        # eigensolver method
        if 'eig_method' not in kwargs:
            self.eig_method = 'SM'
        else:
            self.eig_method = kwargs.get('eig_method', None)

        # Array of singular values at each position
        self.singular_values = [np.array([0])] * (self.nsites + 1)

        # ansatz mps must be in left orth form
        if ansatz_is_left_mps:
            self.left_mps = ansatz.copy()
        else:
            self.left_mps = ansatz.copy()
            self._ansatz_mps_to_left_form(self.left_mps)

        # Getting singular value tensor at last position
        self._get_last_singular()
        # initializing right canonical tensor list
        self.right_mps = [np.array([0]).reshape((1, 1, 1))] * self.nsites

        # Useful parameters
        self.energies = []  # Storing energy value at each sweep
        self.sweep = 0  # Number of sweeps so far
        self.left_env = []
        self.right_env = []

    def _ansatz_mps_to_left_form(self, ansatz):
        """
        Transfers the ansatz mps into the left orthogonal form (i.e. orthog
        center on right). Update the last sigma value tensor.
        """
        for site in range(len(ansatz)-1):
            chil, chid, chir = ansatz[site].shape
            utemp, stemp, vhtemp, _ = simple_reduced_svd(
                ansatz[site].reshape(chil*chid, chir), normalize=True)
            self.left_mps[site] = utemp.reshape(chil, chid, chir)
            self.left_mps[site+1] = np.tensordot(
                np.diag(stemp) @ vhtemp, ansatz[site+1], axes=([1], [0]))
            self.singular_values[self.nsites] = np.diag(stemp) @ vhtemp

    def _get_last_singular(self):
        '''
        Initialize the singular value matrix at last position of MPS. Necessary
        before valid right to left sweep.
        '''
        chil, chid, chir = self.left_mps[self.nsites-1].shape
        utemp, stemp, vhtemp, _ = simple_reduced_svd(
            self.left_mps[self.nsites-1].reshape(chil*chid, chir), normalize=True)
        self.left_mps[self.nsites-1] = utemp.reshape((chil, chid, chir))
        self.singular_values[self.nsites] = (np.diag(stemp) @ vhtemp)

    def run(self):
        """
        DMRG run. Execute sweeps til maximum number (self.nsweep) is reach or
        until conversion is detected.
        """
        self._initialize_env()
        self.sweep = 0  # Counting sweeps
        conversion = False  # Activates when conversion
        while self.sweep <= self.nsweep-1 and conversion is False:
            self.sweep += 1
            # Do back and forth sweep
            self._sweep_orthog_right_to_left()
            self._sweep_orthog_left_to_right()
            if self.debug != 'None':
                print(
                    f'Sweep {self.sweep}/{self.nsweep} [Energy: {self.energies[-1]}]')

            # Evaluates conversion
            if self.sweep > 1:
                ener_dif = np.abs(self.energies[-1]-self.energies[-2])
                if ener_dif <= self.energy_var:
                    if self.debug != 'None':
                        print(
                            f'Energy conversion detected: {ener_dif} < {self.energy_var}.')
                    conversion = True

        return self.left_mps, self.energies

    @staticmethod
    def build_left_env(prev_env, mpo_t, mps_t):
        '''
        Creates the left environment at a position from the previous left
        environment and the right mps and mpo tensors.
        '''
        next_env = np.tensordot(prev_env, mpo_t, axes=([0], [2]))
        next_env = np.tensordot(
            next_env, np.conj(mps_t), axes=([0, 3], [0, 1]))
        next_env = np.tensordot(next_env, mps_t, axes=([0, 1], [0, 1]))

        return next_env

    @staticmethod
    def build_right_env(next_env, mpo_t, mps_t):
        '''
        Creates the right environment at a position from the previous right
        environment and the right mps and mpo tensors.
        '''
        prev_env = np.tensordot(next_env, mpo_t, axes=([0], [3]))
        prev_env = np.tensordot(
            prev_env, np.conj(mps_t), axes=([0, 3], [2, 1]))
        prev_env = np.tensordot(prev_env, mps_t, axes=([0, 1], [2, 1]))

        return prev_env

    @staticmethod
    def full_env(left_env, mpo1, mpo2, right_env, matricize=False):
        """
         Creates the full environment before the eigenvalue solving.
        """

        _temp = np.tensordot(left_env, mpo1, axes=([0], [2]))
        _temp = np.tensordot(_temp, mpo2, axes=([4], [2]))
        _temp = np.tensordot(_temp, right_env, axes=([6], [0]))
        full_env = np.transpose(_temp, (1, 2, 4, 7, 0, 3, 5, 6))

        if matricize:
            _s1, _s2, _s3, _s4, _, _, _, _ = full_env.shape
            full_env = full_env.reshape(_s1*_s2*_s3*_s4, _s1*_s2*_s3*_s4)

        return full_env

    @staticmethod
    def eig_solver(init_vec, env_mat, method='SM'):
        '''
        Finds the largest eigenvector using ''best'' method.
        '''
        val, vec = eigsh(env_mat, k=1,  which=method,
                         v0=init_vec, maxiter=None)

        return val[0], vec

    def _initialize_env(self):
        """ Initialize the environment tensors.

        The left environment tensors will be fully initialized while the
        right ones are simply made as a list of trivial tensors of correct shapes.

        During the DMRG iteration for site i and i + 1, the left environment is
        the contraction of all sites with index j < i and the right environment
        is the contraction of all sites with index j > i + 1.

        The corresponding environments are stored at index i in the left_env and
        right_env parameters.
        """
        left_edge = np.array([1]).reshape((1, 1, 1))
        right_edge = np.array([1]).reshape((1, 1, 1))

        self.left_env = [left_edge] + [None] * (self.nsites - 1)
        self.right_env = [None] * (self.nsites - 1) + [right_edge]

        for _i in range(self.nsites-1):
            self.left_env[_i+1] = self.build_left_env(
                self.left_env[_i], self.mpo[_i], self.left_mps[_i])

    def _sweep_orthog_right_to_left(self):
        '''
        DMRG sweep updating the orthgonality center (sigma matrices) from the
        last (right) to the first (left).

        At each step, it replaces a pair of sites with their ground-state wavefunction.
        '''

        for site in reversed(range(self.nsites - 1)):
            # two-site update
            chi_l, chi_d1, _ = self.left_mps[site].shape
            _, chi_d2, chi_r = self.left_mps[site+1].shape

            # updating the two sites with eigenvalue decomposition
            psi_ground = np.tensordot(
                self.left_mps[site+1], self.singular_values[site+2], axes=([2], [0]))
            psi_ground = np.tensordot(self.left_mps[site], psi_ground, axes=(
                [2], [0])).reshape(chi_l*chi_d1*chi_d2*chi_r)

            my_env = self.full_env(
                self.left_env[site], self.mpo[site], self.mpo[site+1],
                self.right_env[site+1], matricize=True)
            energy, psi_ground = self.eig_solver(
                psi_ground, my_env, method=self.eig_method)

            # getting back all elements from updated ground state
            utemp, stemp, vhtemp, _ = self.svd_func(
                psi_ground.reshape(chi_l*chi_d1, chi_d2*chi_r))

            self.left_mps[site] = utemp.reshape(chi_l, chi_d1, -1)
            self.singular_values[site+1] = np.diag(stemp)
            self.right_mps[site + 1] = vhtemp.reshape(-1, chi_d2, chi_r)

            # generating new right env
            self.right_env[site] = self.build_right_env(
                self.right_env[site+1], self.mpo[site+1], self.right_mps[site+1])

            if self.debug == 'full':
                print(f'Pos. {site}, sweep {self.sweep}, energy: {energy}')

        # left boundary tensor site
        chil, chid, chir = self.left_mps[0].shape
        temp = np.tensordot(self.left_mps[0], self.singular_values[1], axes=(
            [2], [0])).reshape((chil, -1))
        utemp, stemp, vhtemp, _ = self.svd_func(temp)
        self.right_mps[0] = vhtemp.reshape(chil, chid, chir)
        self.singular_values[0] = utemp @ np.diag(stemp)

    def _sweep_orthog_left_to_right(self):
        """ Moves the orthogonality center (singular values matrices) from the
        first (left) site to the last (right) site.

        At each step, it replaces a pair of sites with their ground-state wavefunction.
        """
        for site in range(self.nsites - 1):
            # Contract orthog center, site tensor and next site tensor.
            # Computer lowest energy eigenstate of the contraction.
            # Run SVD and truncates the lowest eigenstate
            # Update both site tensors with the truncated eigenstate.
            # Update envs and Sweights

            chi_l, chi_d1, _ = self.right_mps[site].shape
            _, chi_d2, chi_r = self.right_mps[site+1].shape

            psi_ground = np.tensordot(
                self.singular_values[site], self.right_mps[site], axes=([1], [0]))
            psi_ground = np.tensordot(
                psi_ground, self.right_mps[site+1], axes=([2], [0])).reshape(chi_l*chi_d1*chi_d2*chi_r)

            # sites update
            my_env = self.full_env(
                self.left_env[site], self.mpo[site], self.mpo[site+1], self.right_env[site+1], matricize=True)
            energy, psi_ground = self.eig_solver(
                psi_ground, my_env, method=self.eig_method)

            # getting back all elements from updated ground state
            utemp, stemp, vhtemp, _ = self.svd_func(
                psi_ground.reshape(chi_l*chi_d1, chi_d2*chi_r))

            self.left_mps[site] = utemp.reshape(chi_l, chi_d1, -1)
            self.singular_values[site+1] = np.diag(stemp)
            self.right_mps[site + 1] = vhtemp.reshape(-1, chi_d2, chi_r)

            # New left env
            self.left_env[site+1] = self.build_left_env(
                self.left_env[site], self.mpo[site], self.left_mps[site])

            if self.debug == 'full':
                print(f'Pos. {site}, sweep {self.sweep}, energy: {energy}')

        self.energies.append(energy)  # Storing the final energy
        # right boundary tensor
        chil, chid, chir = self.right_mps[self.nsites-1].shape
        temp = np.tensordot(
            self.singular_values[self.nsites-1], self.right_mps[self.nsites-1], axes=([1], [0])).reshape(chil*chid, chir)
        utemp, stemp, vhtemp, _ = self.svd_func(temp)
        self.left_mps[self.nsites-1] = utemp.reshape(chil, chid, chir)
        self.singular_values[self.nsites] = stemp*vhtemp
