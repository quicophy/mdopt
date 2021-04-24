'''
This a variation of the legacy TN_tools file.

The functions are made to input fantom legs. The format permits open boundary
conditions to the mps

'''


import numpy as np
import scipy.linalg


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
    if max_len is not None:
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


def _two_sites_mps_reduce(site1, site2, renorm=False, ord=2, dir='right'):
    '''
    Contracts two adjacent site tensors and returns them with a reduced bond
    dimension using the reduce_svd function.
    '''

    bond_1, site_size_1, _ = site1.shape
    _, site_size_2, bond_3 = site2.shape
    _temp1 = site1.reshape((bond_1*site_size_1, -1))
    _temp2 = site2.reshape((-1, bond_3*site_size_2))
    _temp = np.dot(_temp1, _temp2)
    _u, _s, _vh, _ = reduced_svd(_temp, normalize=renorm, norm_ord=ord)
    if dir == 'right':
        _vh = np.dot(np.diag(_s), _vh)
    elif dir == 'left':
        _u = np.dot(_u, np.diag(_s))
    elif dir == 'none':
        _u = np.dot(_u, np.diag(np.sqrt(_s)))
        _vh = np.dot(np.diag(np.sqrt(_s)), _vh)
    else:
        raise ValueError(
            "\'dir\' argument must be \'left\' or \'right\' strings")
    _u = _u.reshape((bond_1, site_size_1, -1))
    _vh = _vh.reshape((-1, site_size_1, bond_3))
    return _u, _vh, _s


def mpsrefresh_lefttoright(mps, begin=0, orth_pos=-1, renorm=False, ord=2):
    '''
    Moves the orth center from one site to a site to its right. Can be used to
    put the mps in canonical form and normalize it.
    '''
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
            mps[i], mps[i+1], renorm=renorm, ord=ord, dir='right')

    return mps


def mpsrefresh_righttoleft(mps, begin=-1, orth_pos=0, renorm=False, ord=2):
    '''
    Moves the orth center from one site to a site to its left. Can be used to
    put the mps in canonical form and normalize it.
    '''
    length = len(mps)
    end = orth_pos
    while end < 0:
        end += length

    while begin < 0:
        begin += length

    cover = begin-end
    for i in range(0, cover):
        print(i)
        # Contract, svd and shape-back
        mps[begin-i-1], mps[begin-i], _ = _two_sites_mps_reduce(
            mps[begin-i-1], mps[begin-i], renorm=renorm, ord=ord, dir='left')

    return mps


def move_orthog(mps, begin=0, end=-1, renorm=False, ord=2):
    '''
    This simply moves the orth center from one position to the other calling the
    refresh functions
    '''
    while begin < 0:
        begin += len(mps)
    while end < 0:
        end += len(mps)

    if begin == end:
        return mps
    if begin < end:
        return mpsrefresh_lefttoright(
            mps, begin=begin, orth_pos=end, renorm=renorm, ord=ord)
    if begin > end:
        return mpsrefresh_righttoleft(
            mps, begin=begin, orth_pos=end, renorm=renorm, ord=ord)
    else:
        raise ValueError("\'begin\' and \'end\' values are not compatible")


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


def _mps_mpo_contract_firstsite(mps_tens, mpo_tens, direction='right'):
    _temp = np.tensordot(mps_tens, mpo_tens, axes=([1], [0]))
    # getting rid of mpo fantom leg
    if direction == 'right':
        opened = _temp[:, :, :, 0, :]
    elif direction == 'left':
        opened = _temp[:, :, :, :, 0]

    return opened


def _mps_mpo_contract_opentoright(opened, mps_tens, mpo_tens, orthog=True):
    '''
    Contracts the 'opened' tensor to the next mps and mpo tensors.
    '''
    # contraction with mps tens
    _temp = np.tensordot(opened, mps_tens, axes=([1], [0]))
    _temp = np.tensordot(_temp, mpo_tens, axes=([2, 3], [2, 0]))
    _shape = _temp.shape
    _temp = _temp.reshape((_shape[0]*_shape[1], -1))
    _u, _s, _vh, _ = reduced_svd(_temp)
    if orthog:
        _vh = np.dot(np.diag(_s), _vh)
    else:
        _u = np.dot(_u, np.diag(np.sqrt(_s)))
        _vh = np.dot(np.diag(np.sqrt(_s)), _vh)
    prev_mps = _u.reshape(_shape[:2]+(-1,))
    next_opened = _vh.reshape((-1,)+_shape[2:])

    return prev_mps, next_opened, _s


def mps_mpo_contract_fromlefttoright(mps, mpo, index=0):
    '''
    Partial-mpo to mps contractor. Assumes fantom legs on the mpo ends. Begins 
    at start of mpo.
    '''
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
            open_right, mps[i+index], mpo[i])
    # Form the last mps site
    mps[index+mpo_length-1] = np.transpose(open_right[:, :, :, 0], (0, 2, 1))

    return mps


def _mps_mpo_contract_opentoleft(opened, mps_tens, mpo_tens, orthog=True):
    # contraction with mps tens
    _temp = np.tensordot(opened, mps_tens, axes=([0], [2]))
    _temp = np.tensordot(_temp, mpo_tens, axes=([2, 4], [3, 0]))
    _shape = _temp.shape
    _temp = _temp.reshape((_shape[0]*_shape[1], -1))
    _u, _s, _vh, _ = reduced_svd(_temp)
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


def mps_mpo_contract_fromrighttoleft(mps, mpo, index=0):
    '''
    Partial-mpo to mps contractor. Assumes fantom legs on the mpo ends. Begins
    at end of mpo.
    '''
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
            open_left, mps[start-i], mpo[-i-1])
    mps[index] = np.transpose(open_left[:, :, :, 0], (0, 2, 1))

    return mps


def binary_mps(binary):
    '''
    Turns a classical binary array into a an equivalent normalised MPS.
    '''
    mps = []
    # Initiating tensors
    zero = (np.array([1., 0.])).reshape(1, 2, 1)
    one = (np.array([0., 1.])).reshape(1, 2, 1)

    for _, j in enumerate(binary):
        if j == 0:
            mps.append(zero)
        elif j == 1:
            mps.append(one)
        else:
            raise ValueError(
                " \'binary\' entry must be numpy array with 0/1 values")
    return mps


def max_bond_size(mps):
    '''
    Finds the max bond dimension of an mps.
    '''
    max_bonds = [max(tens.shape[0], tens.shape[2])
                 for _, tens in enumerate(mps)]

    return max(max_bonds)


if __name__ == "__main__":

    '''
    mat = np.random.rand(5, 5)
    _, s, _, i = simple_reduced_svd(mat)
    print(f's={s}')
    print(i)
    # '''

    '''
    phi = np.random.rand(2**10)
    phi = phi/np.linalg.norm(phi, ord=2)

    yo = state_to_mps_build(phi)
    print(len(yo))
    # '''

    '''
    print('Yo!')
    tot = mps_contract(yo)

    non_iso = find_orthog_center(yo)

    print(phi)
    print(tot)
    print(non_iso)
    # '''

    '''
    an_mps = plus_state_mps(6)
    print(an_mps[3])
    print(len(an_mps))

    an_mps = mpsrefresh_lefttoright(an_mps)
    # an_mps = mpsrefresh_righttoleft(an_mps, begin=-1, orth_pos=3)
    an_mps = move_orthog(an_mps, begin=-1, end=3)
    print(find_orthog_center(an_mps))
    # '''
    '''
    an_mps = plus_state_mps(6)
    an_mpo = identity_mpo(4)

    an_mps = move_orthog(an_mps)
    an_mps = move_orthog(an_mps, begin=-1, end=2)
    print(find_orthog_center(an_mps))
    # print(len(an_mps))

    another_mps = mps_mpo_contract_fromlefttoright(an_mps, an_mpo, index=2)

    print(find_orthog_center(an_mps))
    '''
    '''
    an_mps = plus_state_mps(6)
    an_mpo = identity_mpo(4)

    an_mps[2] = an_mps[2]*2
    an_mps = move_orthog(an_mps)
    #an_mps = move_orthog(an_mps, begin=-1, end=2)

    print(find_orthog_center(an_mps))
    # print(len(an_mps))

    another_mps = mps_mpo_contract_fromrighttoleft(an_mps, an_mpo, index=2)

    print(find_orthog_center(another_mps))
    '''
    binarray = np.zeros(4)
    binarray[0] = 1
    an_mps = binary_mps(binarray)
    print(max_bond_size(an_mps))
    print(an_mps)
