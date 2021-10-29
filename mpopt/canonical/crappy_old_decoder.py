import numpy as np
import random
import warnings
import TN_libraries.TNTools as tnt
from tensorcsp import *
import os
import quimb


'''
Classical LDPC decoder using TNTools MPS-MPO formalism. 

'''


'''
The following are old legacy function for generating LDPC codes. Will be
by the correct tools using pyqec in due time.
'''

def gen_43_from_plane(n,scramble=True):
    """ Returns 'punctured' grid with a [4,3] degree sequence.
        n: number of unit cells.
    """
    g = Graph.Lattice([4,4*n])     # Generate periodic square grid
    for i in range(16*n-1,-1,-1):  # Punch "holes"
        x,y = unravel_index(i,[4,4*n],'F')
        if (x%2==1 and y%2==1 and x%4==y%4): g.delete_vertices(i)
    o = argsort(g.degree())     # Low degree vertices (bits) first
    p = list(argsort(o))        # igraph uses inverse permutation
    gs = g.permute_vertices(p)  # g and gs bipartite by definition
    if ( scramble ):
        b = array(gs.get_adjacency().data)[:8*n,8*n:]
        while 1:
            m = curveball(b)
            a = zeros([14*n,14*n],int)
            a[:8*n,8*n:] = m
            a = a + a.T
            gs = Graph.Adjacency(a.tolist(),'MAX')
            if ( gs.is_connected() ): break
    return gs

def curveball(m,n=0):
    nz = [list(nonzero(r)[0]) for r in m]
    nr,nc = m.shape
    l = arange(len(nz))
    if ( n == 0 ): n = 5*min(nr, nc)
    for rep in range(n):
        a,b = random.choice(l, 2)
        ab = set(nz[a])&set(nz[b]) # common elements
        l_ab=len(ab)
        l_a=len(nz[a])
        l_b=len(nz[b])
        if l_ab not in [l_a,l_b]:
            tot=list(set(nz[a]+nz[b])-ab)
            ab=list(ab)
            random.shuffle(tot)
            L=l_a-l_ab
            nz[a] = ab+tot[:L]
            nz[b] = ab+tot[L:]
    out_mat = zeros(m.shape, dtype=int)
    for r in range(nr): out_mat[r, nz[r]] = 1
    result = out_mat
    return result


def ldpc_34_gen(n=1):
    #generate a list of ldpc codes with 3 edgs/var 4 edgs/cnstrt
    g=gen_43_from_plane(n)

    chk_ind=g.vs.select(_degree = 4).indices
    ldpc_checks=[]

    for ind in chk_ind:
        neis = g.neighbors(ind, mode="out")
        ldpc_checks.append(neis)

    return ldpc_checks

def rand_bin_array(K, N):
    arr = np.zeros(N, dtype=int)
    arr[:K]  = 1
    random.shuffle(arr)
    return arr

'''
Theses are functions for creating mpo xor checks following TNTools format
'''

def xor_4t2():
    #creating initial 3 legs xor tensor
    reverse_id=np.array([[0.,1.],[1.,0.]])
    xor_3t2=np.zeros((2,2,2))
    xor_3t2[0,:,:]=reverse_id
    xor_3t2[1,:,:]=np.identity(2)

    #Creating initial delta tensor
    delta=np.zeros((2,2,2))
    delta[0,:,:]=np.array([[1.,0.],[0.,0.]])
    delta[1,:,:]=np.array([[0.,0.],[0.,1.]])

    #contracting the two to generate generic xor MPO tensor in correct format
    tens=np.tensordot(delta,xor_3t2,axes=([2],[0]))

    return tens

def parity_TN(array,par_tens=xor_4t2(),adjacency=False):
    #Creating simple cross tensor
    cross=np.tensordot(np.identity(2),np.identity(2),axes=0)
    #adapting receiving of input, adjacency matrix column or check positions
    if adjacency==True:
        nonzeros=np.nonzero(array)[1]
    else:
        nonzeros=np.array(array)
    

    #init list and tens dimensions
    mpo=[]
    tens_shape = par_tens.shape

    #We save the first element of the MPO, index fixed linked to #tens. parity
    if nonzeros.size%2==0:
        mpo.append(par_tens[:,:,0,:].reshape((tens_shape[0],tens_shape[1],1,tens_shape[3])))
    else:
        mpo.append(par_tens[:,:,1,:].reshape((tens_shape[0],tens_shape[1],1,tens_shape[3])))
    
    #we assign a tensor at each position of MPO with check or not
    for i in range(nonzeros[0]+1,nonzeros[-1]):
        if i in nonzeros:
            mpo.append(par_tens)
        else:
            mpo.append(cross)
    #We save the last tensor
    mpo.append(par_tens[:,:,:,0].reshape((tens_shape[0],tens_shape[1],tens_shape[2],1)))

    #We return the list of tensors, the positions of the mps that are assigned 
    #to the first and last mpo tensors.
    
    return mpo, nonzeros[0], nonzeros[-1]

'''
def classical_ldpc_toMPS(entry,checks,prob=1/np.pi,lim='none'):
    MPS=binary_MPS(entry)
    MPO=boltz_MPO(entry.size,p=prob)
    MPS=MPS_MPO(MPS, MPO)

    i=0
    frm=-1
    bonds=[] #Storing the max bond dimension through time

    if lim=='none':
        lim=len(MPS)

    for check in checks[:lim]:
        print(check)
        MPO, begin, last=parity_TN(check)
        MPS=move_orth(MPS,frm=frm,to=begin,renorm=True)
        MPS=MPS_MPO(MPS, MPO, begin)
        bonds.append(max_bond_size(MPS))
        i+=1
        frm=last

        print(str(i)+' checks done out of '+str(len(checks)))

    MPS=move_orth(MPS,frm=frm,to=-1,renorm=True)

    return MPS, bonds
'''

def classical_ldpc_decoding(entry, checks, extract='exact', chi_max=False, b_prob=0.1, max_len=False, cut=0.0, err_th=1E-30):
    #Initiating svd function
    def svd_func(_m):
        return tnt.reduced_svd(_m, cut=cut, max_len=max_len, normalize=True, init_norm=True,
                norm_ord=2, err_th=err_th)

    #Creating the initial mps state class
    entry_mps=tnt.binary_mps(entry)
    state_mps=tnt.MpsStateCanon(entry_mps, orth_pos = None, svd_func=svd_func)
    state_mps.create_orth() #an orth center is created at the last site by default

    #Inputing boltzmann weight
    boltz_mpo = tnt.boltz_mpo(state_mps.length, prob=b_prob)
    state_mps.mpo_contract(boltz_mpo)

    #Procedure to protect from complete state kill
    try: 
        #Filtering loop
        for check in checks:
            print(check)
            c_mpo, begin, _=parity_TN(check)
            state_mps.mpo_contract(c_mpo, begin)
        

        #check for nans and infs in array
        NanInf, _ = state_mps.nans_infs_find()
        if NanInf is True:
            warnings.warn('Final mps is invalid (NANs or infs). Try again with less approximative svd function.')
            main_comp=None
        else:
            #Main component extraction
            main_comp = state_mps.main_component(method=extract, chi_max=chi_max)
    except ValueError:
        warnings.warn('Total norm elimination of the state. Try again with less approximative svd function.')
        main_comp=None

    return main_comp

def single_test(n=2, error_size=1):
    entry=rand_bin_array(error_size,n*8)
    checks=ldpc_34_gen(n)
    print(classical_ldpc_decoding(entry, checks))

single_test()

