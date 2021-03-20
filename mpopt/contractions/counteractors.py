'''
A collection of tensor tools made for the general MPOpt project.

Samuel Desrosiers

To be done:

-Comment every function
-Adapt the MPS format (fatom legs)
-Add a bottom_to_top MPS-MPO contractor for efficiency measures
-Create a function for adding and deleting sites
-Create a function to build logical operator MPO's
-Every function is testable

Might be useful to look for an altenative
'''


import numpy as np
import scipy.linalg


def dagger(M):
    #Do the transpose conjugate of a vector or matrix
    M=np.transpose(M)
    M=np.conj(M)
    return M

def reduced_SVD(M,cut=0.00,max=1000, normalize=True,init_norm=True, norm_ord=2,
                limit_max=False, err_th=0):
    '''
    This is a reduced SVD function.
    cut is the norm value cut for lower svd values;
    limit_max activates an upper limit to the spectrum's size;
    normalize activates normalization of final svd spectrum;
    norm_ord choose the vector normalization order;
    init_norm make use of relative norm for unormalized tensor's decomposition.

    '''

    #print(M.shape)
    #nans=np.argwhere(np.isnan(M))
    #print(nans)
    #U,S,Vh = scipy.linalg.svd(M,full_matrices=False,lapack_driver='gesdd')


    try:
        U,S,Vh = scipy.linalg.svd(M,full_matrices=False,lapack_driver='gesdd')
    except:
        try:
            Vh,S,U = scipy.linalg.svd(np.transpose(M),full_matrices=False,lapack_driver='gesdd')
            U=np.transpose(U)
            Vh=np.transpose(Vh)
        except:
            U,S,Vh = scipy.linalg.svd(M,full_matrices=False,lapack_driver='gesvd')


    #relative norm calculated for cut evaluation
    if init_norm==True:
        norm_S=S/np.linalg.norm(S.reshape(-1,1),ord=norm_ord)
    else:
        norm_S=S
    norm_sum = 0
    i=0 #vfor last svd value kept index
    one_norm=1
    one_norms=[]
    #Evaluating final SVD value kept (index), for size limit fixed or not
    if limit_max==True:
        while norm_sum<(1-cut) and i<max  and i<S.size and one_norm>err_th:
            one_norm=np.power(norm_S[i],norm_ord)
            norm_sum+=one_norm
            one_norms.append(one_norm)
            i+=1
    else:
        while norm_sum<(1-cut)  and i<S.size and one_norm>err_th:
            one_norm=np.power(norm_S[i],norm_ord)
            norm_sum+=one_norm
            one_norms.append(one_norm)
            i+=1
    #nans=np.argwhere(np.isnan(S[:i]))
    #print(nans)
    if normalize==True:
        #Final renormalization of SVD values kept or not, returning the correct
        #matrices sizes
        S=(S[:i]/np.linalg.norm(S[:i].reshape(-1,1),ord=norm_ord))
        return U[:,:i], S, Vh[:i,:], i
    else:
        return U[:,:i], S[:i], Vh[:i,:], i

def qubit_MPS_build(phi,normalize=True,limit_max=False,max=100):
    #Takes a multi QUBIT state, outputs MPS
    MPS = []
    #We make the first MPS tensor
    phi=phi.reshape(2,-1) #correct reshape for qubit MPS
    U,S,Vh,_=reduced_SVD(phi, normalize=normalize,limit_max=limit_max,max=max)
    MPS.append(U) #Adding first tensor
    phi=np.dot(np.diag(S),Vh) #To arrive at an orthogonality center
    while phi.shape[1] > 2:
        link=phi.shape[0] #keep track of link index size
        phi=phi.reshape(2*link,-1) #correct reshape for qubit MPS
        U,S,Vh,_=reduced_SVD(phi, normalize=normalize,limit_max=limit_max,max=max)
        U=U.reshape(link,2,-1) #Getting back bit index
        MPS.append(U)
        phi=np.dot(np.diag(S),Vh) #For orthogonality center
    #We save the last MPS tensor, the orthogonality center
    MPS.append(phi)
    return MPS

def MPS_contract(MPS,renorm=False,norm_ord=2):
    #Contracts an MPS from MPS_build function, MPS must be of size 2 at least
    #Very unefficient exact way
    #Let's contract first two tensors
    vector=MPS[0]
    i=1
    #We do the contraction for each tensor between the first and last
    while i<=len(MPS)-2:
        vector_qubit_ind,link=vector.shape
        link_2,tens_qubit_ind,_=MPS[i].shape
        vector=np.dot(vector,MPS[i].reshape(link,-1))
        vector=vector.reshape(vector_qubit_ind*tens_qubit_ind,-1)
        i+=1
    #We do the last tensor contraction
    vector=np.dot(vector,MPS[i])
    #We reshape into the vector form
    vector=vector.reshape(1,-1)
    #Depending of choice renormalization of final vector
    #print(np.linalg.norm(vector.reshape(1,-1),ord=norm_ord))
    if renorm==True:
        vector=vector/np.linalg.norm(vector.reshape(-1,1),ord=norm_ord)
    return vector

def find_non_unit_ud(MPS,precision=1e-01):
    '''
    Finds the number of non-unit tensors in a MPS. Considers the orthog center
    on the last site as standard.
    '''
    ind=0
    non_unit=[]
    for i in range(0,len(MPS)):
        if i==0:
            test=np.tensordot(MPS[i],np.conj(MPS[i]),axes=([0],[0]))
            ind=1
        elif i==len(MPS)-1:
            test=np.tensordot(MPS[i],np.conj(MPS[i]),axes=([1],[1]))
            ind=0
        else:
            test=np.tensordot(MPS[i],np.conj(MPS[i]),axes=([0,1],[0,1]))
            ind=2
        size=MPS[i].shape[ind]
        id=np.identity(size)
        error=np.where(np.isclose(test, id, rtol=precision, atol=precision)==False)[0]
        if len(error)!=0:
            non_unit.append(i)

    return non_unit

def find_non_unit_du(MPS,precision=1e-01):
    '''
    Finds the number of non-unit tensors in a MPS. Considers the orthog center
    on the last site as standard.
    '''
    ind=0
    non_unit=[]
    for i in range(0,len(MPS)):
        if i==0:
            test=np.tensordot(MPS[i],np.conj(MPS[i]),axes=([0],[0]))
            ind=1
        elif i==len(MPS)-1:
            test=np.tensordot(MPS[i],np.conj(MPS[i]),axes=([1],[1]))
            ind=0
        else:
            test=np.tensordot(MPS[i],np.conj(MPS[i]),axes=([1,2],[1,2]))
            ind=0
        size=MPS[i].shape[ind]
        id=np.identity(size)
        error=np.where(np.isclose(test, id, rtol=precision, atol=precision)==False)[0]
        if len(error)!=0:
            non_unit.append(i)

    return non_unit

def find_non_unit(MPS,precision=1e-02):
    non_unit1=find_non_unit_ud(MPS,precision=precision)
    non_unit2=find_non_unit_du(MPS,precision=precision)

    return list(set(non_unit1) & set(non_unit2))



def FOTB_MPS(n):
    #for Full One Tensor Bits MPS
    #Takes number of bits as entry
    ones=np.ones(2)
    MPS=[]
    if n==1:
        MPS.append(ones)
    else:
        MPS.append(ones.reshape(2,1))
        ones=ones.reshape(1,2,1)
        [MPS.append(ones) for _ in range(n-2)]
        MPS.append(ones.reshape(1,2))
    return MPS

def move_orth(MPS,frm=0,to=-1,renorm=True):
    '''
    This simply moves the orth center from one position to the other calling the
    refresh functions
    '''
    if frm==-1 or frm==len(MPS)-1:
        if to!=len(MPS)-1 and to!=-1:
            MPS=refresh_MPS_du(MPS,begin=frm,orth_pos=to,renorm=renorm)
    elif to==-1 or to==len(MPS)-1:
        if frm!=len(MPS)-1 and frm!=-1:
            MPS=refresh_MPS_ud(MPS,begin=frm,orth_pos=to,renorm=renorm)
    elif frm<to:
        MPS=refresh_MPS_ud(MPS,begin=frm,orth_pos=to,renorm=renorm)
    elif to<frm:
        MPS=refresh_MPS_du(MPS,begin=frm,orth_pos=to,renorm=renorm)

    return MPS



def refresh_MPS_ud(MPS,begin=0,orth_pos=-1,renorm=False,ord=2):
    '''
    Retransfers all degrees of freedom on the orthogonality center on the last
    site of MPS. MPS size must be of size 3 at least. If orthogonality center on
    top, can be used to transfer it at orth_pos.
    '''
    if begin==0:
        bond_1,site_size_1,_=MPS[1].shape #saving relevant indices sizes
        temp=np.dot(MPS[0],MPS[1].reshape(bond_1,-1)) #Contracting tensors
        MPS[0],S,temp,_=reduced_SVD(temp) #SVD and saving first unitary
        temp=np.dot(np.diag(S),temp) #To generate orthogonality center
        MPS[1]=temp.reshape(len(S),site_size_1,-1) #Save temporary non-unit. tensor
        begin=1
    #Covering all elements but the first and last, or stop at orth_pos
    if orth_pos==-1 or orth_pos>=len(MPS)-2:
        upper=len(MPS)-2

    else:
        upper=orth_pos
    #Covering all but first and last, or stop at orth_pos
    i=begin
    while i>=begin and i<upper:
        #Saving relevant site and bond indices sizes
        bond_1,site_size_1,_=MPS[i].shape
        bond_2,site_size_2,_=MPS[i+1].shape
        #contraction
        temp=np.dot(MPS[i].reshape(bond_1*site_size_1,-1),
                    MPS[i+1].reshape(bond_2,-1))
        if i+1==orth_pos:
            U,S,temp,_=reduced_SVD(temp,normalize=renorm,norm_ord=ord)
        else:
            U,S,temp,_=reduced_SVD(temp)
        MPS[i]=U.reshape(bond_1,site_size_1,-1) #getting back init. shape
        temp=np.dot(np.diag(S),temp) #for orthog. center
        MPS[i+1]=temp.reshape(len(S),site_size_2,-1) #saving temp tensor
        i+=1
    #For the last site
    if orth_pos==-1 or orth_pos>=len(MPS)-1:
        bond_1,site_size_1,_=MPS[i].shape
        bond_2,site_size_2=MPS[i+1].shape
        temp=np.dot(MPS[i].reshape(bond_1*site_size_1,-1),
                    MPS[i+1].reshape(bond_2,-1))
        U,S,temp,_=reduced_SVD(temp,normalize=renorm,norm_ord=ord)
        MPS[i]=U.reshape(bond_1,site_size_1,-1)
        temp=np.dot(np.diag(S),temp)
        MPS[i+1]=temp.reshape(len(S),site_size_2) #saving last tensor

    return MPS

def refresh_MPS_du(MPS,begin=-1,orth_pos=0,renorm=False,ord=2):
    '''
    Same as refresh_MPS_up, but from bottom to top. Brings orth center at
    orth pos.
    '''
    end=len(MPS)-1
    lower=(end-begin)

    if begin==-1 or begin==end:
        _,site_size_1,bond_1=MPS[end-1].shape #saving relevant indices sizes
        temp=np.dot(MPS[end-1].reshape(-1,bond_1),MPS[end]) #Contracting tensors
        temp,S,MPS[end],_=reduced_SVD(temp) #SVD and saving first unitary
        temp=np.dot(temp,np.diag(S)) #To generate orthogonality center
        MPS[end-1]=temp.reshape(-1,site_size_1,len(S)) #Save temporary non-unit tensor
        lower=1
    #Setting loop limit for orth_pos
    if orth_pos<=1:
        upper=end-1
    else:
        upper=end-orth_pos
    #Covering all but first and last, or stop at orth_pos
    i=lower
    while i>=lower and i<upper:
        #Saving relevant site and bond indices sizes
        bond_1,site_size_1,_=MPS[end-i-1].shape
        bond_2,site_size_2,_=MPS[end-i].shape
        #contraction
        temp=np.dot(MPS[end-i-1].reshape(bond_1*site_size_1,-1),
                    MPS[end-i].reshape(bond_2,-1))
        if (end-i-1==orth_pos):
            temp,S,Vh,_=reduced_SVD(temp,normalize=renorm,norm_ord=ord)
        else:
            temp,S,Vh,_=reduced_SVD(temp)
        MPS[end-i]=Vh.reshape(len(S),site_size_2,-1) #getting back init. shape
        temp=np.dot(temp,np.diag(S)) #for orthog. center
        MPS[end-i-1]=temp.reshape(-1,site_size_1,len(S)) #saving temp tensor
        i+=1

    #For the first site
    if orth_pos==0:
        site_size_1,_=MPS[0].shape
        bond_2,site_size_2,_=MPS[1].shape
        temp=np.dot(MPS[0].reshape(site_size_1,-1),
                    MPS[1].reshape(bond_2,-1))
        temp,S,Vh,_=reduced_SVD(temp,normalize=renorm,norm_ord=ord)
        MPS[1]=Vh.reshape(len(S),site_size_2,-1) #getting back init. shape
        MPS[0]=np.dot(temp,np.diag(S)) #for orthog. center

    return MPS

def boltz_MPO(n,p=1/100):
    '''
    This return an MPO in the form of boltzmann probability boxes.
    '''
    #Initiate MPO list
    MPO=[]
    #Boltzmann Box
    boltz=np.array([[1-p,p],[p,1-p]])
    #Reshape for indices ordering
    #Saving first tensor
    MPO.append(boltz.reshape(2,2,1))
    #reshape for all successive tensors done only once
    boltz=boltz.reshape(2,2,1,1)
    #For all inbetween indices
    for _ in range(2,n):
        MPO.append(boltz)
    #Last tensor
    MPO.append(boltz.reshape(2,2,1))
    return MPO

def identity_MPO(n):
    '''
    This return an MPO in the form of identity. Useful to test MPS_MPO
    contraction
    '''
    #Initiate MPO list
    MPO=[]
    #Boltzmann Box
    boltz=np.array([[1,0],[0,1]])
    #Reshape for indices ordering
    #Saving first tensor
    MPO.append(boltz.reshape(2,2,1))
    #reshape for all successive tensors done only once
    boltz=boltz.reshape(2,2,1,1)
    #For all inbetween indices
    for _ in range(2,n):
        MPO.append(boltz)
    #Last tensor
    MPO.append(boltz.reshape(2,2,1))
    return MPO

def MPS_MPO(MPS, MPO, index=0, orthog=False, end_norm=True, ord=2):
    '''
    This contracts a partial MPO to an MPS and return the resulting MPS.
    The MPS and MPO must be of at least size 2.
    '''
    if index==0:
        #Contraction for specific case
        MPO_end_1=MPO[0].shape[1]
        MPO_end_2=MPO[1].shape[1]
        temp=np.tensordot(MPS[0],MPO[0],axes=([0],[0]))
        temp=np.tensordot(temp,MPS[1],axes=([0],[0]))
        temp=np.tensordot(temp,MPO[1],axes=([1,2],[2,0]))
        #Special case for MPO of 2 tensors only
        if len(MPO)==2:
            #Finish specific case
            #if contraction on last MPS tensor
            if (index+1)==(len(MPS)-1):
                MPS[0],S,Vh,_=reduced_SVD(temp.reshape(MPO_end_1,-1)
                                            ,normalize=end_norm,norm_ord=ord)
                temp=np.dot(np.diag(S),Vh)
                MPS[-1]=temp
            else:
                MPS[0],S,Vh,_=reduced_SVD(temp.reshape(MPO_end_1,-1))
                temp=np.dot(np.diag(S),Vh)
                _,_,bond_2=MPS[1].shape
                MPS[1]=(temp.reshape(-1,bond_2,MPO_end_2)).transpose(0,2,1)
        else:
            MPS[0],S,Vh,_=reduced_SVD(temp.reshape(MPO_end_1,-1))
            temp=np.dot(np.diag(S),Vh)
            _,_,bond_2=MPS[1].shape
            temp=temp.reshape(S.size,bond_2,MPO_end_2,-1)
    else:
        #General contraction for first MPO tensor
        bond_1,_,_=MPS[index].shape
        MPO_end_1=MPO[0].shape[1]
        MPO_end_2=MPO[1].shape[1]
        temp=np.tensordot(MPS[index],MPO[0],axes=([1],[0]))
        temp=np.tensordot(temp,MPS[index+1],axes=([1],[0]))
        temp=np.tensordot(temp,MPO[1],axes=([2,3],[2,0]))
        if len(MPO)==2:
            #Finish specific case
            #if contraction on last MPS tensor
            if (index+1)==(len(MPS)-1):
                U,S,Vh,_=reduced_SVD(temp.reshape(bond_1*MPO_end_1,-1)
                                        ,normalize=end_norm,norm_ord=ord)
                MPS[index]=U.reshape(bond_1,MPO_end_1,-1)
                temp=np.dot(np.diag(S),Vh)
                MPS[-1]=temp
            else:
                U,S,Vh,_=reduced_SVD(temp.reshape(bond_1*MPO_end_1,-1))
                MPS[index]=U.reshape(bond_1,MPO_end_1,-1)
                temp=np.dot(np.diag(S),Vh)
                _,_,bond_2=MPS[1].shape
                MPS[1]=(temp.reshape(-1,bond_2,MPO_end_2)).transpose(0,2,1)
        else:
            U,S,Vh,_=reduced_SVD(temp.reshape(bond_1*MPO_end_1,-1))
            MPS[index]=U.reshape(bond_1,MPO_end_1,-1)
            temp=np.dot(np.diag(S),Vh)
            _,_,bond_2=MPS[index+1].shape
            temp=temp.reshape(S.size,bond_2,MPO_end_2,-1)

    #We loop over all MPO elements but the first and two lasts
    #for specific cases, MPO has 2-3 elements,avoid the middle section contracts
    i=1
    if len(MPO)!=2:
        for i in range(2,len(MPO)-1):
            #General contraction plus retrieve
            bond_1,_,MPO_end_1,_=temp.shape
            _,_,bond_2=MPS[index+i].shape
            MPO_end_2=MPO[i].shape[1]
            temp=np.tensordot(temp,MPS[i+index],axes=([1],[0]))
            temp=np.tensordot(temp,MPO[i],axes=([2,3],[2,0]))
            U,S,Vh,_=reduced_SVD(temp.reshape(bond_1*MPO_end_1,-1))
            MPS[index+i-1]=U.reshape(bond_1,MPO_end_1,-1)
            temp=np.dot(np.diag(S),Vh)
            temp=temp.reshape(S.size,bond_2,MPO_end_2,-1)
        #If the last MPO elem is on the last MPS tensor
        if (i+index+1)==(len(MPS)-1):
            bond_1,_,MPO_end_1,_=temp.shape
            MPO_end_2=MPO[-1].shape[1]
            temp=np.tensordot(temp,MPS[-1],axes=([1],[0]))
            temp=np.tensordot(temp,MPO[-1],axes=([2,3],[2,0]))
            U,S,Vh,_=reduced_SVD(temp.reshape(bond_1*MPO_end_1,-1)
                                    ,normalize=end_norm,norm_ord=ord)
            MPS[i+index]=U.reshape(bond_1,MPO_end_1,-1)
            temp=np.dot(np.diag(S),Vh)
            MPS[-1]=temp.reshape(S.size,-1)
        #if not
        else:
            bond_1,_,MPO_end_1,_=temp.shape
            MPO_end_2=MPO[-1].shape[1]
            temp=np.tensordot(temp,MPS[i+index+1],axes=([1],[0]))
            temp=np.tensordot(temp,MPO[-1],axes=([2,3],[2,0]))
            U,S,Vh,_=reduced_SVD(temp.reshape(bond_1*MPO_end_1,-1))
            MPS[i+index]=U.reshape(bond_1,MPO_end_1,-1)
            temp=np.dot(np.diag(S),Vh)
            temp=temp.reshape(S.size,-1,MPO_end_2)
            MPS[i+index+1]=temp.transpose(0,2,1)

    #transfering orthogonality center to end if asked
    if orthog==True and i+index+1<len(MPS)-1:
        j=i+index
        for j in range(i+index,len(MPS)-2):
            #Saving relevant site and bond indices sizes
            bond_1,site_size_1,_=MPS[j].shape
            bond_2,site_size_2,_=MPS[j+1].shape
            #contraction
            temp=np.dot(MPS[j].reshape(bond_1*site_size_1,-1),
                        MPS[j+1].reshape(bond_2,-1))
            U,S,temp,_=reduced_SVD(temp)
            MPS[j]=U.reshape(bond_1,site_size_1,-1) #getting back init. shape
            temp=np.dot(np.diag(S),temp) #for orthog. center
            MPS[j+1]=temp.reshape(len(S),site_size_2,-1) #saving temp tensor
        j+=1 #cover last two tensor contraction
        #For the last site
        bond_1,site_size_1,_=MPS[j].shape
        bond_2,site_size_2=MPS[j+1].shape
        temp=np.dot(MPS[j].reshape(bond_1*site_size_1,-1),
                    MPS[j+1].reshape(bond_2,-1))
        U,S,temp,_=reduced_SVD(temp,normalize=end_norm,norm_ord=ord)
        MPS[j]=U.reshape(bond_1,site_size_1,-1)
        temp=np.dot(np.diag(S),temp)
        MPS[j+1]=temp.reshape(len(S),site_size_2) #saving last tensor

    return MPS

def binary_MPS(binary):
    '''
    Turns a classical binary array into a an equivalent normalised MPS.
    Could maybe be optimized by eliminating if conditions, replacing zero and
    one with matrix only.
    '''
    #basic vector elements association
    zero=np.array([1., 0.])
    one=np.array([0., 1.])
    #begin MPS list
    MPS=[]
    #First element
    if binary[0]==0:
        MPS.append(zero.reshape(2,1)) #right shape
    else:
        MPS.append(one.reshape(2,1))
    #Doing reshape for all inbetweens only once
    zero=zero.reshape(1,2,1)
    one=one.reshape(1,2,1)
    #all elements but first and last
    #special case for binary of size 2
    i=0
    for i in range(1,len(binary)-1):
        if binary[i]==0:
            MPS.append(zero)
        else:
            MPS.append(one)
    #for last element
    if binary[i+1]==0:
        MPS.append(zero.reshape(1,2))
    else:
        MPS.append(one.reshape(1,2))

    return MPS

def par_4t2():
    reverse_id=np.array([[0.,1.],[1.,0.]])
    xor_3t2=np.zeros((2,2,2))
    xor_3t2[0,:,:]=reverse_id
    xor_3t2[1,:,:]=np.identity(2)

    delta=np.zeros((2,2,2))
    delta[0,:,:]=np.array([[1.,0.],[0.,0.]])
    delta[1,:,:]=np.array([[0.,0.],[0.,1.]])

    par=np.tensordot(delta,xor_3t2,axes=([2],[0]))

    return par

def parity_TN(array,par=par_4t2(),adjacency=False):
    cross=np.tensordot(np.identity(2),np.identity(2),axes=0)
    if adjacency==True:
        nonzeros=np.nonzero(array)[1]
    else:
        nonzeros=np.array(array)
    MPO=[]
    #We save the first element of the MPO, index fixed linked to #tens. parity
    if nonzeros.size%2==0:
        MPO.append(par[:,:,0,:])
    else:
        MPO.append(par[:,:,1,:])
    for i in range(nonzeros[0]+1,nonzeros[-1]):
        if i in nonzeros:
            MPO.append(par)
        else:
            MPO.append(cross)
    #We save the last tensor
    MPO.append(par[:,:,:,0])

    return MPO, nonzeros[0], nonzeros[-1]

def max_bond_size(MPS):
    j=0
    for tensor in MPS:
        for i in tensor.shape:
            if i > j:
                j=i
    return j




if __name__ == "__main__":


    phi = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #phi = np.array([0,0,0])
    MPS=binary_MPS(phi)


    MPO=identity_MPO(14)

    for i in range(0,len(MPO)):
        if i%2==1:
            shape=MPO[i].shape
            MPO[i]=np.ones((2,2)).reshape(shape)-MPO[i]
