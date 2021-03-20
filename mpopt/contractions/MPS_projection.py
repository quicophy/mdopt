from TN_tools import *
import copy
import glob
import matplotlib.pyplot as plt



def MPS_MPS_contract(MPS01,MPS02,frm=0, to=-1, reformat=True):
    #Contracts parts of two MPS of similar dimensions as to create left
    # environment or scalar
    MPS1=copy.deepcopy(MPS01)
    MPS2=copy.deepcopy(MPS02)

    if reformat==True:
        #We reshape MPS tensors to uniformise format of env
        d1,d2=MPS1[0].shape
        MPS1[0]=MPS1[0].reshape(1,d1,d2)
        d1,d2=MPS2[0].shape
        MPS2[0]=MPS2[0].reshape(1,d1,d2)
        d1,d2=MPS1[-1].shape
        MPS1[-1]=MPS1[-1].reshape(d1,d2,1)
        d1,d2=MPS2[-1].shape
        MPS2[-1]=MPS2[-1].reshape(d1,d2,1)

    if to<0:
        to=len(MPS1)+to
    if frm<0:
        frm=len(MPS1)+frm

    if frm<to:
        #contract left to right
        env=np.tensordot(MPS1[frm],MPS2[frm],axes=([1],[1])).transpose(0,2,1,3)
        for i in range(frm+1,to+1):
            temp=np.tensordot(MPS1[i],MPS2[i],axes=([1],[1])).transpose(0,2,1,3)
            env=np.tensordot(env,temp,axes=([2,3],[0,1]))
    elif to<frm:
        #right to left
        env=np.tensordot(MPS1[frm],MPS2[frm],axes=([1],[1])).transpose(0,2,1,3)
        for i in range(1,frm-to+1):
            temp=np.tensordot(MPS1[frm-i],MPS2[frm-i],axes=([1],[1])).transpose(0,2,1,3)
            env=np.tensordot(temp,env,axes=([2,3],[0,1]))
    else:
        env=np.tensordot(MPS1[frm],MPS2[frm],axes=([1],[1])).transpose(0,2,1,3)

    return env



def project_basis(basis_word,MPS,orthog=2):
    word_MPS=binary_MPS(basis_word)
    temp=np.tensordot(MPS[orthog],MPS[orthog+1],axes=([2],[0]))
    i1,i2,i3,i4=temp.shape
    temp=temp.reshape(i1*i2,i3*i4)
    U,S,Vh = scipy.linalg.svd(temp,full_matrices=False,lapack_driver='gesdd')
    _,_,_,i=reduced_SVD(temp)
    MPS0=copy.deepcopy(MPS)
    MPS0[orthog]=U.reshape(i1,i2,-1)
    MPS0[orthog+1]=Vh.reshape(-1,i3,i4)
    env1=MPS_MPS_contract(MPS0,word_MPS,frm=0, to=orthog)
    env2=MPS_MPS_contract(MPS0,word_MPS,frm=-1, to=orthog+1)

    proj_matr=np.tensordot(env1,env2,axes=([0,1,3],[2,3,1]))

    projection=np.diagonal(proj_matr)

    return projection, S, i


for filename in glob.glob('time_MPS/*.npy'):
    print(filename)
    time_MPS=np.load(filename,allow_pickle=True)
    times=len(time_MPS)
    MPS_size=len(time_MPS[0])
    basis_word=np.zeros(MPS_size)
    for i in range(0,times):
        print(i)
        proj,S,cut=project_basis(basis_word,time_MPS[i],orthog=int(MPS_size/2))
        S=np.square(S)
        S=S/np.sum(S)
        projection=np.square(proj)
        projection=projection/np.sum(projection)


        fig = plt.figure()

        ax = plt.subplot()
        plt.title("timestep_no_"+str(i+1))
        ax.plot(S,color="k",label='Global State')
        ax.plot(projection,color="orange",label='Projection')
        ax.axvline(cut,color="r",linestyle="--",ymax=0.4,label='Cut')
        ax.legend()
        plt.savefig(filename+"timestep_no_"+str(i+1)+".pdf")
        plt.close()
