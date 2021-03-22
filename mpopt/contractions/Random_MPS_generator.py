'''
12/02/2020 by Samuel Desrosiers
Quick program to build test MPSs for the DMRG main component solver.
Noise used is gaussian.
'''


from TN_tools import *


def random_MPS_gen(n=8,noise_ratio=0.1,highest_value_index=1,limit_max=True,max=100):
    #Vector for the most likely element
    most_likely=np.zeros(2**n)
    most_likely[highest_value_index]=1

    #vector of the noise model
    noise=noise_ratio*np.random.rand(2**n)
    vector=most_likely+noise

    #We can refer to my MPS builder for qubit states
    MPS=qubit_MPS_build(vector,normalize=True,limit_max=limit_max,max=max)

    return MPS


if __name__ == "__main__":


    nb_MPS=100 #number of MPS per file
    Sizes=[8,16] #list of num of sites for each file
    Ratios=[0.1,0.2] #error ratios ([0,1[)
    Max_Bonds=[50,30,20,10] #Max bond dimension
    Highest_vals_ind=[1] #Highest value element index

    for s in Sizes:
        for r in Ratios:
            for mb in Max_Bonds:
                for high in Highest_vals_ind:
                    MPSs=[] #empty list or np arrays
                    #Print current file name in terminal
                    print('MPS_maxat'+str(high)+"_"+str(r)+"erroratio_"+str(s)+"sites_"+str(mb)+"maxbondsize")
                    for _ in range(0,nb_MPS):
                        #Obtains one instance
                        MPS=random_MPS_gen(n=s, noise_ratio=r, highest_value_index=high, limit_max=True,max=mb)
                        MPSs.append(MPS) #Add to list
                    #Saving the list of np.arrays
                    np.save('MPS_maxat'+str(high)+"_"+str(r)+"erroratio_"+str(s)+"sites_"+str(mb)+"maxbondsize.npy",MPSs)

    print('All MPS reporting. Ready for duty!')
