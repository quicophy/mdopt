'''
Unfinished file containing an incomplete version of the decoder_state class and
LDPC_code class using pyqec.

Has to be completed for the full contraction sequence of the decoding process.
Has to be updated and made fuctionnal for pyqec's latest version.
Has to be commented.
Has to be made testable.


'''


from TN_tools import *
import matplotlib.pyplot as plt
from pyqec.sparse import BinaryMatrix, BinaryVector
from pyqec.classical import LinearCode, BinarySymmetricChannel, random_regular_code
from pyqec.experiments import ClassicalDecodingExperiment, Laboratory


def main_component(MPS,method='normed_brute'):
    max_pos=-1
    if method == 'DMRG':
        raise ValueError('Not coded yet')
    elif method == 'normed_brute':
        vector=MPS_contract(MPSs[i],renorm=True)
        if vector[0,0]>=0.71:
            max_pos=0
        else:
            max_pos=np.argmax(vector)

    return max_pos


def LDPC_code(n,bit_degree=3,check_degree=4,rand_id=0):
    #creates a code class object using the parity check matrix of random codes
    length=n*check_degree
    parity_check_matrix=random_regular_code(
        block_size=length,
        number_of_checks=bit_degree*n,
        bit_degree=bit_degree,
        check_degree=check_degree).parity_check_matrix()
    #Eventually the bandwidth reduction would come here
    return LinearCode(parity_check_matrix,
        tag=f"length = {length}, bd = {bit_degree}, cd = {check_degree}, id={rand_id}")

def bin_vector_to_array(bin_v):
    #Transform Max's bin vect into numpy array
    length=len(bin_v)
    bin_array=np.zeros(length)
    for i in range(0,length):
        print(i)
        bin_array[i]=bin_v[i]
    return bin_array


class ClassicalTN_Decoder:
    #add the decoder parameters, cut ratios, boltz_prob, etc...
    def __init__(self, code, bprob, cut=0.0, max=1000, limit_max=False):
        self.code = code
        self.length = len(code)
        self.bprob=bprob
        self.cut=cut
        self.max=max
        self.limit_max=limit_max
        self.MPS=[]

    #what should be quantities to be calculated
    def decode(self, message):
        #message is a bin-vector of all flipped bits
        self.MPS=binary_MPS(message)
        #Boltzmann weight smulates noise-model
        MPO=boltz_MPO(self.length,p=self.bprob)
        self.MPS=MPS_MPO(self.MPS,MPO,index=0)
        #applying list of parity checks


        print(self.MPS)

        output=[0,1]
        #The output is also a vector of flipped bits
        #return BinaryVector(self.length, list(output))
        return BinaryVector(self.length, list(output))






id=0
def test_experiment(n, probability,bprob,id):
    code = LDPC_code(n,rand_id=id)
    decoder = ClassicalTN_Decoder(code,bprob)
    noise = BinarySymmetricChannel(probability)
    return ClassicalDecodingExperiment(code, decoder, noise)

laboratory = Laboratory(2)

laboratory.add_experiment(test_experiment(3, 0.1, 0.1, id))
#id+=1
#laboratory.add_experiment(test_experiment(3, 0.1,id))



results = laboratory.run_all_experiments_n_times(2)
'''
print(results.tags)
print(results.probabilities)
print(results.statistics[0].failure_rate())

results.plot()
plt.show()
'''
yup=LDPC_code(4)
