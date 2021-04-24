import numpy as np
import mpopt.contractions.TNTools as tnt


def test_state_to_mps():
    '''
    Builds a random vector mps and checks if the contraction of mps returns
    right vector.
    '''
    phi0 = np.random.rand(2**10)
    phi0 = phi0/np.linalg.norm(phi0, ord=2)

    mps = tnt.state_to_mps_build(phi0)
    phi1 = tnt.mps_contract(mps)
    print(phi1)

    return np.testing.assert_allclose(phi0, phi1)
