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

    np.testing.assert_allclose(phi0, phi1)


def test_mpsbuild_singlesize(bit_numb=10, qudit_lvl=2):
    '''
    Verifies that the mps is built with right length.
    '''

    phi = np.random.rand(qudit_lvl**bit_numb)
    phi = phi/np.linalg.norm(phi, ord=2)

    an_mps = tnt.state_to_mps_build(phi, qudit_level=qudit_lvl)
    assert len(an_mps) == bit_numb


def test_mpsbuild_listssize(bit_nums=[2, 5, 10], qudit_lvls=[2, 3]):
    '''
    Verifies calls the test_mpsbuild_single multiple times.
    '''
    for bit_numb in bit_nums:
        for qudit_lvl in qudit_lvls:
            test_mpsbuild_singlesize(
                bit_numb=bit_numb, qudit_lvl=qudit_lvl)


def test_move_orthog(bit_numb=10, qudit_lvl=2, orth_pos=0):
    '''
    Verifies the moving of an orthogonality tensor in a mps.
    '''

    phi = np.random.rand(qudit_lvl**bit_numb)
    phi = phi/np.linalg.norm(phi, ord=2)

    an_mps = tnt.state_to_mps_build(phi, qudit_level=qudit_lvl)
    an_mps = tnt.move_orthog(an_mps, begin=-1, end=0)
    an_mps = tnt.move_orthog(an_mps, begin=0, end=orth_pos)

    orth_poses = tnt.find_orthog_center(an_mps)

    assert len(orth_poses) == 1
    assert orth_poses[0] == orth_pos


def test_moveorthog_lists(bit_nums=[2, 5, 10], qudit_lvls=[2, 3]):
    '''
    Verifies the moving of an orthogonality tensor in a mps. Does it for random
    positions for diffrent length of mps and qudit levels.
    '''
    for bit_numb in bit_nums:
        for qudit_lvl in qudit_lvls:
            orth_pos = np.random.randint(0, bit_numb-1)
            print(orth_pos)
            test_move_orthog(bit_numb=bit_numb,
                             qudit_lvl=qudit_lvl, orth_pos=orth_pos)


def test_mps_mpo_contraction(bit_numb=10, qudit_lvl=2):
    '''
    Tests the mps-mpo contraction by watching the movement 
    '''
    # Initiate mps and random size identity-mpo
    phi = np.random.rand(qudit_lvl**bit_numb)
    phi = phi/np.linalg.norm(phi, ord=2)
    an_mps = tnt.state_to_mps_build(phi)
    length_mpo = np.random.randint(2, bit_numb)
    an_mpo = tnt.identity_mpo(length_mpo, qudit=qudit_lvl)

    # Assign orthog center at random position
    orth_pos = np.random.randint(0, bit_numb-1)
    an_mps = tnt.move_orthog(an_mps, begin=-1, end=orth_pos)

    orthogonality = tnt.find_orthog_center(an_mps)[0]

    index = np.random.randint(0, bit_numb-len(an_mpo))
    an_mps, new_orthog = tnt.mps_mpo_contract_shortest_moves(
        an_mps, an_mpo, index=index, current_orth=orthogonality)

    orthogonality = tnt.find_orthog_center(an_mps)[0]
    assert new_orthog == orthogonality
    vector = tnt.mps_contract(an_mps)

    np.testing.assert_allclose(phi, vector)


