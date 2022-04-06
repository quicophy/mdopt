"""
In this experiment, we decode a classical error correction code.
First, we build the codeword MPS and test it against the dense form.
Then, we demostrate simple decoding of a classical LDPC code with DMRG.
The script should be launched from the root of the project directory.
"""

import sys
import numpy as np
import qecstruct as qec

sys.path[0] += "/.."

from mpopt.mps.explicit import create_product_state, create_custom_product_state
from mpopt.mps.canonical import (
    to_dense,
    move_orth_centre,
    find_orth_centre,
    is_canonical,
    to_explicit,
    inner_product,
)
from mpopt.optimizer import DMRG as dmrg
from mpopt.utils.utils import mpo_to_matrix
from mpopt.contractor.contractor import mps_mpo_contract


class ConstraintString:
    """
    Class for storing a string of logical constraints in the MPO format.
    Logical constraints are passed in the form of Matrix Product Operators.

    Attributes:
        constraints : list
            A list of logical constraints of which the string consists.
        sites : list of lists
            Each list inside corresponds to a constraint from the `constraints` list,
            and contains the sites to which each constraint is applied.
            For example, [[3, 5], [2, 4, 6], ...] means applying
            `constraints[0]` to sites 3 and 5, `constraints[1]` to sites 2, 4, 6, etc.

    Exceptions:
        ValueError:
            Empty list of constraints.
        ValueError:
            Empty list of sites.
        ValueError:
            The `sites` list is longer than the `constraints` list.
        ValueError:
            Non-unique sites in the `sites` list.
    """

    def __init__(self, constraints, sites):
        self.constraints = constraints
        self.sites = sites

        if self.constraints == []:
            raise ValueError("Empty list of constraints passed.")

        if self.sites == []:
            raise ValueError("Empty list of sites passed.")

        if len(self.sites) > len(self.constraints):
            raise ValueError(
                f"We have ({len(self.constraints)}) constraints in the constraints list, "
                f"({len(self.sites)}) constraints assumed by the sites list."
            )

        seen = set()
        uniq = [site for site in self.flat() if site not in seen and not seen.add(site)]
        if uniq != self.flat():
            raise ValueError("Non-unique sites encountered in the list.")

    def __getitem__(self, site):
        """
        Returns the constraint applied at site `site`
        in the format of a tensor as a np.array.

        Arguments:
            site : int
                The site index.
        """

        index = np.where(np.array(self.sites) == site)[0]
        return self.constraints[index]

    def flat(self):
        """
        Returns a flattened list of sites.
        """

        return [item for sublist in self.sites for item in sublist]

    def span(self):
        """
        Returns the span (length) of the constraint string.
        """

        return max(self.flat()) - min(self.flat()) + 1

    def get_mpo(self):
        """
        Returns the constraint string in the MPO format.
        Note, that it will not include identities, which means
        it needs to be manually adjusted to a corresponding MPS,
        as the MPO can be smaller in size.
        """

        mpo = [None for _ in range(self.span())]
        for index, sites_sublist in enumerate(self.sites):
            for site in sites_sublist:
                mpo[site - min(self.flat())] = self.constraints[index]

        return mpo


# Let's define the noise model -- a binary symmetric bit flip channel.
# Simply speaking, it flips the bit with probability `p` and leaves it
# the same with probability `1-p`.

# TODO this exists in qecstruct (find out if there's a square root)
def binary_symmetric_channel(prob):
    """
    This function returns a single-site MPO,
    corresponding to a binary symmetric channel.

    Arguments:
        prob : float
            Probability of the binary symmetric channel.
    """

    assert 0 <= prob <= 1
    return np.sqrt(np.array([[1 - prob, prob], [prob, 1 - prob]])).reshape((1, 1, 2, 2))


# Below, we define some utility functions to operate with error correction codes from qecstruct.
# Co-authored by Stefanos Kourtis.

# TODO exists in qecstruct
def linear_code_checks(code):
    """
    Given a linear code, returns a list of its checks, where each check
    is represented as a list of indices of the bits adjacent to it.

    Arguments:
        code : qecstruct.LinearCode
            Linear code object.

    Returns:
        checks : list of lists
            List of checks.
    """

    checks = []
    bits = range(code.length())
    parity_matrix = code.par_mat()

    for row in parity_matrix.rows():
        checks.append(np.flatnonzero([row.is_one_at(i) for i in bits]))

    return checks


# TODO exists in qecstruct
def get_codewords(code):
    """
    Return the list of codewords of a linear code. Codewords are returned
    as integers in most-significant-bit-first convention.

    Arguments:
        code : qecstruct.LinearCode
            Linear code object.

    Returns:
        codewords : list of ints
            List of codewords.
    """

    length = code.length()
    codewords = []

    for word in range(2 ** length):
        msg = np.array(list(np.binary_repr(word, length)), int)
        vec = qec.BinaryVector(length, list(np.flatnonzero(msg)))
        if code.has_codeword(vec):
            codewords.append(word)

    return np.array(codewords)


if __name__ == "__main__":

    # Here, we define the tensors we use to represent a code.
    # We use the following tensors: XOR, SWAP, IDENTITY.
    # See the notes for additional information.

    # According to our convention, each tensor has legs (vL, vR, pU, pD),
    # where v stands for "virtual", p -- for "physical",
    # and L, R, U, D stand for "left", "right", "up", "down".

    IDENTITY = np.eye(2).reshape((1, 1, 2, 2))
    # TODO ravel unravel for the loops
    XOR_BULK = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    XOR_BULK[i, j, k, l] = (i ^ j ^ k ^ 1) * np.eye(2)[k, l]

    XOR_LEFT = np.zeros((1, 2, 2, 2))
    for j in range(2):
        for k in range(2):
            for l in range(2):
                XOR_LEFT[0, j, k, l] = np.eye(2)[j, k] * np.eye(2)[k, l]

    XOR_RIGHT = np.zeros((2, 1, 2, 2))
    for i in range(2):
        for k in range(2):
            for l in range(2):
                XOR_RIGHT[i, 0, k, l] = np.eye(2)[i, k] * np.eye(2)[k, l]

    SWAP = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    SWAP[i, j, k, l] = np.eye(2)[i, j] * np.eye(2)[k, l]

    # Defining the parameters of a classical LDPC code.
    NUM_BITS = 12
    NUM_CHECKS = 9
    BIT_DEGREE = 3
    CHECK_DEGREE = 4
    # TODO prob_noise -> prob_channel
    PROB_NOISE = 0.2
    PROB_ERROR = 0.1
    if NUM_BITS / NUM_CHECKS != CHECK_DEGREE / BIT_DEGREE:
        raise ValueError("The graph must be bipartite.")

    # Constructing the code as a qecstruct object, preparing the state.
    example_code = qec.random_regular_code(
        NUM_BITS, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qec.Rng()
    )

    state = create_product_state(NUM_BITS, which="0").to_right_canonical()

    channel = [binary_symmetric_channel(PROB_NOISE) for _ in range(NUM_BITS)]
    state = mps_mpo_contract(state, channel, 0)
    state_dense = to_dense(state)

    tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]

    # Getting the sites for which the constraints should be applied.
    sites_all = linear_code_checks(example_code)

    # Rebuilding the sites lists as used in the ConstrainedString class.
    strings = []
    for _, sublist in enumerate(sites_all):

        # Retreiving the sites indices where we apply the "bulk"/"boundary" XOR tensors.
        xor_left_sites = [sublist[0]]
        xor_bulk_sites = [sublist[i] for i in range(1, CHECK_DEGREE - 1)]
        xor_right_sites = [sublist[-1]]

        # Retreiving the sites indices where we apply the SWAP tensors.
        swap_sites = list(range(sublist[0] + 1, sublist[-1]))
        for k in range(1, CHECK_DEGREE - 1):
            swap_sites.remove(sublist[k])

        strings.append([xor_left_sites, xor_bulk_sites, swap_sites, xor_right_sites])

    print(
        "__________________________________________________________________________________________"
    )
    print("")
    print("Checking the codeword superposition state: ")

    # Preparing the codeword superposition by MPS-MPO evolutions and also in the dense form.
    for i in range(NUM_CHECKS):

        # Checking the orthogonality conditions.
        assert is_canonical(state)

        # Finding the orthogonality centre.
        orth_centre_init = find_orth_centre(state)[0]

        # Preparing the MPO.
        constraint_string = ConstraintString(tensors, strings[i])
        constraint_mpo = constraint_string.get_mpo()

        # Finding the starting site of the MPS to perform contraction.
        START_SITE = min(constraint_string.flat())

        # Preparing the dense form.
        identities_l = [IDENTITY for _ in range(START_SITE)]
        identities_r = [
            IDENTITY for _ in range(NUM_BITS - len(constraint_mpo) - START_SITE)
        ]
        full_mpo = identities_l + constraint_mpo + identities_r
        mpo_dense = mpo_to_matrix(full_mpo, interlace=False, group=True)

        # Doing the contraction.
        state = move_orth_centre(state, orth_centre_init, START_SITE)
        state = mps_mpo_contract(state, constraint_mpo, START_SITE)

        # Doing the contraction in dense form.
        state_dense = mpo_dense @ state_dense

    # Tolerance under which we round tensor elements to zero.
    TOL = 1e-15
    mps_dense = to_dense(state)
    mps_dense[np.abs(mps_dense) < TOL] = 0

    # Retreiving codewords
    words = np.array(get_codewords(example_code))
    words_to_compare_mps = np.flatnonzero(mps_dense)
    words_to_compare_dense = np.flatnonzero(state_dense)

    print("Codewords from exhaustive search:")
    print(words)
    print("Codewords from the dense form simulation:")
    print(words_to_compare_mps)
    print("Codewords from the MPS form simulation:")
    print(words_to_compare_dense)
    print(
        "All codewords' lists match:",
        np.logical_and(
            (words == words_to_compare_mps).all(),
            (words_to_compare_mps == words_to_compare_dense).all(),
        ),
    )
    print(
        "__________________________________________________________________________________________"
    )

    print("")
    print("Retreiving a perturbed codeword: ")

    # Building an initial and a perturbed codeword.
    # TODO start from random codeword
    # TODO the error model exists in qecstruct
    INITIAL_STRING = "0" * NUM_BITS
    PERTURBED_STRING = ""
    for i in range(NUM_BITS):
        if np.random.uniform() < PROB_ERROR:
            PERTURBED_STRING += str(bool(list(INITIAL_STRING)[i]) ^ 1)
        else:
            PERTURBED_STRING += list(INITIAL_STRING)[i]

    # Building the corresponding MPSs.
    codeword = create_custom_product_state(INITIAL_STRING).to_right_canonical()
    perturbed_codeword = create_custom_product_state(
        PERTURBED_STRING
    ).to_right_canonical()

    # Passing the perturbed codeword state through the bitflip channel
    channel = [binary_symmetric_channel(PROB_NOISE) for _ in range(NUM_BITS)]
    perturbed_codeword = mps_mpo_contract(perturbed_codeword, channel, 0)

    # Passing the perturbed codeword state through the parity constraints defined by the code
    for i in range(NUM_CHECKS):

        orth_centre_init = find_orth_centre(perturbed_codeword)[0]

        constraint_string = ConstraintString(tensors, strings[i])
        constraint_mpo = constraint_string.get_mpo()

        START_SITE = min(constraint_string.flat())

        perturbed_codeword = move_orth_centre(
            perturbed_codeword, orth_centre_init, START_SITE
        )
        perturbed_codeword = mps_mpo_contract(
            perturbed_codeword, constraint_mpo, START_SITE
        )

    # TODO avoid to_expl form conversion, use interlace_tensor
    # Building the density matrix MPO.
    density_mpo = to_explicit(state).density_mpo()

    print("DMRG running:")
    mps_dmrg_start = create_product_state(len(density_mpo), which="0")
    # TODO start with random product state
    engine = dmrg(mps_dmrg_start, density_mpo, chi_max=128, cut=1e-14, mode="LA")
    engine.run(20)
    mps_dmrg_final = engine.mps.to_right_canonical()

    print(
        "The overlap of the density MPO main component and the initial codeword: ",
        inner_product(mps_dmrg_final, codeword),
    )
    print(
        "__________________________________________________________________________________________"
    )
