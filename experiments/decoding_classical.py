"""
In this experiment, we decode a classical error correction code.
First, we build the codeword MPS and test it against the dense form.
Then, we demostrate simple decoding of a classical LDPC code with DMRG.
The script should be launched from the root of the project directory.
Note, this is supposed to run for about 10 minutes!
"""

import sys

from functools import reduce
from more_itertools import powerset

import numpy as np
import qecstruct as qec

sys.path[0] += "/.."

from mpopt.mps.explicit import create_simple_product_state, create_custom_product_state
from mpopt.mps.canonical import (
    to_dense,
    move_orth_centre,
    find_orth_centre,
    is_canonical,
    inner_product,
    to_density_mpo,
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
        return [index, self.constraints[index]]

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


# Here, we define a bias channel -- the operator which will bias us towards the initial message
# by ranging the bitstrings according to Hamming distance from it.
# For this role we employ binary symmetric bit flip channel --
# it flips the bit with probability `p` and leaves it the same with probability `1-p`.


def bias_channel(p_bias):
    """
    This function returns a single-site MPO,
    corresponding to a bias channel.

    Arguments:
        p_bias : float
            Probability of the channel.
    """

    assert 0 <= p_bias <= 1
    return np.sqrt(np.array([[1 - p_bias, p_bias], [p_bias, 1 - p_bias]])).reshape(
        (1, 1, 2, 2)
    )


# Below, we define some utility functions to operate with data structures from `qecstruct` --
# an error-correction library we are using in this example.


def bin_vec_to_dense(vector):
    """
    Given a vector (1D array) in the `BinaryVector` format (native to `qecstruct`),
    return its dense representation.

    Arguments:
        vector : qecstruct.sparse.BinaryVector
            Vector object.

    Returns:
        array : np.array
            The dense representation.
    """

    array = np.zeros(vector.len(), dtype=np.int32)
    for pos in vector:
        array[pos] = 1
    return array


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

    parity_matrix = code.par_mat()
    array = np.zeros(
        (parity_matrix.num_rows(), parity_matrix.num_columns()), dtype=np.int32
    )
    for row, cols in enumerate(parity_matrix.rows()):
        for col in cols:
            array[row, col] = 1
    return [np.nonzero(row)[0] for row in array]


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

    codewords = []

    gen_mat = code.gen_mat()
    rows_bin = gen_mat.rows()
    rows_dense = [bin_vec_to_dense(row_bin) for row_bin in rows_bin]
    rows_int = [row.dot(1 << np.arange(row.size)[::-1]) for row in rows_dense]

    # Append the all-zeros codeword which is always a codeword.
    codewords.append(0)

    # Append the rows of the generator matrix.
    for basis_codeword in rows_int:
        codewords.append(basis_codeword)

    # Append all linear combinations.
    for generators in powerset(rows_int):
        if len(generators) > 1:
            codewords.append(reduce(np.bitwise_xor, generators))

    return np.sort(np.array(codewords))


if __name__ == "__main__":

    # Fixing a random seed
    SEED = 123

    # Here, we define the tensors which represent logical constraints.
    # We use the following tensors: XOR, SWAP, IDENTITY.
    # See the notes for additional information.

    # According to our convention, each tensor has legs (vL, vR, pU, pD),
    # where v stands for "virtual", p -- for "physical",
    # and L, R, U, D stand for "left", "right", "up", "down".

    IDENTITY = np.eye(2).reshape((1, 1, 2, 2))
    XOR_BULK = np.fromfunction(
        lambda i, j, k, l: (i ^ j ^ k ^ 1) * np.eye(2)[k, l],
        (2, 2, 2, 2),
        dtype=np.int32,
    )
    XOR_LEFT = np.fromfunction(
        lambda i, j, k, l: np.eye(2)[j, k] * np.eye(2)[k, l],
        (1, 2, 2, 2),
        dtype=np.int32,
    )
    XOR_RIGHT = np.fromfunction(
        lambda i, j, k, l: np.eye(2)[i, k] * np.eye(2)[k, l],
        (2, 1, 2, 2),
        dtype=np.int32,
    )
    SWAP = np.fromfunction(
        lambda i, j, k, l: np.eye(2)[i, j] * np.eye(2)[k, l],
        (2, 2, 2, 2),
        dtype=np.int32,
    )
    tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]

    # Defining the parameters of a classical LDPC code.
    NUM_BITS = 12
    NUM_CHECKS = 9
    BIT_DEGREE = 3
    CHECK_DEGREE = 4
    PROB_CHANNEL = 0.2
    PROB_ERROR = 0.2
    if NUM_BITS / NUM_CHECKS != CHECK_DEGREE / BIT_DEGREE:
        raise ValueError("The graph must be bipartite.")

    # Constructing the code as a qecstruct object.
    example_code = qec.random_regular_code(
        NUM_BITS, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qec.Rng(SEED)
    )

    # Preparing an initial state.
    state = create_simple_product_state(NUM_BITS, which="0").to_right_canonical()

    channel = [bias_channel(PROB_CHANNEL) for _ in range(NUM_BITS)]
    state = mps_mpo_contract(state, channel, 0)
    state_dense = to_dense(state)

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
    TOL = 1e-14
    mps_dense = to_dense(state)
    mps_dense[np.abs(mps_dense) < TOL] = 0

    # Retreiving codewords.
    cwords = get_codewords(example_code)
    cwords_to_compare_mps = np.flatnonzero(mps_dense)
    cwords_to_compare_dense = np.flatnonzero(state_dense)

    print()
    print("Codewords from the generator matrix:")
    print(cwords)
    print("Codewords from the dense-form simulation:")
    print(cwords_to_compare_mps)
    print("Codewords from the MPS-form simulation:")
    print(cwords_to_compare_dense)
    print("")
    print(
        "All lists of codewords match:",
        np.logical_and(
            (cwords == cwords_to_compare_mps).all(),
            (cwords_to_compare_mps == cwords_to_compare_dense).all(),
        ),
    )
    print(
        "__________________________________________________________________________________________"
    )

    print("")
    print("Retreiving a perturbed codeword: ")
    print("")

    # Defining the parameters of a classical LDPC code.
    NUM_BITS = 12
    NUM_CHECKS = 4
    BIT_DEGREE = 3
    CHECK_DEGREE = 9
    PROB_CHANNEL = 0.2
    PROB_ERROR = 0.2
    if NUM_BITS / NUM_CHECKS != CHECK_DEGREE / BIT_DEGREE:
        raise ValueError("The graph must be bipartite.")

    # Constructing the code as a qecstruct object.
    example_code = qec.random_regular_code(
        NUM_BITS, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qec.Rng(SEED)
    )

    # Getting the sites for which the constraints should be applied.
    sites_all = linear_code_checks(example_code)

    # Rebuilding the sites lists as used in the ConstrainedString class.
    strings = []
    for _, sublist in enumerate(sites_all):

        # Retreiving the sites indices where we apply the bulk/boundary XOR tensors.
        xor_left_sites = [sublist[0]]
        xor_bulk_sites = [sublist[i] for i in range(1, CHECK_DEGREE - 1)]
        xor_right_sites = [sublist[-1]]

        # Retreiving the sites indices where we apply the SWAP tensors.
        swap_sites = list(range(sublist[0] + 1, sublist[-1]))
        for k in range(1, CHECK_DEGREE - 1):
            swap_sites.remove(sublist[k])

        strings.append([xor_left_sites, xor_bulk_sites, swap_sites, xor_right_sites])

    # Building an initial and a perturbed codeword.
    INITIAL_CODEWORD = example_code.random_codeword(qec.Rng(SEED))
    PERTURBED_CODEWORD = INITIAL_CODEWORD + qec.BinarySymmetricChannel(
        PROB_ERROR
    ).sample(NUM_BITS, qec.Rng(SEED))
    INITIAL_CODEWORD = "".join(str(bit) for bit in bin_vec_to_dense(INITIAL_CODEWORD))
    PERTURBED_CODEWORD = "".join(
        str(bit) for bit in bin_vec_to_dense(PERTURBED_CODEWORD)
    )
    print("The initial codeword is", INITIAL_CODEWORD)
    print("The perturbed codeword is", PERTURBED_CODEWORD)

    # Building the corresponding states.
    initial_codeword_state = create_custom_product_state(
        INITIAL_CODEWORD
    ).to_right_canonical()
    perturbed_codeword_state = create_custom_product_state(
        PERTURBED_CODEWORD
    ).to_right_canonical()

    # Passing the perturbed codeword state through the bias channel.
    b_channel = [bias_channel(PROB_CHANNEL) for _ in range(NUM_BITS)]
    perturbed_codeword_state = mps_mpo_contract(perturbed_codeword_state, b_channel, 0, chi_max=1)

    # Applying the parity constraints defined by the code.
    for i in range(NUM_CHECKS):
        orth_centre_init = find_orth_centre(perturbed_codeword_state)[0]
        """
        try:
            orth_centre_init = find_orth_centre(perturbed_codeword_state)[0]
        except IndexError:
            print(i)
            _, flags_left, flags_right = find_orth_centre(
                perturbed_codeword_state, return_flags=True
            )
            print(flags_left, flags_right, np.logical_and(flags_left, flags_right))
            print("**************************")
            for t in perturbed_codeword_state:
                print(t.shape)
            if flags_left == [True] * NUM_BITS:
                orth_centre_init = 0
        """
        constraint_string = ConstraintString(tensors, strings[i])
        constraint_mpo = constraint_string.get_mpo()

        START_SITE = min(constraint_string.flat())

        perturbed_codeword_state = move_orth_centre(
            perturbed_codeword_state, orth_centre_init, START_SITE
        )
        perturbed_codeword_state = mps_mpo_contract(
            perturbed_codeword_state, constraint_mpo, START_SITE
        )

    # Building the density matrix MPO.
    density_mpo = to_density_mpo(perturbed_codeword_state)

    print("DMRG running:")

    # Creating a random product state to start the DMRG with.
    np.random.seed(SEED)
    INIT_STATE_DMRG = "".join(
        str(bit)
        for bit in np.random.randint(low=0, high=2, size=NUM_BITS, dtype=np.int32)
    )
    mps_dmrg_start = create_custom_product_state(INIT_STATE_DMRG)

    print("Start state for the DMRG:", INIT_STATE_DMRG)
    engine = dmrg(mps_dmrg_start, density_mpo, chi_max=128, cut=1e-14, mode="LA")
    engine.run(NUM_BITS)
    mps_dmrg_final = engine.mps.to_right_canonical()
    print(
        "The overlap of the density MPO main component and the initial codeword state: ",
        inner_product(mps_dmrg_final, initial_codeword_state),
    )
    print(
        "__________________________________________________________________________________________"
    )
