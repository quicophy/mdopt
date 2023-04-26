"""
Below, we define some decoding-specific functions over the MPS/MPO entities
we encounter during the decoding process as well as the functions we use
to generate and operate over both classical and quantum error correcting codes.
"""

from functools import reduce
from typing import cast, Union, Optional, List, Tuple

import numpy as np
from tqdm import tqdm
import qecstruct as qec
from more_itertools import powerset

from mdopt.mps.explicit import ExplicitMPS
from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.utils import find_orth_centre, inner_product, create_simple_product_state
from mdopt.contractor.contractor import apply_one_site_operator, mps_mpo_contract
from mdopt.optimiser.dephasing_dmrg import DephasingDMRG
from mdopt.optimiser.utils import ConstraintString


def bias_channel(p_bias: np.float32 = np.float32(0.5), which: str = "0") -> np.ndarray:
    """
    Here, we define bias channel -- an operator which will bias us towards the initial message
    while decoding by ranking the bitstrings according to Hamming distance from the latter.
    This function returns a one-site bias channel MPO which
    acts on one-qubit computational basis states as follows:
    |0> -> √(1-p)|0> + √p|1>,
    |1> -> √p|0> - √(1-p)|1>,
    Note, that this operation is unitary, which means that it preserves the canonical form.

    Parameters
    ----------
    p_bias : np.float32
        Probability of the channel.
    which : str
        "0" or "1", depending on which one-qubit basis state we are acting on.

    Returns
    -------
    b_ch : np.ndarray
        The corresponding one-site MPO.
    """

    if not 0 <= p_bias <= 1:
        raise ValueError(
            f"The channel parameter `p_bias` should be a probability, "
            f"given {p_bias}."
        )
    if which not in ["0", "1", "+"]:
        raise ValueError("Invalid one-qubit basis state given.")

    if which == "0":
        b_channel = np.array(
            [
                [np.sqrt(1 - p_bias), np.sqrt(p_bias)],
                [np.sqrt(p_bias), -np.sqrt(1 - p_bias)],
            ]
        )
    if which == "1":
        b_channel = np.array(
            [
                [-np.sqrt(1 - p_bias), np.sqrt(p_bias)],
                [np.sqrt(p_bias), np.sqrt(1 - p_bias)],
            ]
        )
    if which == "+":
        b_channel = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

    return b_channel


def apply_bias_channel(
    basis_mps: Union[ExplicitMPS, CanonicalMPS],
    basis_string: str,
    prob_channel: np.float32 = np.float32(0.5),
) -> Union[ExplicitMPS, CanonicalMPS]:
    """
    The function which applies a bias channel to a computational-basis-state MPS.

    Parameters
    ----------
    basis_mps : Union[ExplicitMPS, CanonicalMPS]
        The computational-basis-state MPS, e.g., ``|010010>``.
    basis_string : str
        The string of "0", "1" and "+" which corresponds to ``basis_mps``.
    prob_channel : np.float32
        The bias channel probability.


    Returns
    -------
    biased_mps : CanonicalMPS
        The resulting MPS.
    """

    if len(basis_mps) != len(basis_string):
        raise ValueError(
            f"The lengths of `basis_mps` and `codeword_string` should be equal, but given the "
            f"MPS of length {len(basis_mps)} and the string of length {len(basis_string)}."
        )

    biased_mps_tensors = []
    for i, mps_tensor in enumerate(basis_mps.tensors):
        biased_mps_tensors.append(
            apply_one_site_operator(
                tensor=mps_tensor,
                operator=bias_channel(prob_channel, which=basis_string[i]),
            )
        )

    if isinstance(basis_mps, ExplicitMPS):
        return ExplicitMPS(
            tensors=biased_mps_tensors,
            singular_values=basis_mps.singular_values,
            tolerance=basis_mps.tolerance,
            chi_max=basis_mps.chi_max,
        )

    if isinstance(basis_mps, CanonicalMPS):
        return CanonicalMPS(
            tensors=biased_mps_tensors,
            orth_centre=basis_mps.orth_centre,
            tolerance=basis_mps.tolerance,
            chi_max=basis_mps.chi_max,
        )


# Below, we define some utility functions to operate with data structures from `qecstruct` --
# an error-correction library we are using in this example.


def bin_vec_to_dense(vector: "qec.sparse.BinaryVector") -> np.ndarray:
    """
    Given a vector (1D array) in the ``qecstruct.sparse.BinaryVector`` format
    (native to ``qecstruct``), returns its dense representation.

    Parameters
    ----------
    vector : qec.sparse.BinaryVector
        The vector we want to densify.

    Returns
    -------
    array : np.ndarray
        The dense representation.
    """

    array = np.zeros(vector.len(), dtype=int)
    for pos in vector:
        array[pos] = 1
    return array


def linear_code_checks(code: "qec.LinearCode") -> List[List[int]]:
    """
    Given a linear code, returns a list of its checks, where each check
    is represented as a list of indices of the bits touched by it.

    Parameters
    ----------
    code : qec.LinearCode
        Linear code object.

    Returns
    -------
    checks : List[List[int]]
        List of checks.
    """

    parity_matrix = code.par_mat()
    array = np.zeros((parity_matrix.num_rows(), parity_matrix.num_columns()), dtype=int)
    for row, cols in enumerate(parity_matrix.rows()):
        for col in cols:
            array[row, col] = 1
    return [list(np.nonzero(row)[0]) for row in array]


def linear_code_constraint_sites(code: "qec.LinearCode") -> List[List[List[int]]]:
    """
    Returns the list of MPS sites where the logical constraints should be applied.

    Parameters
    ----------
    code : qec.LinearCode
        Linear code object.

    Returns
    -------
    strings : List[List[List[int]]]
        List of MPS sites.
    """

    sites_all = linear_code_checks(code)
    check_degree = len(sites_all[0])
    constraints_strings = []

    for sites in sites_all:
        # Retreiving the sites indices where we apply the "bulk"/"boundary" XOR tensors.
        xor_left_sites = [sites[0]]
        xor_bulk_sites = [sites[i] for i in range(1, check_degree - 1)]
        xor_right_sites = [sites[-1]]

        # Retreiving the sites indices where we apply the SWAP tensors.
        swap_sites = list(range(sites[0] + 1, sites[-1]))
        for k in range(1, check_degree - 1):
            swap_sites.remove(sites[k])

        constraints_strings.append(
            [xor_left_sites, xor_bulk_sites, swap_sites, xor_right_sites]
        )

    return cast(List[List[List[int]]], constraints_strings)


def linear_code_codewords(code: "qec.LinearCode") -> np.ndarray:
    """
    Returns the list of codewords of a linear code. Codewords are returned
    as integers in most-significant-bit-first convention.

    Parameters
    ----------
    code : qec.LinearCode
        Linear code object.

    Returns
    -------
    codewords : np.ndarray
        The codewords.
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


def css_code_checks(code: qec.CssCode) -> Tuple[List[int]]:
    """
    Given a quantum CSS code, returns a list of its checks, where each check
    is represented as a list of indices of the bits adjacent to it.

    Parameters
    ----------
    code : qec.CssCode
        The CSS code object.

    Returns
    -------
    checks : Tuple[List[List[int]]
        A tuple of two lists, where the first one corresponds to X checks and
        the second one -- to Z checks.
    """

    parity_matrix_x = code.x_stabs_binary()
    array_x = np.zeros(
        (parity_matrix_x.num_rows(), parity_matrix_x.num_columns()), dtype=int
    )
    for row, cols in enumerate(parity_matrix_x.rows()):
        for col in cols:
            array_x[row, col] = 1

    parity_matrix_z = code.z_stabs_binary()
    array_z = np.zeros(
        (parity_matrix_z.num_rows(), parity_matrix_z.num_columns()), dtype=int
    )
    for row, cols in enumerate(parity_matrix_z.rows()):
        for col in cols:
            array_z[row, col] = 1

    checks_x = [
        2 * np.nonzero(row)[0] + code.num_x_logicals() + code.num_z_logicals()
        for row in array_x
    ]
    checks_x = [list(check_x) for check_x in checks_x]
    checks_z = [
        2 * np.nonzero(row)[0] + code.num_x_logicals() + code.num_z_logicals() + 1
        for row in array_z
    ]
    checks_z = [list(check_z) for check_z in checks_z]

    return checks_x, checks_z


def css_code_constraint_sites(code: qec.CssCode) -> Tuple[List[int]]:
    """
    Returns the list of MPS sites where the logical constraints should be applied.

    Parameters
    ----------
    code : qec.CssCode
        CSS code object.

    Returns
    -------
    strings : Tuple[List[int]]
        List of MPS sites.
    """

    sites_x, sites_z = css_code_checks(code)

    constraints_strings_x = []
    constraints_strings_z = []

    for sites in sites_x:
        xor_left_sites_x = [sites[0]]
        xor_bulk_sites_x = [sites[i] for i in range(1, len(sites) - 1)]
        xor_right_sites_x = [sites[-1]]

        swap_sites_x = list(range(sites[0] + 1, sites[-1]))
        for k in range(1, len(sites) - 1):
            swap_sites_x.remove(sites[k])

        constraints_strings_x.append(
            [xor_left_sites_x, xor_bulk_sites_x, swap_sites_x, xor_right_sites_x]
        )

    for sites in sites_z:
        xor_left_sites_z = [sites[0]]
        xor_bulk_sites_z = [sites[i] for i in range(1, len(sites) - 1)]
        xor_right_sites_z = [sites[-1]]

        swap_sites_z = list(range(sites[0] + 1, sites[-1]))
        for k in range(1, len(sites) - 1):
            swap_sites_z.remove(sites[k])

        constraints_strings_z.append(
            [xor_left_sites_z, xor_bulk_sites_z, swap_sites_z, xor_right_sites_z]
        )

    return constraints_strings_x, constraints_strings_z


def css_code_logicals(code: qec.CssCode):
    """
    Returns the list of MPS sites where the logical constraints should be applied.

    Parameters
    ----------
    code : qec.CssCode
        The CSS code object.

    Returns
    -------
    logicals : Tuple[List[int]]
        List of logical operators, first X, then Z.
    """

    log_matrix_x = code.z_logicals_binary()
    array_x = np.zeros((log_matrix_x.num_rows(), log_matrix_x.num_columns()), dtype=int)
    for row, cols in enumerate(log_matrix_x.rows()):
        for col in cols:
            array_x[row, col] = 1

    log_matrix_z = code.x_logicals_binary()
    array_z = np.zeros((log_matrix_z.num_rows(), log_matrix_z.num_columns()), dtype=int)
    for row, cols in enumerate(log_matrix_z.rows()):
        for col in cols:
            array_z[row, col] = 1

    x_logicals = [
        2 * np.nonzero(row)[0] + code.num_x_logicals() + code.num_z_logicals() + 1
        for row in array_x
    ]
    x_logicals = [list(x_logical) for x_logical in x_logicals]
    z_logicals = [
        2 * np.nonzero(row)[0] + code.num_x_logicals() + code.num_z_logicals()
        for row in array_z
    ]
    z_logicals = [list(z_logical) for z_logical in z_logicals]

    return z_logicals[0], x_logicals[0]


def css_code_logicals_sites(code: qec.CssCode) -> Tuple[List[int]]:
    """
    Returns the list of MPS sites where the logical operators should be applied.

    Parameters
    ----------
    code : qec.CssCode
        CSS code object.

    Returns
    -------
    strings : Tuple[List[int]]
        List of MPS sites.
    """

    sites_x, sites_z = css_code_logicals(code)

    copy_site_x = [0]
    copy_site_z = [1]

    xor_right_site_x = [sites_x[-1]]
    xor_right_site_z = [sites_z[-1]]

    xor_bulk_sites_x = [sites_x[i] for i in range(len(sites_x) - 1)]
    xor_bulk_sites_z = [sites_z[i] for i in range(len(sites_z) - 1)]

    swap_sites_x = list(range(copy_site_x[0] + 1, xor_right_site_x[0]))
    swap_sites_x = [site for site in swap_sites_x if site not in xor_bulk_sites_x]
    swap_sites_z = list(range(copy_site_z[0] + 1, xor_right_site_z[0]))
    swap_sites_z = [site for site in swap_sites_z if site not in xor_bulk_sites_z]

    string_x = [copy_site_x, xor_bulk_sites_x, swap_sites_x, xor_right_site_x]
    string_z = [copy_site_z, xor_bulk_sites_z, swap_sites_z, xor_right_site_z]

    return string_x, string_z


def linear_code_prepare_message(
    code: "qec.LinearCode",
    prob_error: np.float32 = np.float32(0.5),
    error_model: "qec.noise_model" = qec.BinarySymmetricChannel,
    seed: Optional[int] = None,
) -> Tuple[str, str]:
    """
    This function prepares a message in the form of a random codeword
    and its perturbed version after applying an error model.

    Parameters
    ----------
    code : qec.LinearCode
        Linear code object.
    prob_error : np.float32
        Error probability of the error model.
    error_model : qec.noise_model
        The error model used to flip bits of a random codeword.
    seed : Optional[int]
        Random seed.

    Returns
    -------
    initial_codeword : str
        The bitstring of the initial codeword.
    perturbed_codeword : str
        The bitstring of the perturbed codeword.
    """

    num_bits = len(code)
    initial_codeword = code.random_codeword(qec.Rng(seed))
    perturbed_codeword = initial_codeword + error_model(prob_error).sample(
        num_bits, qec.Rng(seed)
    )
    initial_codeword = "".join(str(bit) for bit in bin_vec_to_dense(initial_codeword))
    perturbed_codeword = "".join(
        str(bit) for bit in bin_vec_to_dense(perturbed_codeword)
    )

    return initial_codeword, perturbed_codeword


# The functions below are used to apply constraints to a codeword MPS and perform actual decoding.


def apply_constraints(
    mps: Union[ExplicitMPS, CanonicalMPS],
    strings: List[List[int]],
    logical_tensors: List[np.ndarray],
    chi_max: int = int(1e4),
    renormalise: bool = False,
    strategy: str = "naive",
    silent: bool = False,
) -> CanonicalMPS:
    """
    This function applies logical constraints to an MPS.

    Parameters
    ----------
    mps : Union[ExplicitMPS, CanonicalMPS]
        The MPS to which the logical constraints are being applied.
    strings : List[List[int]]
        The list of arguments for :class:`ConstraintString`.
    logical_tensors : List[np.ndarray]
        List of logical tensors for :class:`ConstraintString`.
    chi_max : int
        Maximum bond dimension to keep in the contractor.
    renormalise : bool
        Whether to renormalise the singular values at each MPS bond involved in contraction.
    strategy : str
        The contractor strategy.
    silent : bool
        Whether to show the progress bar or not.

    Returns
    -------
    mps : CanonicalMPS
        The resulting MPS.
    """

    if strategy == "naive":
        for string in tqdm(strings, disable=silent):
            # Preparing the MPO.
            string = ConstraintString(logical_tensors, string)
            mpo = string.mpo()

            # Finding the starting site for the MPS to perform contraction.
            start_site = min(string.flat())

            # Preparing the MPS for contraction.
            if isinstance(mps, ExplicitMPS):
                mps = mps.mixed_canonical(orth_centre=start_site)

            if isinstance(mps, CanonicalMPS):
                if mps.orth_centre is None:
                    orth_centres, flags_left, flags_right = find_orth_centre(
                        mps, return_orth_flags=True
                    )

                    # Managing possible issues with multiple orthogonality centres
                    # arising if we do not renormalise while contracting.
                    if orth_centres and len(orth_centres) == 1:
                        mps.orth_centre = orth_centres[0]
                    # Convention.
                    if all(flags_left) and all(flags_right):
                        mps.orth_centre = 0
                    elif flags_left in ([True] + [False] * (mps.num_sites - 1)):
                        if flags_right == [not flag for flag in flags_left]:
                            mps.orth_centre = mps.num_sites - 1
                    elif flags_left in ([True] * (mps.num_sites - 1) + [False]):
                        if flags_right == [not flag for flag in flags_left]:
                            mps.orth_centre = 0
                    elif all(flags_right):
                        mps.orth_centre = 0
                    elif all(flags_left):
                        mps.orth_centre = mps.num_sites - 1

                mps = cast(
                    Union[ExplicitMPS, CanonicalMPS],
                    mps.move_orth_centre(final_pos=start_site),
                )

            # Doing the contraction.
            mps = mps_mpo_contract(
                mps,
                mpo,
                start_site,
                renormalise=renormalise,
                chi_max=chi_max,
                inplace=False,
            )

    return cast(CanonicalMPS, mps)


def decode(
    message: Union[ExplicitMPS, CanonicalMPS],
    codeword: Union[ExplicitMPS, CanonicalMPS],
    code: "qec.LinearCode",
    num_runs: int = int(1),
    chi_max_dmrg: int = int(1e4),
    cut: np.float32 = np.float32(1e-12),
    silent: bool = False,
) -> Tuple[DephasingDMRG, np.float32]:
    """
    This function performs actual decoding of a message given a code and
    the DMRG truncation parameters.
    Returns the overlap between the decoded message given the initial message.

    Parameters
    ----------
    message : Union[ExplicitMPS, CanonicalMPS]
        The message MPS.
    codeword : Union[ExplicitMPS, CanonicalMPS]
        The codeword MPS.
    code : qec.LinearCode
        Linear code object.
    num_runs : int
        Number of DMRG sweeps.
    chi_max_dmrg : int
        Maximum bond dimension to keep in the DMRG algorithm.
    cut : np.float32
        The lower boundary of the spectrum in the DMRG algorithm.
        All the singular values smaller than that will be discarded.

    Returns
    -------
    engine : DephasingDMRG
        The container class for the Deohasing DMRG algorithm, see :class:`mdopt.optimiser.DMRG`.
    overlap : np.float32
        The overlap between the decoded message and a given codeword,
        computed as the following inner product |<decoded_message|codeword>|.
    """

    # Creating an all-plus state to start the DMRG with.
    num_bits = len(code)
    mps_dmrg_start = create_simple_product_state(num_bits, which="+")
    engine = DephasingDMRG(
        mps_dmrg_start,
        message,
        chi_max=chi_max_dmrg,
        cut=cut,
        mode="LA",
        silent=silent,
    )
    engine.run(num_runs)
    mps_dmrg_final = engine.mps.right_canonical()
    overlap = abs(inner_product(mps_dmrg_final, codeword))

    return engine, overlap
