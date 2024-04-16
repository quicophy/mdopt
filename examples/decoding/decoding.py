"""
Below, we define some decoding-specific functions over the MPS/MPO entities
we encounter during the decoding process as well as the functions we use
to generate and operate over both classical and quantum error correcting codes.
Note, that this is auxiliary code which is not included into the library,
thus not tested and provided as is.
"""

from functools import reduce
from typing import cast, Union, Optional, List, Tuple

import numpy as np
from tqdm import tqdm
from opt_einsum import contract
from more_itertools import powerset
from qecstruct import (
    Rng,
    BinarySymmetricChannel,
    shor_code,
    CssCode,
    LinearCode,
)  # pylint: disable=E0611

from mdopt.mps.explicit import ExplicitMPS
from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.utils import (
    marginalise,
    inner_product,
    find_orth_centre,
    create_simple_product_state,
    create_custom_product_state,
)
from mdopt.contractor.contractor import apply_one_site_operator, mps_mpo_contract
from mdopt.optimiser.utils import XOR_LEFT, XOR_BULK, XOR_RIGHT, COPY_LEFT, SWAP
from mdopt.optimiser.dephasing_dmrg import DephasingDMRG
from mdopt.utils.utils import split_two_site_tensor
from mdopt.optimiser.utils import ConstraintString


def bitflip_bias(prob_bias: float = float(0.5)) -> np.ndarray:
    """
    This function returns a bitflip bias operator -- the operator which will bias us
    towards the initial input by ranking the bitstrings according to
    the Hamming distance from the latter by virtue of bitflip.

    Parameters
    ----------
    prob_bias : float
        Probability of the operator.

    Returns
    -------
    bias_operator : np.ndarray
        The corresponding one-site MPO.

    Raises
    ------
    ValueError
        If the channel's probability has incorrect value.

    Notes
    -----
    This function returns a one-site bias channel MPO which
    acts on one-qubit computational basis states as follows:
    |0> -> √(1-p)|0> + √p|1>,
    |1> -> √p|0>  + √(1-p)|1>,
    Note, that this operation is not unitary, which means that it does not
    preserve the canonical form without enforcing renormalisation.
    """

    if not 0 <= prob_bias <= 1:
        raise ValueError(
            f"The channel parameter `prob_bias` should be a probability, "
            f"given {prob_bias}."
        )

    bias_operator = np.full(shape=(2, 2), fill_value=np.sqrt(prob_bias))
    np.fill_diagonal(bias_operator, np.sqrt(1 - prob_bias))

    return bias_operator


def depolarising_bias(prob_bias: float = float(0.5)) -> np.ndarray:
    """
    This function returns a depolarising bias operator -- the operator which will bias us
    towards the initial input by ranking the bitstrings according to
    the Hamming distance from the latter by virtue of depolarisation.

    Parameters
    ----------
    prob_bias : float
        Probability of the operator.

    Returns
    -------
    bias_operator : np.ndarray
        The corresponding two-site MPO.

    Raises
    ------
    ValueError
        If the channel's probability has incorrect value.

    Notes
    -----
    This function returns a two-site bias channel MPO which
    acts on two-qubit computational basis states as follows:
    |00> -> √(1-p)|00> + √(p/3)|01> + √(p/3)|10> + √(p/3)|11>,
    |01> -> √(p/3)|00> + √(1-p)|01> + √(p/3)|10> + √(p/3)|11>,
    |10> -> √(p/3)|00> + √(p/3)|01> + √(1-p)|10> + √(p/3)|11>,
    |11> -> √(p/3)|00> + √(p/3)|01> + √(p/3)|10> + √(1-p)|11>,
    Note, that this operation is not unitary, which means that it does not
    preserve the canonical form without enforcing renormalisation.
    Following our convention, the operator has legs ``(pUL, pUR, pDL, pDR)``,
    where ``p`` stands for "physical", and
    ``L``, ``R``, ``U``, ``D`` -- for "left", "right", "up", "down" accordingly.
    """

    if not 0 <= prob_bias <= 1:
        raise ValueError(
            f"The channel parameter `prob_bias` should be a probability, "
            f"given {prob_bias}."
        )

    bias_operator = np.full(shape=(2, 2, 2, 2), fill_value=np.sqrt(prob_bias / 3))
    np.fill_diagonal(bias_operator, np.sqrt(1 - prob_bias))

    return bias_operator


def apply_bitflip_bias(
    mps: Union[ExplicitMPS, CanonicalMPS],
    sites_to_bias: Union[str, List[int]] = "All",
    prob_bias_list: Union[float, List[float]] = 0.1,
    renormalise: bool = True,
) -> CanonicalMPS:
    """
    The function which applies a bitflip bias to a given MPS.

    Parameters
    ----------
    mps : Union[ExplicitMPS, CanonicalMPS]
        The MPS to apply the operator to.
    sites_to_bias : Union[str, List[int]]
        The list of sites to which the operators are applied.
        If set to "All", takes all sites of the MPS.
    prob_bias_list : Union[float, List[float]]
        The list of probabilities of each operator at each site.
        If set to a number, applies it to all of the sites.
    renormalise : bool
        Whether to renormalise spectra during contraction.

    Returns
    -------
    biased_mps : CanonicalMPS
        The resulting MPS.
    """

    if sites_to_bias == "All":
        sites_to_bias = list(range(mps.num_sites))

    if not isinstance(prob_bias_list, List):
        prob_bias_list = [prob_bias_list for _ in range(len(sites_to_bias))]

    if len(sites_to_bias) != len(prob_bias_list):
        raise ValueError(
            f"The number of sites in the list is {len(sites_to_bias)}, which is not"
            f"equal to the number of probabilies -- {len(prob_bias_list)}."
        )

    for site, probability in enumerate(prob_bias_list):
        if not 0 <= probability <= 1:
            raise ValueError(
                f"The channel parameter should be a probability, "
                f"given {probability} at site {site}."
            )

    for site, prob_bias in zip(sites_to_bias, prob_bias_list):
        mps.tensors[site] = apply_one_site_operator(
            tensor=mps.tensors[site],
            operator=bitflip_bias(prob_bias),
        )

    if isinstance(mps, ExplicitMPS):
        mps = mps.mixed_canonical(orth_centre=mps.num_sites - 1)
        mps = mps.move_orth_centre(final_pos=0, renormalise=renormalise)

    return mps


def apply_depolarising_bias(
    mps: Union[ExplicitMPS, CanonicalMPS],
    sites_to_bias: Union[str, List[int]] = "All",
    prob_bias_list: Union[float, List[float]] = 0.1,
    renormalise: bool = True,
    result_to_explicit: bool = False,
) -> Union[ExplicitMPS, CanonicalMPS]:
    """
    The function which applies a depolarising bias to a MPS.

    Parameters
    ----------
    mps : Union[ExplicitMPS, CanonicalMPS]
        The MPS to apply the operator to.
    sites_to_bias : Union[str, List[int]]
        The list of sites to which the operators are applied.
        If set to "All", takes all sites of the MPS.
        Note, each site in this list means the next site is also
        taken into account.
    prob_bias_list : Union[float, List[float]]
        The list of probabilities of each operator at each site.
        If set to a number, applies it to all of the sites.
    renormalise : bool
        Whether to renormalise spectra during contraction.
    result_to_explicit : bool
        Whether to transform the resulting MPS into the Explicit form.

    Returns
    -------
    biased_mps : Union[ExplicitMPS, CanonicalMPS]
        The resulting MPS.
    """

    if sites_to_bias == "All":
        sites_to_bias = list(range(0, mps.num_sites, step=2))

    if not isinstance(prob_bias_list, List):
        prob_bias_list = [prob_bias_list for _ in range(len(sites_to_bias))]

    if len(sites_to_bias) != len(prob_bias_list):
        raise ValueError(
            f"The number of sites in the list is {len(sites_to_bias)}, which is not"
            f"equal to the number of probabilies -- {len(prob_bias_list)}."
        )

    for site, probability in enumerate(prob_bias_list):
        if not 0 <= probability <= 1:
            raise ValueError(
                f"The channel parameter should be a probability, "
                f"given {probability} at site {site}."
            )

    mps = mps.mixed_canonical(orth_centre=min(sites_to_bias))

    for site, prob_bias in zip(sites_to_bias, prob_bias_list):
        two_site_tensor = contract(
            "ijk, klm, jlno -> inom",
            mps.tensors[site],
            mps.tensors[site + 1],
            depolarising_bias(prob_bias=prob_bias),
            optimize=[(0, 1), (0, 1)],
        )
        mps.tensors[site], singular_values, b_r, _ = split_two_site_tensor(
            two_site_tensor,
            renormalise=renormalise,
            return_truncation_error=True,
        )
        mps.tensors[site + 1] = contract(
            "ij, jkl -> ikl", np.diag(singular_values), b_r, optimize=[(0, 1)]
        )
        mps.orth_centre = site + 1

    if result_to_explicit:
        return mps.explicit(renormalise=renormalise)

    return mps


# Below, we define some utility functions to operate with data structures from qecstruct and
# qecsim -- quantum error-correction libraries we use for our decoding examples.


def pauli_to_mps(pauli_string: str) -> str:
    """
    This function converts a Pauli string to our MPS decoder string.
    The encoding is done as follows:
    "I" -> "00"
    "X" -> "10"
    "Y" -> "11"
    "Z" -> "01"
    Example: "IXYZ" -> "00101101".

    Parameters
    ----------
    pauli_string : str
        The Pauli string.

    Returns
    -------
    mps_string : str
        The MPS string.
    """

    mps_string = ""
    for pauli in pauli_string:
        if pauli == "I":
            mps_string += "00"
        elif pauli == "X":
            mps_string += "10"
        elif pauli == "Y":
            mps_string += "11"
        elif pauli == "Z":
            mps_string += "01"
        else:
            raise ValueError(f"Invalid Pauli encountered -- {pauli}.")

    return mps_string


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


def linear_code_checks(code: LinearCode) -> List[List[int]]:
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


def linear_code_constraint_sites(code: LinearCode) -> List[List[List[int]]]:
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


def linear_code_codewords(code: LinearCode) -> np.ndarray:
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


def css_code_checks(code: CssCode) -> Tuple[List[List[int]]]:
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


def css_code_constraint_sites(code: CssCode) -> Tuple[List[int]]:
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


def css_code_logicals(code: CssCode):
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


def css_code_logicals_sites(code: CssCode) -> Tuple[List[int]]:
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
    code: LinearCode,
    prob_error: float = float(0.5),
    error_model: "qec.noise_model" = BinarySymmetricChannel,
    seed: Optional[int] = None,
) -> Tuple[str, str]:
    """
    This function prepares a message in the form of a random codeword
    and its perturbed version after applying an error model.

    Parameters
    ----------
    code : qec.LinearCode
        Linear code object.
    prob_error : float
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
    initial_codeword = code.random_codeword(Rng(seed))
    perturbed_codeword = initial_codeword + error_model(prob_error).sample(
        num_bits, Rng(seed)
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
    result_to_explicit: bool = True,
    strategy: str = "Naive",
    silent: bool = False,
    return_entropies_and_bond_dims: bool = False,
) -> Union[CanonicalMPS, ExplicitMPS]:
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
    result_to_explicit : bool
        Whether to transform the resulting MPS into the Explicit form.
    strategy : str
        The contractor strategy.
    silent : bool
        Whether to show the progress bar or not.
    return_entropies_and_bond_dims : bool
        Whether to return the entanglement entropies and bond dimensions at each bond.

    Returns
    -------
    mps : Union[CanonicalMPS, ExplicitMPS]
        The resulting MPS.
    entropies, bond_dims : List[float], List[int]
        The list of entanglement entropies at each bond.
        Returned only if ``return_entropies_and_bond_dims`` is set to ``True``.
    """

    entropies = []
    bond_dims = []
    if strategy == "Naive":
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
                    mps.move_orth_centre(final_pos=start_site, renormalise=True),
                )

            mps = mps_mpo_contract(
                mps,
                mpo,
                start_site,
                renormalise=renormalise,
                chi_max=chi_max,
                inplace=False,
                result_to_explicit=result_to_explicit,
            )

            if return_entropies_and_bond_dims:
                entropies.append(mps.entanglement_entropy())
                bond_dims.append(mps.bond_dimensions)

    if return_entropies_and_bond_dims:
        return cast(CanonicalMPS, mps), entropies, bond_dims

    return cast(CanonicalMPS, mps)


def decode_linear(
    message: Union[ExplicitMPS, CanonicalMPS],
    codeword: Union[ExplicitMPS, CanonicalMPS],
    code: LinearCode,
    num_runs: int = int(1),
    chi_max_dmrg: int = int(1e4),
    cut: float = float(1e-12),
    silent: bool = False,
) -> Tuple[DephasingDMRG, float]:
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
    cut : float
        The lower boundary of the spectrum in the DMRG algorithm.
        All the singular values smaller than that will be discarded.

    Returns
    -------
    engine : DephasingDMRG
        The container class for the Dephasing DMRG algorithm,
        see :class:`mdopt.optimiser.DephasingDMRG`.
    overlap : float
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


def decode_shor(
    error: str,
    renormalise: bool = True,
    silent: bool = True,
):
    """
    This function does error-based decoding of Shor's code.
    It takes as input an error string and returns the most likely Pauli correction.

    Parameters
    ----------
    error : str
        The error in a string format.
        The way the decoder takes the error is as follows:
        "X_0 Z_0 X_1 Z_1 ..."
    renormalise : bool
        Whether to renormalise the singular values during contraction.
    silent : bool
        Whether to show the progress bars or not.

    Raises
    ------
    ValueError
        If the error string length does not correspond to the code.
    """

    code = shor_code()
    num_sites = 2 * len(code) + code.num_x_logicals() + code.num_z_logicals()
    num_logicals = code.num_x_logicals() + code.num_z_logicals()
    assert num_sites == 20
    assert num_logicals == 2

    if len(error) != num_sites - num_logicals:
        raise ValueError(
            f"The error length is {len(error)}, expected {num_sites - num_logicals}."
        )

    logicals_state = "+" * num_logicals
    state_string = logicals_state + error
    error_mps = create_custom_product_state(string=state_string)
    constraints_tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]
    logicals_tensors = [COPY_LEFT, XOR_BULK, SWAP, XOR_RIGHT]
    constraints_sites = css_code_constraint_sites(code)
    logicals_sites = css_code_logicals_sites(code)
    sites_to_bias = list(range(num_logicals, num_sites))

    error_mps = apply_bitflip_bias(
        mps=error_mps, sites_to_bias=sites_to_bias, renormalise=renormalise
    )

    error_mps = apply_constraints(
        error_mps,
        constraints_sites[0],
        constraints_tensors,
        renormalise=renormalise,
        silent=silent,
    )
    error_mps = apply_constraints(
        error_mps,
        constraints_sites[1],
        constraints_tensors,
        renormalise=renormalise,
        silent=silent,
    )
    error_mps = apply_constraints(
        error_mps,
        logicals_sites,
        logicals_tensors,
        renormalise=renormalise,
        silent=silent,
    )

    sites_to_marginalise = list(range(num_logicals, len(error) + num_logicals))
    logical = marginalise(
        mps=error_mps,
        sites_to_marginalise=sites_to_marginalise,
    ).dense(flatten=True, renormalise=True, norm=1)

    if np.argmax(logical) == 0:
        return "I", logical
    if np.argmax(logical) == 1:
        return "X", logical
    if np.argmax(logical) == 2:
        return "Z", logical
    if np.argmax(logical) == 3:
        return "Y", logical
