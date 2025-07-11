"""
Here, we define some decoding-specific functions over the MPS/MPO entities
we encounter during the decoding process as well as the functions we use
to generate and operate over both classical and quantum error correcting codes.
Note, this is auxiliary code which isn't included into the library and thus provided as is.
"""

import logging
from functools import reduce
from typing import cast, Union, Optional, List, Tuple

import numpy as np
from opt_einsum import contract
from more_itertools import powerset

# pylint: disable=E0611
from qecstruct import (
    BinarySymmetricChannel,
    BinaryMatrix,
    BinaryVector,
    LinearCode,
    CssCode,
    Rng,
)

import sympy as sp
from sympy.abc import x, y
from qldpc.codes import BBCode

from mdopt.mps.explicit import ExplicitMPS
from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.utils import (
    inner_product,
    create_simple_product_state,
    create_custom_product_state,
)
from mdopt.optimiser.utils import apply_constraints
from mdopt.utils.utils import split_two_site_tensor
from mdopt.optimiser.dephasing_dmrg import DephasingDMRG
from mdopt.contractor.contractor import apply_one_site_operator
from mdopt.optimiser.utils import XOR_LEFT, XOR_BULK, XOR_RIGHT, COPY_LEFT, SWAP


# Setting up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    mps: CanonicalMPS,
    sites_to_bias: Union[str, List[int]] = "All",
    prob_bias_list: Union[float, List[float]] = 0.1,
) -> CanonicalMPS:
    """
    The function which applies a bitflip bias to a given MPS.

    Parameters
    ----------
    mps : CanonicalMPS
        The MPS to apply the operator to.
    sites_to_bias : Union[str, List[int]]
        The list of sites to which the operators are applied.
        If set to "All", takes all sites of the MPS.
    prob_bias_list : Union[float, List[float]]
        The list of probabilities of each operator at each site.
        If set to a number, applies it to all of the sites.

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

    return mps


def apply_depolarising_bias(
    mps: CanonicalMPS,
    sites_to_bias: Union[str, List[int]] = "All",
    prob_bias_list: Union[float, List[float]] = 0.1,
    renormalise: bool = True,
) -> CanonicalMPS:
    """
    The function which applies a depolarising bias to a given MPS.

    Parameters
    ----------
    mps : CanonicalMPS
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

    Raises
    ------
    ValueError
        If the number of sites in the list is not even.
        If the number of sites in the list is not equal to the number of probabilities.
        If the channel parameter is not a probability.

    Returns
    -------
    biased_mps : CanonicalMPS
        The resulting MPS.
    """

    if sites_to_bias == "All":
        sites_to_bias = list(range(0, mps.num_sites, step=2))

    if not isinstance(prob_bias_list, List):
        prob_bias_list = [prob_bias_list for _ in range(len(sites_to_bias))]

    if len(sites_to_bias) % 2 != 0:
        raise ValueError(
            f"The number of sites in the list should be even, given {len(sites_to_bias)}."
        )

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

    for site, prob_bias in zip(sites_to_bias[:-1], prob_bias_list[:-1]):
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
        if renormalise:
            mps.tensors[mps.orth_centre] /= np.linalg.norm(mps.tensors[mps.orth_centre])

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
    "E" -> "++" (erasure)
    Example: "IXYZE" -> "00101101++".

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
        elif pauli == "Z":
            mps_string += "01"
        elif pauli == "Y":
            mps_string += "11"
        elif pauli == "E":
            mps_string += "++"
        else:
            raise ValueError(f"Invalid Pauli encountered -- {pauli}.")

    return mps_string


def bin_vec_to_dense(vector: BinaryVector) -> np.ndarray:
    """
    Given a vector (1D array) in the BinaryVector format
    (native to ``qecstruct``), returns its dense representation.

    Parameters
    ----------
    vector : BinaryVector
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


def linear_code_parity_matrix_dense(code: LinearCode) -> np.ndarray:
    """
    Given a linear code, returns its parity check matrix in dense form.

    Parameters
    ----------
    code : qec.LinearCode
        Linear code object.

    Returns
    -------
    parity_matrix : np.ndarray
        The parity check matrix.
    """

    parity_matrix = code.par_mat()
    array = np.zeros((parity_matrix.num_rows(), parity_matrix.num_columns()), dtype=int)
    for row, cols in enumerate(parity_matrix.rows()):
        for col in cols:
            array[row, col] = 1
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

    parity_matrix_dense = linear_code_parity_matrix_dense(code)
    return [list(np.nonzero(row)[0]) for row in parity_matrix_dense]


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
    as integers in the big-endian (a.k.a. most-significant-bit-first) convention.

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

    # Append the all-zero codeword which is always a codeword.
    codewords.append(0)

    # Append the rows of the generator matrix.
    for basis_codeword in rows_int:
        codewords.append(basis_codeword)

    # Append all linear combinations.
    for generators in powerset(rows_int):
        if len(generators) > 1:
            codewords.append(reduce(np.bitwise_xor, generators))

    return np.sort(np.array(codewords))


def css_code_stabilisers(code: CssCode) -> Tuple[List[str], List[str]]:
    """
    Given a quantum CSS code, returns a list of its stabilisers as Pauli strings.

    Parameters
    ----------
    code : qec.CssCode
        The CSS code object.

    Returns
    -------
    stabilisers : Tuple[List[str], List[str]]
        A tuple of two lists, where the first one corresponds to X stabilisers and
        the second one -- to Z stabilisers. Each stabiliser is represented as a Pauli string.
    """

    def _binary_to_pauli(binary_row, num_qubits, pauli) -> str:
        """Helper function to convert a binary row to a Pauli string."""
        pauli_string = []
        for i in range(num_qubits):
            if binary_row[i] == 1:
                pauli_string.append(pauli)
            else:
                pauli_string.append("I")
        return "".join(pauli_string)

    num_qubits = len(code)

    # X stabilisers
    parity_matrix_x = code.x_stabs_binary()
    stabilisers_x = []
    for row in parity_matrix_x.rows():
        binary_row = np.zeros(num_qubits, dtype=int)
        for col in row:
            binary_row[col] = 1
        stabilisers_x.append(_binary_to_pauli(binary_row, num_qubits, "Z"))

    # Z stabilisers
    parity_matrix_z = code.z_stabs_binary()
    stabilisers_z = []
    for row in parity_matrix_z.rows():
        binary_row = np.zeros(num_qubits, dtype=int)
        for col in row:
            binary_row[col] = 1
        stabilisers_z.append(_binary_to_pauli(binary_row, num_qubits, "X"))

    return stabilisers_x, stabilisers_z


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


def css_code_constraint_sites(code: CssCode) -> Tuple[List[List[List[int]]]]:
    """
    Returns the list of MPS sites where the logical constraints should be applied.

    Parameters
    ----------
    code : qec.CssCode
        CSS code object.

    Returns
    -------
    sites : Tuple[List[List[List[int]]]]
        List of MPS sites.
    """

    checks_x, checks_z = css_code_checks(code)

    constraint_sites_x = []
    constraint_sites_z = []

    for checks in checks_x:
        xor_left_sites_x = [checks[0]]
        xor_bulk_sites_x = [checks[i] for i in range(1, len(checks) - 1)]
        xor_right_sites_x = [checks[-1]]

        swap_sites_x = list(range(checks[0] + 1, checks[-1]))
        for k in range(1, len(checks) - 1):
            swap_sites_x.remove(checks[k])

        constraint_sites_x.append(
            [xor_left_sites_x, xor_bulk_sites_x, swap_sites_x, xor_right_sites_x]
        )

    for checks in checks_z:
        xor_left_sites_z = [checks[0]]
        xor_bulk_sites_z = [checks[i] for i in range(1, len(checks) - 1)]
        xor_right_sites_z = [checks[-1]]

        swap_sites_z = list(range(checks[0] + 1, checks[-1]))
        for k in range(1, len(checks) - 1):
            swap_sites_z.remove(checks[k])

        constraint_sites_z.append(
            [xor_left_sites_z, xor_bulk_sites_z, swap_sites_z, xor_right_sites_z]
        )

    return constraint_sites_x, constraint_sites_z


def css_code_logicals(code: CssCode) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Returns the list of MPS sites where the logical constraints should be applied.

    Parameters
    ----------
    code : qec.CssCode
        The CSS code object.

    Returns
    -------
    logicals : Tuple[List[List[int]], List[List[int]]]
        Two lists of logical operator sites: the first for X-type logicals,
        and the second for Z-type logicals.
    """

    log_matrix_x = code.x_logicals_binary()
    array_x = np.zeros((log_matrix_x.num_rows(), log_matrix_x.num_columns()), dtype=int)
    for row, cols in enumerate(log_matrix_x.rows()):
        for col in cols:
            array_x[row, col] = 1

    log_matrix_z = code.z_logicals_binary()
    array_z = np.zeros((log_matrix_z.num_rows(), log_matrix_z.num_columns()), dtype=int)
    for row, cols in enumerate(log_matrix_z.rows()):
        for col in cols:
            array_z[row, col] = 1

    x_logicals = [
        2 * np.nonzero(row)[0] + code.num_x_logicals() + code.num_z_logicals()
        for row in array_x
    ]
    x_logicals = [list(x_logical) for x_logical in x_logicals]
    z_logicals = [
        2 * np.nonzero(row)[0] + code.num_x_logicals() + code.num_z_logicals() + 1
        for row in array_z
    ]
    z_logicals = [list(z_logical) for z_logical in z_logicals]

    return x_logicals, z_logicals


def css_code_logicals_sites(
    code: CssCode,
) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    """
    Returns the list of MPS sites where the logical operators should be applied.

    Parameters
    ----------
    code : qec.CssCode
        CSS code object.

    Returns
    -------
    strings : Tuple[List[List[List[int]]], List[List[List[int]]]]
        List of MPS sites.
    """

    sites_x, sites_z = css_code_logicals(code)

    logical_sites_x = []
    logical_sites_z = []

    for index, x_logical in enumerate(sites_x):
        copy_site_x = [index]
        xor_bulk_sites_x = [x_logical[i] for i in range(len(x_logical) - 1)]
        xor_right_site_x = [x_logical[-1]]

        swap_sites_x = list(range(copy_site_x[0] + 1, xor_right_site_x[0]))
        swap_sites_x = [site for site in swap_sites_x if site not in xor_bulk_sites_x]

        logical_sites_x.append(
            [copy_site_x, xor_bulk_sites_x, swap_sites_x, xor_right_site_x]
        )

    for index, z_logical in enumerate(sites_z):
        copy_site_z = [len(sites_x) + index]
        xor_bulk_sites_z = [z_logical[i] for i in range(len(z_logical) - 1)]
        xor_right_site_z = [z_logical[-1]]

        swap_sites_z = list(range(copy_site_z[0] + 1, xor_right_site_z[0]))
        swap_sites_z = [site for site in swap_sites_z if site not in xor_bulk_sites_z]

        logical_sites_z.append(
            [copy_site_z, xor_bulk_sites_z, swap_sites_z, xor_right_site_z]
        )

    return logical_sites_x, logical_sites_z


def create_bb_code(
    order_x: int,
    order_y: int,
    poly_a: str,
    poly_b: str,
) -> CssCode:
    """
    Builds a bivariate-bicycle CSS code from given group orders and polynomials,
    extracts its stabilizer and logical supports, and wraps it into a qecstruct.CssCode.

    Parameters
    ----------
    order_x : int
        Group order along x-axis.
    order_y : int
        Group order along y-axis.
    poly_a : str
        The polynomial A(x,y) as a string, e.g. "1 + x + y".
    poly_b : str
        The polynomial B(x,y) as a string, e.g. "1 + x**2 + y**2".

    Returns
    -------
        A qecstruct CssCode instance for the constructed bivariate bicycle code.
    """
    # Build the orders dictionary from the two integer orders using Sympy symbols
    orders = {x: order_x, y: order_y}

    # Prepare local namespace for sympify
    local_syms = {symbol.name: symbol for symbol in orders.keys()}
    # Convert string polynomials into Sympy expressions
    poly_a_expr = sp.sympify(poly_a, locals=local_syms)
    poly_b_expr = sp.sympify(poly_b, locals=local_syms)

    # Instantiate the BBCode with sympy polynomials
    bb = BBCode(orders, poly_a_expr, poly_b_expr)

    # Extract stabilizer supports as lists of qubit indices
    x_parity_check_matrix = bb.code_x.matrix
    z_parity_check_matrix = bb.code_z.matrix
    x_stabs = [[i for i, b in enumerate(row) if b] for row in x_parity_check_matrix]
    z_stabs = [[i for i, b in enumerate(row) if b] for row in z_parity_check_matrix]

    # Extract logical operator supports (uncomment if need be)
    # x_logical = bb.get_logical_ops(Pauli.X)
    # z_logical = bb.get_logical_ops(Pauli.Z)
    # x_logicals = [[i for i, b in enumerate(row) if b] for row in x_logical]
    # z_logicals = [[i for i, b in enumerate(row) if b] for row in z_logical]

    # Wrap into a qecstruct CssCode
    n = bb.code_x.matrix.shape[1]
    x_code = LinearCode(BinaryMatrix(num_columns=n, rows=x_stabs))
    z_code = LinearCode(BinaryMatrix(num_columns=n, rows=z_stabs))
    return CssCode(x_code=x_code, z_code=z_code)


def custom_code_checks(stabilizers: List[str], logicals: List[str]) -> List[List[int]]:
    """
    Given a list of stabilizers and logicals, returns a list of checks,
    where each check is represented as a list of MPS sites affected by it.

    Parameters
    ----------
    stabilizers : List[str]
        List of stabilizer generators as Pauli strings.
    logicals : List[str]
        List of logical operators as Pauli strings.

    Returns
    -------
    checks : List[List[int]]
        List of checks, each represented as a list of MPS site indices.
    """
    checks = []

    for stabilizer in stabilizers:
        bitstring = pauli_to_mps(stabilizer)
        check = len(logicals) + np.nonzero([int(bit) for bit in bitstring])[0]
        checks.append(list(check))

    return checks


def custom_code_constraint_sites(
    stabilizers: List[str], logicals: List[str]
) -> List[List[List[int]]]:
    """
    Returns the list of MPS sites where the logical constraints should be applied
    for a general quantum code.

    Parameters
    ----------
    stabilizers : List[str]
        List of stabilizer generators as Pauli strings.
    logicals : List[str]
        List of logical operators as Pauli strings.

    Returns
    -------
    constraint_sites : List[List[List[int]]]
        List of MPS sites for constraints, where each constraint corresponds
        to the locations of tensors such as XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT.
    """
    constraint_sites = []

    checks = custom_code_checks(stabilizers, logicals)

    for check in checks:
        xor_left_site = [check[0]]
        xor_bulk_sites = [check[i] for i in range(1, len(check) - 1)]
        xor_right_site = [check[-1]]

        # Identify SWAP tensor sites
        swap_sites = list(range(check[0] + 1, check[-1]))
        for bulk_site in xor_bulk_sites:
            if bulk_site in swap_sites:
                swap_sites.remove(bulk_site)

        constraint_sites.append(
            [xor_left_site, xor_bulk_sites, swap_sites, xor_right_site]
        )

    return constraint_sites


def custom_code_logicals(
    x_logicals: List[str], z_logicals: List[str]
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Returns the list of MPS sites where the logical constraints should be applied.

    Parameters
    ----------
    x_logicals : List[str]
        List of X logical operators as Pauli strings.
    z_logicals : List[str]
        List of Z logical operators as Pauli strings.

    Returns
    -------
    logicals : Tuple[List[List[int]], List[List[int]]]
        Two lists of logical operator sites: the first for X-type logicals,
        and the second for Z-type logicals.
    """
    logicals_x = []
    logicals_z = []

    # Transform X logical operators
    for logical in x_logicals:
        bitstring = pauli_to_mps(logical)
        # Find positions of non-zero entries
        x_sites = np.nonzero([int(bit) for bit in bitstring])[0]
        # Offset for X logicals
        x_sites += len(x_logicals) + len(z_logicals)
        logicals_x.append(list(x_sites))

    # Transform Z logical operators
    for logical in z_logicals:
        bitstring = pauli_to_mps(logical)
        # Find positions of non-zero entries
        z_sites = np.nonzero([int(bit) for bit in bitstring])[0]
        # Offset for Z logicals
        z_sites += len(x_logicals) + len(z_logicals)
        logicals_z.append(list(z_sites))

    return logicals_x, logicals_z


def custom_code_logicals_sites(
    x_logicals: List[str], z_logicals: List[str]
) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    """
    Returns the list of MPS sites where the logical operators should be applied
    for a general quantum error-correcting code.

    Parameters
    ----------
    x_logicals : List[str]
        List of X logical operators as Pauli strings.
    z_logicals : List[str]
        List of Z logical operators as Pauli strings.

    Returns
    -------
    logical_sites : Tuple[List[List[List[int]]], List[List[List[int]]]]
        Two lists of MPS logical sites for X-type and Z-type logicals, where each list contains:
        - COPY tensor site (first position of the logical operator)
        - XOR_BULK tensor sites (middle positions of the logical operator)
        - XOR_RIGHT tensor site (last position of the logical operator)
        - SWAP tensor sites (all remaining positions).
    """
    # Generate sites for X and Z logicals
    sites_x, sites_z = custom_code_logicals(x_logicals, z_logicals)

    logical_sites_x = []
    logical_sites_z = []

    for index, x_logical in enumerate(sites_x):
        copy_site_x = [index]
        xor_bulk_sites_x = [x_logical[i] for i in range(len(x_logical) - 1)]
        xor_right_site_x = [x_logical[-1]]

        swap_sites_x = list(range(copy_site_x[0] + 1, xor_right_site_x[0]))
        swap_sites_x = [site for site in swap_sites_x if site not in xor_bulk_sites_x]

        logical_sites_x.append(
            [copy_site_x, xor_bulk_sites_x, swap_sites_x, xor_right_site_x]
        )

    for index, z_logical in enumerate(sites_z):
        copy_site_z = [len(x_logicals) + index]
        xor_bulk_sites_z = [z_logical[i] for i in range(len(z_logical) - 1)]
        xor_right_site_z = [z_logical[-1]]

        swap_sites_z = list(range(copy_site_z[0] + 1, xor_right_site_z[0]))
        swap_sites_z = [site for site in swap_sites_z if site not in xor_bulk_sites_z]

        logical_sites_z.append(
            [copy_site_z, xor_bulk_sites_z, swap_sites_z, xor_right_site_z]
        )

    return logical_sites_x, logical_sites_z


def linear_code_prepare_message(
    code: LinearCode,
    error_rate: float = float(0.5),
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
    error_rate : float
        Error rate of the error model.
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
    perturbed_codeword = initial_codeword + error_model(error_rate).sample(
        num_bits, Rng(seed)
    )
    initial_codeword = "".join(str(bit) for bit in bin_vec_to_dense(initial_codeword))
    perturbed_codeword = "".join(
        str(bit) for bit in bin_vec_to_dense(perturbed_codeword)
    )

    return initial_codeword, perturbed_codeword


def map_distribution_to_pauli(distribution):
    """Map a distribution of logicals to Pauli operators."""
    mapping = {0: "I", 1: "X", 2: "Z", 3: "Y"}
    result = []

    for array in distribution:
        max_index = np.argmax(array)
        result.append(mapping[max_index])

    return result


def generate_pauli_error_string(
    num_qubits: int,
    error_rate: float,
    error_model: str = "Depolarising",
    rng: Optional[np.random.Generator] = None,
    erasure_rate: Optional[float] = None,
) -> str:
    """
    This function generates a random Pauli error string based on a given noise model.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the surface code.
    error_rate : float
        Physical error rate for generating Pauli errors.
    error_model : str
        The noise model to use for generating Pauli errors.
        Options are "Depolarising", "Bitflip", "Phaseflip", "Amplitudedamping", "Erasure".
    seed : Optional[int]
        Seed for the random number generator.
    erasure_rate : Optional[float]
        Probability of erasure for the erasure channel. Only used if `error_model` is "Erasure".

    Returns
    -------
    str
        A string representing the Pauli errors in the format "XZYEI...",
        where "E" represents an erasure error.
    """

    if rng is None:
        rng = np.random.default_rng()
    error_string = []

    if error_model == "Erasure" and erasure_rate is None:
        raise ValueError("Erasure rate must be specified for the erasure channel.")

    for _ in range(num_qubits):
        if error_model == "Depolarising":
            if rng.random() < error_rate:
                error = np.random.choice(["X", "Y", "Z"], p=[1 / 3, 1 / 3, 1 / 3])
            else:
                error = "I"
        elif error_model == "Bitflip":
            error = "X" if rng.random() < error_rate else "I"
        elif error_model == "Phaseflip":
            error = "Z" if rng.random() < error_rate else "I"
        elif error_model == "Amplitude Damping":
            error = rng.choice(["I", "X"], p=[1 - error_rate, error_rate])
        elif error_model == "Erasure":
            if rng.random() < erasure_rate:
                error = "E"
            elif rng.random() < error_rate:
                error = rng.choice(["X", "Z"])
            else:
                error = "I"
        else:
            raise ValueError(f"Unknown error model: {error_model}")

        error_string.append(error)

    return "".join(error_string)


def multiply_pauli_strings(pauli1: str, pauli2: str) -> str:
    """
    Multiplies two Pauli strings of the same length without considering phase.

    Parameters
    ----------
    pauli1 : str
        The first Pauli string. Each character represents a Pauli operator ('I', 'X', 'Y', 'Z').
    pauli2 : str
        The second Pauli string. Each character represents a Pauli operator ('I', 'X', 'Y', 'Z').

    Returns
    -------
    result : str
        The resulting Pauli string after multiplying pauli1 by pauli2.

    Raises
    ------
    ValueError
        If the two Pauli strings have different lengths.
    """

    if len(pauli1) != len(pauli2):
        raise ValueError(
            f"The Pauli strings must have the same length, but got {len(pauli1)} and {len(pauli2)}."
        )

    # Pauli multiplication table without phases
    pauli_multiplication_table = {
        ("I", "I"): "I",
        ("I", "X"): "X",
        ("I", "Y"): "Y",
        ("I", "Z"): "Z",
        ("X", "I"): "X",
        ("X", "X"): "I",
        ("X", "Y"): "Z",
        ("X", "Z"): "Y",
        ("Y", "I"): "Y",
        ("Y", "X"): "Z",
        ("Y", "Y"): "I",
        ("Y", "Z"): "X",
        ("Z", "I"): "Z",
        ("Z", "X"): "Y",
        ("Z", "Y"): "X",
        ("Z", "Z"): "I",
    }

    result = []

    for p1, p2 in zip(pauli1, pauli2):
        result.append(pauli_multiplication_table[(p1, p2)])

    return "".join(result)


def decode_message(
    message: Union[ExplicitMPS, CanonicalMPS],
    codeword: Union[ExplicitMPS, CanonicalMPS],
    num_runs: int = int(1),
    chi_max_dmrg: int = int(1e4),
    cut: float = float(1e-17),
    silent: bool = False,
) -> Tuple[DephasingDMRG, float]:
    """
    This function performs decoding of a message given the message state, i.e.,
    the message MPS after applying a bias channel and constraints as well as
    the codeword to compare the decoding result against.
    Returns the overlap between the decoded message given the initial message.
    This function is used independently of code generation and applying constraints.
    It is thus agnostic to code type.

    Parameters
    ----------
    message : Union[ExplicitMPS, CanonicalMPS]
        The message MPS.
    codeword : Union[ExplicitMPS, CanonicalMPS]
        The codeword MPS.
    num_runs : int
        Number of DMRG sweeps.
    chi_max_dmrg : int
        Maximum bond dimension to keep in the Dephasing DMRG algorithm.
    cut : float
        The lower boundary of the spectrum in the Dephasing DMRG algorithm.
        All the singular values smaller than that will be discarded.
    silent : bool
        Whether to show the progress bar or not.

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
    num_bits = len(message)
    mps_dmrg_start = create_simple_product_state(num_bits, which="+")

    # Running the Dephasing DMRG algorithm,
    # which finds the closest basis product state to a given MPDO,
    # which is formed from the message MPS.
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

    # Computing the overlap between the final MPS and the codeword.
    overlap = abs(inner_product(mps_dmrg_final, codeword))

    return engine, overlap


def decode_css(
    code: CssCode,
    error: str,
    num_runs: int = int(1),
    chi_max: int = int(1e4),
    cut: float = float(1e-17),
    bias_type: str = "Depolarising",
    bias_prob: float = float(0.1),
    renormalise: bool = True,
    multiply_by_stabiliser: bool = False,
    silent: bool = False,
    contraction_strategy: str = "Naive",
    optimiser: str = "Dephasing DMRG",
    tolerance: float = float(1e-12),
):
    """
    This function performs error-based decoding of a CSS code via MPS marginalisation and
    subsequent reading out the main component via densifying or Dephasing DMRG.
    It takes as input an error string and returns the most likely Pauli correction.

    Parameters
    ----------
    code : qec.CssCode
        The CSS code object.
    error : str
        The error in a string format (e.g., "XZXY...").
    num_runs : int
        Number of DMRG sweeps.
    chi_max : int
        Maximum bond dimension to keep during contractions
        and in the Dephasing DMRG algorithm.
    cut : float
        Singular value cut-off for the SVD.
    bias_type : str
        The type of the bias applied before checks.
        Available options: "Bitflip" and "Depolarising".
    bias_prob : float
        The probability of the depolarising bias applied before checks.
    renormalise : bool
        Whether to renormalise the MPS during decoding.
    multiply_by_stabiliser : bool
        Whether to multiply the error by a random stabiliser before decoding.
    silent : bool
        Whether to show the progress bars or not.
    contraction_strategy : str
        The contractor's strategy.
    optimiser : str
        The optimiser used to find the closest basis product state to a given MPDO.
        Available options: "Dephasing DMRG", "Dense", "Optima TT".
    tolerance : float
        The tolerance for the MPS classes.

    Raises
    ------
    ValueError
        If the error string length does not correspond to the code.
    """

    if not silent:
        logging.info("Starting the decoding.")

    if error == "I" * len(error):
        if not silent:
            logging.info("No error detected.")
        return [1.0, 0.0, 0.0, 0.0], 1

    error_contains_x = "X" in error
    error_contains_z = "Z" in error

    erased_qubits = [
        index for index, single_error in enumerate(error) if single_error == "E"
    ]

    if multiply_by_stabiliser and not erased_qubits:
        stabilisers_x, stabilisers_z = css_code_stabilisers(code)
        stabilisers = stabilisers_x + stabilisers_z
        error = multiply_pauli_strings(error, np.random.choice(stabilisers))

    error = pauli_to_mps(error)

    num_sites = 2 * len(code) + code.num_x_logicals() + code.num_z_logicals()
    num_logicals = code.num_x_logicals() + code.num_z_logicals()

    if not silent:
        logging.info(f"The total number of sites: {num_sites}.")
    if len(error) != num_sites - num_logicals:
        raise ValueError(
            f"The error length is {len(error)}, expected {num_sites - num_logicals}."
        )

    logicals_state = "+" * num_logicals
    state_string = logicals_state + error
    error_mps = create_custom_product_state(
        string=state_string, tolerance=tolerance, form="Right-canonical"
    )

    constraints_tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]
    logicals_tensors = [COPY_LEFT, XOR_BULK, SWAP, XOR_RIGHT]

    constraint_sites = css_code_constraint_sites(code)
    logicals_sites = css_code_logicals_sites(code)
    sites_to_bias = list(range(num_logicals, num_sites))

    if bias_type == "Bitflip":
        if not silent:
            logging.info("Applying bitflip bias.")
        error_mps = apply_bitflip_bias(
            mps=error_mps,
            sites_to_bias=sites_to_bias,
            prob_bias_list=bias_prob,
        )
    else:
        if not silent:
            logging.info("Applying depolarising bias.")
        error_mps = apply_depolarising_bias(
            mps=error_mps,
            sites_to_bias=sites_to_bias,
            prob_bias_list=bias_prob,
            renormalise=renormalise,
        )

    if error_contains_x:
        if not silent:
            logging.info("Applying X logicals' constraints.")
        error_mps = apply_constraints(
            error_mps,
            logicals_sites[0],
            logicals_tensors,
            chi_max=chi_max,
            cut=cut,
            renormalise=renormalise,
            silent=silent,
            strategy=contraction_strategy,
        )

    if error_contains_z:
        if not silent:
            logging.info("Applying Z logicals' constraints.")
        error_mps = apply_constraints(
            error_mps,
            logicals_sites[1],
            logicals_tensors,
            chi_max=chi_max,
            cut=cut,
            renormalise=renormalise,
            silent=silent,
            strategy=contraction_strategy,
        )

    if error_contains_x:
        if not silent:
            logging.info("Applying X checks' constraints.")
        error_mps = apply_constraints(
            error_mps,
            constraint_sites[0],
            constraints_tensors,
            chi_max=chi_max,
            cut=cut,
            renormalise=renormalise,
            silent=silent,
            strategy=contraction_strategy,
        )

    if error_contains_z:
        if not silent:
            logging.info("Applying Z checks' constraints.")
        error_mps = apply_constraints(
            error_mps,
            constraint_sites[1],
            constraints_tensors,
            chi_max=chi_max,
            cut=cut,
            renormalise=renormalise,
            silent=silent,
            strategy=contraction_strategy,
        )

    if erased_qubits:
        if not silent:
            logging.info("Tracing out the erased qubits.")
        error_mps = error_mps.marginal(
            sites_to_marginalise=erased_qubits,
            renormalise=renormalise,
        )

    if not silent:
        logging.info("Marginalising the error MPS.")
    sites_to_marginalise = list(
        range(num_logicals, len(error) + num_logicals - len(erased_qubits))
    )
    logical_mps = error_mps.marginal(
        sites_to_marginalise=sites_to_marginalise, renormalise=renormalise
    ).reverse()

    num_logical_sites = len(logical_mps)
    if not silent:
        logging.info(f"The number of logical sites: {num_logical_sites}.")

    if num_logical_sites <= 10:
        logical_dense = abs(
            logical_mps.dense(flatten=True, renormalise=renormalise, norm=2)
        )
        result = logical_dense, int(
            np.argmax(logical_dense) == 0 and logical_dense[0] > max(logical_dense[1:])
        )
        if not silent:
            logging.info("Decoding completed with result: %s", result)
        return result
        # Encoding: 0 -> I, 1 -> X, 2 -> Z, 3 -> Y, where the number is np.argmax(logical_dense).

    if num_logical_sites > 10 or optimiser == "Dephasing DMRG":
        mps_dmrg_start = create_simple_product_state(num_logical_sites, which="+")
        mps_dmrg_target = create_simple_product_state(num_logical_sites, which="0")
        engine = DephasingDMRG(
            mps=mps_dmrg_start,
            mps_target=logical_mps,
            chi_max=chi_max,
            cut=cut,
            mode="LA",
            silent=silent,
        )
        if not silent:
            logging.info("Running the Dephasing DMRG engine.")
        engine.run(num_iter=num_runs)
        mps_dmrg_final = engine.mps
        overlap = abs(inner_product(mps_dmrg_final, mps_dmrg_target))
        if not silent:
            logging.info("Dephasing DMRG run completed with overlap: %f", overlap)
        return engine, overlap

    if optimiser == "Optima TT":
        raise NotImplementedError("Optima TT is not implemented yet.")
    raise ValueError("Invalid optimiser chosen.")


def decode_custom(
    stabilizers: List[str],
    x_logicals: List[str],
    z_logicals: List[str],
    error: str,
    num_runs: int = int(1),
    chi_max: int = int(1e4),
    cut: float = float(1e-17),
    bias_type: str = "Depolarising",
    bias_prob: float = float(0.1),
    renormalise: bool = True,
    multiply_by_stabiliser: bool = False,
    silent: bool = False,
    contraction_strategy: str = "Naive",
    optimiser: str = "Dephasing DMRG",
    tolerance: float = float(1e-12),
):
    """
    This function performs error-based decoding for a custom quantum error-correcting code.

    Parameters
    ----------
    stabilizers : List[str]
        List of stabilizer generators as Pauli strings.
    x_logicals : List[str]
        List of X logical operators as Pauli strings.
    z_logicals : List[str]
        List of Z logical operators as Pauli strings.
    error : str
        The error in a string format (e.g., "XZXY...").
    num_runs : int
        Number of DMRG sweeps.
    chi_max : int
        Maximum bond dimension to keep during contractions
        and in the Dephasing DMRG algorithm.
    cut : float
        Singular value cut-off for the SVD.
    bias_type : str
        The type of the bias applied before the parity checks.
        Available options: "Bitflip" and "Depolarising".
    bias_prob : float
        The probability of the depolarising bias applied before the parity checks.
    renormalise : bool
        Whether to renormalise the MPS during decoding.
    multiply_by_stabiliser : bool
        Whether to multiply the error by a random stabilizer before decoding.
    silent : bool
        Whether to show the progress bars or not.
    contraction_strategy : str
        The contractor's strategy.
    optimiser : str
        The optimiser used to find the closest basis product state to a given MPDO.
        Available options: "Dephasing DMRG", "Dense", "Optima TT".
    tolerance : float
        The tolerance for the MPS classes.

    Returns
    -------
    result : Tuple
        Decoding results, depending on the chosen optimiser.
    """
    if not silent:
        logging.info("Starting the decoding.")

    if error == "I" * len(error):
        if not silent:
            logging.info("No error detected.")
        return [1.0, 0.0, 0.0, 0.0], 1

    erased_qubits = [
        index for index, single_error in enumerate(error) if single_error == "E"
    ]

    if multiply_by_stabiliser and not erased_qubits:
        chosen_stabiliser = np.random.choice(stabilizers)
        error = multiply_pauli_strings(error, chosen_stabiliser)

    error = pauli_to_mps(error)

    num_sites = len(stabilizers[0]) * 2 + len(x_logicals) + len(z_logicals)
    num_logicals = len(x_logicals) + len(z_logicals)

    if not silent:
        logging.info(f"The total number of sites: {num_sites}.")
    if len(error) != num_sites - num_logicals:
        raise ValueError(
            f"The error length is {len(error)}, expected {num_sites - num_logicals}."
        )

    logicals_state = "+" * num_logicals
    state_string = logicals_state + error

    error_mps = create_custom_product_state(
        string=state_string, tolerance=tolerance, form="Right-canonical"
    )

    constraints_tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]
    logicals_tensors = [COPY_LEFT, XOR_BULK, SWAP, XOR_RIGHT]

    constraint_sites = custom_code_constraint_sites(
        stabilizers, x_logicals + z_logicals
    )
    logicals_sites = custom_code_logicals_sites(x_logicals, z_logicals)
    sites_to_bias = list(range(num_logicals, num_sites))

    if bias_type == "Bitflip":
        if not silent:
            logging.info("Applying bitflip bias.")
        error_mps = apply_bitflip_bias(
            mps=error_mps,
            sites_to_bias=sites_to_bias,
            prob_bias_list=bias_prob,
        )
    else:
        if not silent:
            logging.info("Applying depolarising bias.")
        error_mps = apply_depolarising_bias(
            mps=error_mps,
            sites_to_bias=sites_to_bias,
            prob_bias_list=bias_prob,
            renormalise=renormalise,
        )

    if not silent:
        logging.info("Applying X logicals' constraints.")
    error_mps = apply_constraints(
        error_mps,
        logicals_sites[0],
        logicals_tensors,
        chi_max=chi_max,
        cut=cut,
        renormalise=renormalise,
        silent=silent,
        strategy=contraction_strategy,
    )

    if not silent:
        logging.info("Applying Z logicals' constraints.")
    error_mps = apply_constraints(
        error_mps,
        logicals_sites[1],
        logicals_tensors,
        chi_max=chi_max,
        cut=cut,
        renormalise=renormalise,
        silent=silent,
        strategy=contraction_strategy,
    )

    if not silent:
        logging.info("Applying X and Z checks' constraints.")
    error_mps = apply_constraints(
        error_mps,
        constraint_sites,
        constraints_tensors,
        chi_max=chi_max,
        cut=cut,
        renormalise=renormalise,
        silent=silent,
        strategy=contraction_strategy,
    )

    if erased_qubits:
        if not silent:
            logging.info("Tracing out the erased qubits.")
        error_mps = error_mps.marginal(
            sites_to_marginalise=erased_qubits,
            renormalise=renormalise,
        )

    if not silent:
        logging.info("Marginalising the error MPS.")
    sites_to_marginalise = list(
        range(num_logicals, len(error) + num_logicals - len(erased_qubits))
    )
    logical_mps = error_mps.marginal(
        sites_to_marginalise=sites_to_marginalise, renormalise=renormalise
    ).reverse()

    num_logical_sites = len(logical_mps)
    if not silent:
        logging.info(f"The number of logical sites: {num_logical_sites}.")

    if num_logical_sites <= 10:
        logical_dense = abs(
            logical_mps.dense(flatten=True, renormalise=renormalise, norm=2)
        )
        result = logical_dense, int(
            np.argmax(logical_dense) == 0 and logical_dense[0] > max(logical_dense[1:])
        )
        if not silent:
            logging.info("Decoding completed with result: %s", result)
        return result
    if num_logical_sites > 10 or optimiser == "Dephasing DMRG":
        mps_dmrg_start = create_simple_product_state(num_logical_sites, which="+")
        mps_dmrg_target = create_simple_product_state(num_logical_sites, which="0")
        engine = DephasingDMRG(
            mps=mps_dmrg_start,
            mps_target=logical_mps,
            chi_max=chi_max,
            cut=cut,
            mode="LA",
            silent=silent,
        )
        if not silent:
            logging.info("Running the Dephasing DMRG engine.")
        engine.run(num_iter=num_runs)
        mps_dmrg_final = engine.mps
        overlap = abs(inner_product(mps_dmrg_final, mps_dmrg_target))
        if not silent:
            logging.info("Dephasing DMRG run completed with overlap: %f", overlap)
        return engine, overlap
    if optimiser == "Optima TT":
        raise NotImplementedError("Optima TT is not implemented yet.")
    raise ValueError("Invalid optimiser chosen.")
