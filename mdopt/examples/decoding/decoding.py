"""
Here, we define some decoding-specific functions over the MPS/MPO entities
we encounter during the decoding process as well as the functions we use
to generate and operate over both classical and quantum error correcting codes.
Note, this is example code which isn't included into the library and thus provided as is.
"""

import argparse
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

try:
    from qldpc.codes import BBCode
except ModuleNotFoundError:
    BBCode = None  # only needed for BB-code specific functions

from mdopt.mps.explicit import ExplicitMPS
from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.utils import (
    inner_product,
    create_simple_product_state,
    create_custom_product_state,
)
from mdopt.optimiser.utils import apply_constraints, optimise_qubit_order
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

    # Build it as a 4x4 map on the two-qubit basis and only then fold it back
    # into legs (pUL, pUR, pDL, pDR). np.fill_diagonal on a rank-4 array walks
    # B[i, i, i, i], which sets only |00> -> |00> and |11> -> |11>, leaving
    # |01> -> |01> and |10> -> |10> at the off-diagonal weight.
    bias_operator = np.full(shape=(4, 4), fill_value=np.sqrt(prob_bias / 3))
    np.fill_diagonal(bias_operator, np.sqrt(1 - prob_bias))

    return bias_operator.reshape(2, 2, 2, 2)


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
        sites_to_bias = list(range(0, mps.num_sites, 2))

    if not isinstance(prob_bias_list, List):
        prob_bias_list = [prob_bias_list for _ in range(len(sites_to_bias))]

    if any(site + 1 >= mps.num_sites for site in sites_to_bias):
        raise ValueError(
            "Each site in the list is the first of a two-site pair, so every "
            "entry needs a right neighbour within the chain."
        )
    if len(set(sites_to_bias)) != len(sites_to_bias):
        raise ValueError("The sites to bias should be distinct.")

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
        if renormalise:
            centre_norm = float(np.linalg.norm(mps.tensors[mps.orth_centre]))
            if centre_norm > 0:
                mps.tensors[mps.orth_centre] /= centre_norm

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
        the second one -- to Z stabilisers. Each stabiliser is spelled with the
        letters of its type: an X-type generator reads "X..X". (It used to emit
        the letters crossed, which the component swap in ``custom_code_checks``
        then silently expected -- issue #531.)
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
        stabilisers_x.append(_binary_to_pauli(binary_row, num_qubits, "X"))

    # Z stabilisers
    parity_matrix_z = code.z_stabs_binary()
    stabilisers_z = []
    for row in parity_matrix_z.rows():
        binary_row = np.zeros(num_qubits, dtype=int)
        for col in row:
            binary_row[col] = 1
        stabilisers_z.append(_binary_to_pauli(binary_row, num_qubits, "Z"))

    return stabilisers_x, stabilisers_z


def css_code_checks(
    code: CssCode, qubit_perm: Optional[np.ndarray] = None
) -> Tuple[List[List[int]]]:
    """
    Given a quantum CSS code, returns a list of its checks, where each check
    is represented as a list of indices of the bits adjacent to it.

    Parameters
    ----------
    code : qec.CssCode
        The CSS code object.
    qubit_perm : np.ndarray, optional
        Permutation array of length ``num_qubits`` such that ``qubit_perm[i]``
        is the original qubit index placed at MPS position ``i``.
        When provided, qubit indices in each check are remapped via the
        inverse permutation so that the returned MPS site indices reflect
        the new qubit ordering along the chain.

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

    offset = code.num_x_logicals() + code.num_z_logicals()

    if qubit_perm is not None:
        inv_perm = np.argsort(qubit_perm)
        checks_x = [
            sorted(list(2 * inv_perm[np.nonzero(row)[0]] + offset)) for row in array_x
        ]
        checks_z = [
            sorted(list(2 * inv_perm[np.nonzero(row)[0]] + offset + 1))
            for row in array_z
        ]
    else:
        checks_x = [list(2 * np.nonzero(row)[0] + offset) for row in array_x]
        checks_z = [list(2 * np.nonzero(row)[0] + offset + 1) for row in array_z]

    return checks_x, checks_z


def css_code_constraint_sites(
    code: CssCode, qubit_perm: Optional[np.ndarray] = None
) -> Tuple[List[List[List[int]]]]:
    """
    Returns the list of MPS sites where the logical constraints should be applied.

    Parameters
    ----------
    code : qec.CssCode
        CSS code object.
    qubit_perm : np.ndarray, optional
        Qubit permutation; see :func:`css_code_checks`.

    Returns
    -------
    sites : Tuple[List[List[List[int]]]]
        List of MPS sites.
    """

    checks_x, checks_z = css_code_checks(code, qubit_perm=qubit_perm)

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


def css_code_logicals(
    code: CssCode, qubit_perm: Optional[np.ndarray] = None
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Returns the list of MPS sites where the logical constraints should be applied.

    Parameters
    ----------
    code : qec.CssCode
        The CSS code object.
    qubit_perm : np.ndarray, optional
        Qubit permutation; see :func:`css_code_checks`.

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

    offset = code.num_x_logicals() + code.num_z_logicals()

    if qubit_perm is not None:
        inv_perm = np.argsort(qubit_perm)
        x_logicals = [
            sorted(list(2 * inv_perm[np.nonzero(row)[0]] + offset)) for row in array_x
        ]
        z_logicals = [
            sorted(list(2 * inv_perm[np.nonzero(row)[0]] + offset + 1))
            for row in array_z
        ]
    else:
        x_logicals = [list(2 * np.nonzero(row)[0] + offset) for row in array_x]
        z_logicals = [list(2 * np.nonzero(row)[0] + offset + 1) for row in array_z]

    return x_logicals, z_logicals


def css_code_logicals_sites(
    code: CssCode,
    qubit_perm: Optional[np.ndarray] = None,
) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    """
    Returns the list of MPS sites where the logical operators should be applied.

    Parameters
    ----------
    code : qec.CssCode
        CSS code object.
    qubit_perm : np.ndarray, optional
        Qubit permutation; see :func:`css_code_checks`.

    Returns
    -------
    strings : Tuple[List[List[List[int]]], List[List[List[int]]]]
        List of MPS sites.
    """

    sites_x, sites_z = css_code_logicals(code, qubit_perm=qubit_perm)

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


def _cross_pauli_letters(pauli_string: str) -> str:
    """Exchange X and Z letters in a Pauli string (Y, I and E unchanged).

    The decoders' internal convention couples each operator's parity check to
    its *own* component (an X-type check constrains the first component of
    every qubit pair, which is where :func:`pauli_to_mps` records an input
    ``X``). ``decode_css``'s stabiliser-multiplication retry was written
    against the letter-crossed strings ``css_code_stabilisers`` used to emit,
    so it crosses the honest strings back locally to keep its behaviour
    -- verified invariant to 1e-14 -- unchanged (issue #531).
    """
    table = {"X": "Z", "Z": "X"}
    return "".join(table.get(letter, letter) for letter in pauli_string)


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
        # mdopt's convention, here and in css_code_checks: a P-lettered
        # generator constrains the P-letter record -- an "X..X" string fixes
        # parities of the first component of each qubit pair, which is where
        # pauli_to_mps writes an input "X". This is NOT the textbook symplectic
        # pairing (where an X-type stabiliser would detect Z components); it is
        # the language the whole pipeline speaks: decode_css, the exact
        # brute-force reference, and the 3-qubit notebook -- which pairs
        # ["XXI", "IXX"] with "Bitflip" errors and validates against the
        # classical repetition-code curve 3p^2 - 2p^3, meaningful only under
        # this reading. The former component swap here compensated the
        # letter-crossed strings css_code_stabilisers used to emit; with
        # type-lettered strings it mirrored the wiring and broke gauge
        # invariance of the readout (issue #531): measured LER at p = 0.2 was
        # 0.404 against the analytic 0.104 this convention reproduces (0.102).
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

    # Each class is read off its own operator's components, matching the
    # constraint convention above: with checks on their own components, the
    # deformation space of a qubit's first component is null(H_X), whose gauge
    # directions all overlap supp(X-bar) evenly (they commute), while the
    # Z-bar direction overlaps it oddly -- so this labelling is exactly the
    # gauge-invariant one (issue #531).
    for logical in x_logicals:
        bitstring = pauli_to_mps(logical)
        # Find positions of non-zero entries
        x_sites = np.nonzero([int(bit) for bit in bitstring])[0]
        # Offset for X logicals
        x_sites += len(x_logicals) + len(z_logicals)
        logicals_x.append(list(x_sites))

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
                # Draw from the generator that was passed in, not the global
                # numpy one: using np.random here left the *positions* of the
                # errors reproducible while their Pauli types were not, so a
                # depolarising run could not be reproduced from its seed.
                error = rng.choice(["X", "Y", "Z"], p=[1 / 3, 1 / 3, 1 / 3])
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


def str_to_bool(value: str) -> bool:
    """Parse a command-line boolean.

    ``argparse(type=bool)`` calls ``bool()`` on the raw string, so every
    non-empty value -- including ``"false"`` -- comes back ``True``. Every
    cluster script passes ``--silent false``, which therefore silenced the run
    and suppressed exactly the diagnostics those runs were meant to surface.
    """
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised in {"true", "t", "yes", "y", "1"}:
        return True
    if normalised in {"false", "f", "no", "n", "0", ""}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _score_tie(is_map_identity: bool, degeneracy: int, tie_policy: str) -> float:
    """Turn "identity is among the maximisers" into a success score.

    A decoder that has to name one class picks arbitrarily among the maximisers,
    so it succeeds a fraction ``1/degeneracy`` of the time. "fractional" reports
    that expectation directly: unbiased, and being deterministic it has lower
    variance than actually sampling the tie-break. "optimistic" is the
    long-standing behaviour here and never scores a tie as a failure;
    "pessimistic" never scores one as a success.

    The choice is not a small one. Under depolarising noise a fifth to a half of
    shots have a degenerate MAP set and the policy moves the failure rate by
    tens of percent; under bit-flip noise the Z sector is frozen, no exact
    degeneracies arise, and all three policies agree.
    """
    if not is_map_identity:
        return 0.0
    if tie_policy == "optimistic":
        return 1.0
    if tie_policy == "fractional":
        return 1.0 / max(1, degeneracy)
    if tie_policy == "pessimistic":
        return 1.0 if degeneracy == 1 else 0.0
    raise ValueError(
        f"Unknown tie_policy {tie_policy!r}; expected 'optimistic', "
        "'fractional' or 'pessimistic'."
    )


def max_amplitude_bound(logical_mps) -> float:
    """Upper bound on ``max_s |<s|logical_mps>|``, in ``O(k d chi^2)``.

    Propagates absolute values along the chain, maximising over the physical
    index at each site. Since ``|sum_a v[a] A[a, b]| <= sum_a |v[a]| |A[a, b]|``
    this can only over-estimate, so the result is always a valid upper bound --
    no enumeration of the ``4^k`` classes and no variational search.

    The bound is *exact* when the logical MPS has no negative amplitudes, which
    is the case for an untruncated run: the initial state, the bias MPOs and the
    XOR/COPY/SWAP tensors all have non-negative entries, and marginalisation
    traces against all-ones, so nothing can cancel. Truncation breaks that --
    the best rank-``chi`` approximation of a non-negative object need not be
    non-negative -- and then the bound is only an upper bound. In practice the
    negative amplitudes that appear are small and the bound has stayed tight,
    but that is an observation rather than a guarantee, so treat it as a bound.
    """
    vector = np.array([1.0])
    for tensor in logical_mps.tensors:
        magnitudes = np.abs(np.asarray(tensor, dtype=float))
        vector = np.einsum("a,apb->pb", vector, magnitudes).max(axis=0)
    return float(vector.max())


def max_product_readout(logical_mps, beam_width: int = 4):
    """Search the logical MPS for the largest-amplitude basis state, by beam search.

    Runs a max-product pass along the chain, keeping the ``beam_width`` best
    partial bitstrings at each site. Because the pass works on ``|A|`` the score
    it carries is only an over-estimate, so each surviving candidate is scored
    exactly at the end with a signed inner product and the best is returned.

    Deterministic and free of local minima in the sense that matters here: it
    never has to escape a basin, it simply keeps several. Paired with
    :func:`max_amplitude_bound` it usually settles the readout outright -- when
    the returned amplitude meets the bound, no basis state can do better and the
    answer is proven optimal without touching DMRG.

    Returns
    -------
    bitstring : str
        The best basis state found.
    amplitude : float
        ``|<bitstring|logical_mps>|``, evaluated exactly.
    """
    beams = [(np.array([1.0]), "")]
    for tensor in logical_mps.tensors:
        magnitudes = np.abs(np.asarray(tensor, dtype=float))
        candidates = []
        for vector, bits in beams:
            for value in range(magnitudes.shape[1]):
                candidates.append((vector @ magnitudes[:, value, :], bits + str(value)))
        candidates.sort(key=lambda item: -float(item[0].max()))
        beams = candidates[: max(1, beam_width)]

    best_bits, best_amplitude = None, -1.0
    for _, bits in beams:
        amplitude = abs(inner_product(create_custom_product_state(bits), logical_mps))
        if amplitude > best_amplitude:
            best_bits, best_amplitude = bits, amplitude
    return best_bits, best_amplitude


class _ReadoutResult:
    """What the logical readout settled on.

    Mirrors the two attributes callers used to reach for on the DMRG engine, so
    a readout decided by beam search can be returned in its place.
    """

    def __init__(self, mps, mps_target):
        self.mps = mps
        self.mps_target = mps_target


def _logical_readout(
    logical_mps,
    num_sites: int,
    chi_max: int,
    cut: float,
    num_runs: int,
    num_restarts: int,
    silent: bool,
    beam_width: int = 4,
):
    """Find the largest-amplitude computational basis state of ``logical_mps``.

    Tries the cheap route first. :func:`max_product_readout` proposes a basis
    state and :func:`max_amplitude_bound` caps what any basis state could reach;
    when the two agree the proposal is provably optimal and the search is over,
    at ``O(k d chi^2)`` and with no variational sweep at all. Only when that
    bracket stays open does Dephasing DMRG run, and its answer is then taken
    together with the beam-search one, so the result is never worse than DMRG
    alone.

    Returns
    -------
    result : _ReadoutResult or DephasingDMRG
        Carries ``mps`` (the chosen basis state) and ``mps_target``.
    amplitude : float
        ``|<s*|logical_mps>|`` for the best basis state found.
    certified : bool
        Whether the amplitude provably equals the maximum over all basis states.
    """
    bound = max_amplitude_bound(logical_mps)
    bits, amplitude = max_product_readout(logical_mps, beam_width=beam_width)

    if amplitude >= bound * (1 - 1e-9):
        if not silent:
            logging.info(
                "Readout settled by beam search: |<s*|psi>| = %.6e matches the "
                "upper bound, so no basis state can do better.",
                amplitude,
            )
        return (
            _ReadoutResult(create_custom_product_state(bits), logical_mps),
            amplitude,
            True,
        )

    if not silent:
        logging.info(
            "Beam search reached %.6e against an upper bound of %.6e; "
            "falling back to Dephasing DMRG.",
            amplitude,
            bound,
        )
    try:
        engine, dmrg_amplitude = _dmrg_readout(
            logical_mps, num_sites, chi_max, cut, num_runs, num_restarts, silent
        )
    except Exception as error:  # pylint: disable=broad-except
        # ARPACK refuses to start ("error -9: Starting vector is zero") when
        # either the starting vector or the operator itself is numerically zero;
        # the message names only the first. First measured on [[4,2,2]] at
        # chi_max<=3, which is why this was once described as unreachable in
        # production -- that was wrong. It ended full-scale classical_ldpc runs
        # at chi_max=128 twice, through the classical decode_message path, and
        # the cause was the zero operator rather than the starting vector.
        # Both are now guarded at source in optimiser/dmrg.py and
        # dephasing_dmrg.py; this stays as a backstop for anything else.
        #
        # It is cheap insurance rather than a correction: the beam search has
        # already produced a valid basis state and its exact amplitude, so a
        # solver failure can degrade to that instead of taking the shot down
        # and being recorded as a decoding failure by the experiment drivers.
        if not silent:
            logging.warning(
                "Dephasing DMRG failed (%s: %s); keeping the beam-search "
                "result |<s*|psi>| = %.6e, which is not certified optimal.",
                type(error).__name__,
                str(error)[:80],
                amplitude,
            )
        return (
            _ReadoutResult(create_custom_product_state(bits), logical_mps),
            amplitude,
            False,
        )

    if dmrg_amplitude >= amplitude:
        return engine, dmrg_amplitude, dmrg_amplitude >= bound * (1 - 1e-9)
    return (
        _ReadoutResult(create_custom_product_state(bits), logical_mps),
        amplitude,
        False,
    )


def _dmrg_readout(
    logical_mps,
    num_sites: int,
    chi_max: int,
    cut: float,
    num_runs: int,
    num_restarts: int,
    silent: bool,
):
    """Find the largest-amplitude computational basis state by Dephasing DMRG.

    Dephasing DMRG is a local optimiser, so a single sweep from a single start
    often settles on a class that is not the global maximum. That matters here
    because the success test compares the identity's amplitude against whatever
    DMRG returned: if DMRG stops short, the identity can clear a bar that the
    true maximiser would not have, turning a logical error into a reported
    success. Restarting from several product states and keeping the best result
    makes that far less likely; empirically a single start recovers the true
    maximum only when it exceeds the runner-up by roughly a factor of five,
    whereas eight starts stay reliable well below that.

    The all-zeros state is always among the starts, so the identity class is
    never missed for want of a lucky initialisation.

    Returns
    -------
    engine : DephasingDMRG
        The engine of the best restart.
    best_amplitude : float
        ``|<s*|logical_mps>|`` for the best basis state found.
    """
    rng = np.random.default_rng(0)
    best_engine, best_amplitude = None, -1.0

    for restart in range(max(1, num_restarts)):
        if restart == 0:
            start = create_simple_product_state(num_sites, which="+")
        elif restart == 1:
            start = create_simple_product_state(num_sites, which="0")
        else:
            start = create_custom_product_state(
                "".join(rng.choice(["0", "1"], size=num_sites))
            )
        engine = DephasingDMRG(
            mps=start,
            mps_target=logical_mps,
            chi_max=chi_max,
            cut=cut,
            mode="LA",
            silent=True,
        )
        engine.run(num_iter=num_runs)
        amplitude = abs(inner_product(engine.mps, logical_mps))
        if amplitude > best_amplitude:
            best_engine, best_amplitude = engine, amplitude

    if not silent:
        logging.info(
            "Dephasing DMRG: best of %d restart(s), |<s*|psi>| = %.6e",
            max(1, num_restarts),
            best_amplitude,
        )
    return best_engine, best_amplitude


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
    # which finds the computational basis product state
    # contributing the most to a given MPDO,
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
    qubit_order_strategy: str = "Natural",
    optimiser: str = "Dephasing DMRG",
    tolerance: float = float(1e-12),
    dense_readout_max_sites: int = 30,
    num_restarts: int = 8,
    tie_policy: str = "optimistic",
    rng: Optional[np.random.Generator] = None,
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
    qubit_order_strategy : str
        Strategy for ordering qubits along the MPS chain.
        ``"Natural"`` keeps the original column order of the parity check matrix.
        ``"Optimised"`` applies the Reverse Cuthill-McKee algorithm to the qubit
        interaction graph to reduce the bandwidth of each MPO constraint string,
        lowering the required bond dimension.
    optimiser : str
        The optimiser used to find the closest basis product state to a given MPDO.
        Available options: "Dephasing DMRG", "Dense", "Optima TT".
    tolerance : float
        The tolerance for the MPS classes.
    dense_readout_max_sites : int
        Read the logical class out by dense contraction while the logical MPS has
        at most this many sites, and by Dephasing DMRG beyond it. The default
        keeps the usual behaviour; setting it to 0 forces the DMRG path, which is
        how the two readouts can be compared on a code small enough to check.
    rng : np.random.Generator, optional
        Source of randomness for ``multiply_by_stabiliser``. Pass the same
        generator used to sample the error if the run needs to be reproducible;
        when omitted a fresh generator is created and the choice will differ
        between runs.

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
        # Deliberate fast path: low-p Monte Carlo is dominated by no-error
        # shots, and the identity class is provably the MAP answer for a
        # trivial error (verified by exact enumeration up to p = 0.49). The
        # returned vector is a k = 1-shaped STUB, not a real posterior -- do
        # not "fix" it to 2**(2k) entries, which would allocate 128 MB and cost
        # ~10 ms per shot on a k = 12 BB code, on the hot path this exists to
        # skip. Callers here consume only the success flag.
        return [1.0, 0.0, 0.0, 0.0], 1

    # Both sectors always carry a live degree of freedom: the bias channel
    # spreads amplitude onto every physical site regardless of which Paulis the
    # input error happens to contain, so a sector whose constraints are skipped
    # keeps its logical site in |+> and marginalises to a uniform -- and
    # therefore meaningless -- distribution over that sector's classes.
    error_contains_x = True
    error_contains_z = True

    erased_qubits = [
        index for index, single_error in enumerate(error) if single_error == "E"
    ]

    if multiply_by_stabiliser and not erased_qubits:
        # Draw from the generator that was passed in. Reaching for np.random
        # here would make the choice depend on global state, so a run using this
        # path could not be reproduced from its seed.
        generator = np.random.default_rng() if rng is None else rng
        stabilisers_x, stabilisers_z = css_code_stabilisers(code)
        # decode_css's constraint wiring couples each check to its own
        # component, so the invariant retry direction is the letter-crossed
        # string -- see _cross_pauli_letters. This preserves the behaviour the
        # invariance test pins to 1e-9 (and chi-convergence measured at 1e-14).
        stabilisers = [
            _cross_pauli_letters(stabiliser)
            for stabiliser in stabilisers_x + stabilisers_z
        ]
        error = multiply_pauli_strings(error, str(generator.choice(stabilisers)))

    # Compute qubit permutation that minimises MPO bandwidth.
    if qubit_order_strategy == "Optimised":
        pm_x = code.x_stabs_binary()
        H_x = np.zeros((pm_x.num_rows(), pm_x.num_columns()), dtype=int)
        for r, cols in enumerate(pm_x.rows()):
            for c in cols:
                H_x[r, c] = 1
        pm_z = code.z_stabs_binary()
        H_z = np.zeros((pm_z.num_rows(), pm_z.num_columns()), dtype=int)
        for r, cols in enumerate(pm_z.rows()):
            for c in cols:
                H_z[r, c] = 1
        qubit_perm = optimise_qubit_order(np.vstack([H_x, H_z]))
        # Rearrange the error string so that MPS site i carries the Pauli
        # of original qubit qubit_perm[i].
        error = "".join(error[qubit_perm[i]] for i in range(len(error)))
        if not silent:
            logging.info("Applied optimised qubit ordering.")
    else:
        qubit_perm = None

    # Recompute the erased positions in the MPS frame: the reordering above
    # moves qubits along the chain, so the indices collected from the original
    # error string no longer point at the right sites.
    erased_qubits = [
        index for index, single_error in enumerate(error) if single_error == "E"
    ]

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

    constraint_sites = css_code_constraint_sites(code, qubit_perm=qubit_perm)
    logicals_sites = css_code_logicals_sites(code, qubit_perm=qubit_perm)

    # Exclude erased qubits from the bias: they are initialised as |+>, which
    # already represents complete ignorance, so biasing them would corrupt that
    # state. Physical qubit q occupies MPS sites (num_logicals + 2q) and
    # (num_logicals + 2q + 1).
    #
    # The bit-flip bias is a one-site MPO, so it wants every site. The
    # depolarising bias is a two-site MPO acting jointly on a qubit's pair, so
    # it wants only the first site of each pair -- passing both would apply it
    # to overlapping pairs and the result would not be a depolarising channel.
    unerased = [q for q in range(len(code)) if q not in erased_qubits]
    if bias_type == "Bitflip":
        sites_to_bias = [
            s
            for q in unerased
            for s in (num_logicals + 2 * q, num_logicals + 2 * q + 1)
        ]
    else:
        sites_to_bias = [num_logicals + 2 * q for q in unerased]

    if sites_to_bias:
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

    if not silent:
        logging.info("Marginalising the error MPS.")
    # Marginalise ALL physical qubit sites in one pass.  Erased qubits are
    # already in |+> and are naturally included here -- no separate
    # intermediate marginalisation is needed.
    sites_to_marginalise = list(range(num_logicals, num_sites))
    logical_mps = error_mps.marginal(
        sites_to_marginalise=sites_to_marginalise, renormalise=renormalise
    ).reverse()

    num_logical_sites = len(logical_mps)
    if not silent:
        logging.info(f"The number of logical sites: {num_logical_sites}.")

    if num_logical_sites <= dense_readout_max_sites:
        logical_signed = logical_mps.dense(
            flatten=True, renormalise=renormalise, norm=2
        )
        logical_dense = abs(logical_signed)

        # An exact run cannot produce a negative amplitude: every tensor in the
        # pipeline is non-negative and marginalisation traces against all-ones.
        # A negative one is therefore a truncation artefact and a direct signal
        # that chi_max is too small for this instance -- the cheapest
        # convergence diagnostic available, since the vector is already here.
        most_negative = float(np.min(np.real(np.asarray(logical_signed))))
        peak = float(np.max(logical_dense))

        # A collapsed posterior carries no information. Truncation is what
        # destroys it: at low chi_max a whole site tensor can be driven to zero.
        if not np.isfinite(peak) or peak == 0.0:
            # Scoring must stop here. Every entry of an all-zero vector is within
            # eps of the maximum, so the identity would be "among the maximisers"
            # and the shot would score a success -- turning numerical collapse
            # into a correctly decoded shot and biasing the failure rate
            # downward, invisibly when silent=True. Report the failure instead.
            if not silent:
                logging.warning(
                    "The logical posterior collapsed to zero at chi_max=%d; this "
                    "shot carries no information and is scored as a failure.",
                    chi_max,
                )
            return logical_dense, 0.0
        if most_negative < -1e-12 * max(peak, 1.0) and not silent:
            logging.warning(
                "Negative logical amplitude %.3e (%.1f%% of the peak): chi_max=%d "
                "is not converged for this instance.",
                most_negative,
                100.0 * abs(most_negative) / peak,
                chi_max,
            )

        # Normalise to the peak so that tie tolerances are scale-independent.
        # Partially underflowed vectors (peak ~1e-200) would otherwise pass the
        # collapse guard but have every entry within the fixed 1e-12 absolute
        # tolerance of the maximum, marking all classes as tied.
        logical_normed = logical_dense / peak

        # Global maximum amplitude (always 1.0 after normalisation)
        max_amp = np.max(logical_normed)

        # Machine-precision–level tolerance (relative + absolute)
        rel_tol = 1e-9
        abs_tol = 1e-12
        eps = max(rel_tol * max_amp, abs_tol)

        # Success <=> the identity logical is in the MAP set (degeneracy allowed).
        #
        # Note the asymmetry with the Dephasing DMRG branch below: when several
        # classes tie for the maximum this counts as a success, whereas DMRG
        # returns whichever tied class its sweep lands on and so usually scores
        # the same shot as a failure. Small codes read out densely and large ones
        # by DMRG, so the two regimes apply different conventions to degenerate
        # posteriors; a fully uniform posterior always reads as success here.
        is_map_identity = logical_normed[0] >= max_amp - eps
        degeneracy = int(np.count_nonzero(logical_normed >= max_amp - eps))
        score = _score_tie(is_map_identity, degeneracy, tie_policy)

        if degeneracy > 1 and not silent:
            logging.warning(
                "The MAP set is %d-fold degenerate; scored under the '%s' "
                "policy as %.4f.",
                degeneracy,
                tie_policy,
                score,
            )

        result = logical_dense, score
        return result
        # Encoding: 0 -> I, 1 -> X, 2 -> Z, 3 -> Y, where the number is np.argmax(logical_dense).

    if optimiser == "Optima TT":
        raise NotImplementedError("Optima TT is not implemented yet.")
    if optimiser != "Dephasing DMRG":
        raise ValueError("Invalid optimiser chosen.")
    if tie_policy != "optimistic":
        raise NotImplementedError(
            f"tie_policy={tie_policy!r} is not supported on the Dephasing DMRG "
            "path; the DMRG readout does not enumerate all tied classes and "
            "therefore cannot implement 'fractional' or 'pessimistic' scoring. "
            "Use tie_policy='optimistic' or reduce dense_readout_max_sites so "
            "that the dense branch is taken instead."
        )

    if not silent:
        logging.info("Reading out the logical class.")
    engine, amplitude_found, certified = _logical_readout(
        logical_mps,
        num_logical_sites,
        chi_max,
        cut,
        num_runs,
        num_restarts,
        silent,
    )

    # DMRG returns a single basis state, so on a degenerate posterior it lands on
    # whichever tied class its sweep reached first. Comparing only that state
    # with the identity would then score a tie as a failure, while the dense
    # branch above scores the same shot as a success. Instead ask the question
    # dense readout asks -- is the identity among the maximisers? -- by pulling
    # both amplitudes out of the logical MPS directly, which costs O(k chi^2) and
    # needs no enumeration of the 4^k classes.
    mps_dmrg_target = create_simple_product_state(num_logical_sites, which="0")
    amplitude_identity = abs(inner_product(mps_dmrg_target, logical_mps))

    # Collapse check: treat zero or non-finite amplitudes as posterior collapse,
    # mirroring the dense branch.
    if not np.isfinite(amplitude_found) or amplitude_found == 0.0:
        if not silent:
            logging.warning(
                "The logical posterior collapsed to zero at chi_max=%d; this shot "
                "carries no information and is scored as a failure.",
                chi_max,
            )
        return engine, 0

    # Compare amplitudes on a unit scale (normalised by the DMRG maximum) so
    # that the fixed tolerance is meaningful regardless of overall scale.
    normed_identity = amplitude_identity / amplitude_found
    eps = 1e-9
    is_map_identity = normed_identity >= 1.0 - eps

    bound = max_amplitude_bound(logical_mps)
    if not silent and not certified:
        if amplitude_found < bound * (1 - 1e-6):
            logging.warning(
                "Dephasing DMRG reached %.6e but the maximum is at most %.6e; "
                "the sweep may have stopped at a local optimum. Consider raising "
                "num_restarts or num_runs.",
                amplitude_found,
                bound,
            )
        if is_map_identity and amplitude_identity < bound * (1 - 1e-6):
            logging.warning(
                "Success here rests on DMRG's estimate: the identity amplitude "
                "%.6e clears |<s*|psi>| but not the upper bound %.6e.",
                amplitude_identity,
                bound,
            )

    if not silent:
        logging.info(
            "Dephasing DMRG finished: |<s*|psi>| = %.6e, |<0|psi>| = %.6e, "
            "identity in the MAP set: %s",
            amplitude_found,
            amplitude_identity,
            bool(is_map_identity),
        )
    return engine, int(is_map_identity)


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
    dense_readout_max_sites: int = 12,
    num_restarts: int = 8,
    tie_policy: str = "optimistic",
    rng: Optional[np.random.Generator] = None,
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
    dense_readout_max_sites : int
        Read the logical class out by dense contraction while the logical MPS has
        at most this many sites, and by Dephasing DMRG beyond it. See
        :func:`decode_css`.

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
        # Deliberate fast path: low-p Monte Carlo is dominated by no-error
        # shots, and the identity class is provably the MAP answer for a
        # trivial error (verified by exact enumeration up to p = 0.49). The
        # returned vector is a k = 1-shaped STUB, not a real posterior -- do
        # not "fix" it to 2**(2k) entries, which would allocate 128 MB and cost
        # ~10 ms per shot on a k = 12 BB code, on the hot path this exists to
        # skip. Callers here consume only the success flag.
        return [1.0, 0.0, 0.0, 0.0], 1

    erased_qubits = [
        index for index, single_error in enumerate(error) if single_error == "E"
    ]

    if multiply_by_stabiliser and not erased_qubits:
        # See decode_css: the choice must come from the passed-in generator so
        # the run stays reproducible from its seed. And as there, the
        # gauge-preserving direction is the letter-CROSSED string: crossing
        # flips the first component over the Z-part of the stabiliser and the
        # second over its X-part, so the enforced parities are preserved exactly
        # when the symplectic product with every generator vanishes -- which is
        # stabiliser commutation itself, so this holds for non-CSS codes too.
        # Multiplying by the uncrossed string instead requires the Euclidean
        # products to vanish, which non-self-orthogonal generators (Shor's
        # X-type rows overlap in three positions) do not satisfy.
        generator = np.random.default_rng() if rng is None else rng
        chosen_stabiliser = str(generator.choice(stabilizers))
        error = multiply_pauli_strings(error, _cross_pauli_letters(chosen_stabiliser))

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

    # Exclude erased qubit sites from bias: erased qubits are initialised as
    # |+>, which already represents complete ignorance, so biasing them would
    # corrupt that state.  Each physical qubit q occupies MPS sites
    # (num_logicals + 2q) and (num_logicals + 2q + 1).
    # See decode_css: the one-site bit-flip bias wants every site, the two-site
    # depolarising bias only the first site of each qubit pair.
    num_qubits = len(stabilizers[0])
    unerased = [q for q in range(num_qubits) if q not in erased_qubits]
    if bias_type == "Bitflip":
        sites_to_bias = [
            s
            for q in unerased
            for s in (num_logicals + 2 * q, num_logicals + 2 * q + 1)
        ]
    else:
        sites_to_bias = [num_logicals + 2 * q for q in unerased]

    if sites_to_bias:
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

    if not silent:
        logging.info("Marginalising the error MPS.")
    # Marginalise ALL physical qubit sites in one pass.  Erased qubits are
    # already in |+> and are naturally included here — no separate
    # intermediate marginalization is needed.
    sites_to_marginalise = list(range(num_logicals, num_sites))
    logical_mps = error_mps.marginal(
        sites_to_marginalise=sites_to_marginalise, renormalise=renormalise
    ).reverse()

    num_logical_sites = len(logical_mps)
    if not silent:
        logging.info(f"The number of logical sites: {num_logical_sites}.")

    if num_logical_sites <= dense_readout_max_sites:
        logical_signed = logical_mps.dense(
            flatten=True, renormalise=renormalise, norm=2
        )
        logical_dense = abs(logical_signed)

        # An exact run cannot produce a negative amplitude: every tensor in the
        # pipeline is non-negative and marginalisation traces against all-ones.
        # A negative one is therefore a truncation artefact and a direct signal
        # that chi_max is too small for this instance -- the cheapest
        # convergence diagnostic available, since the vector is already here.
        most_negative = float(np.min(np.real(np.asarray(logical_signed))))
        peak = float(np.max(logical_dense))

        # A collapsed posterior carries no information. Truncation is what
        # destroys it: at low chi_max a whole site tensor can be driven to zero.
        if not np.isfinite(peak) or peak == 0.0:
            # Scoring must stop here. Every entry of an all-zero vector is within
            # eps of the maximum, so the identity would be "among the maximisers"
            # and the shot would score a success -- turning numerical collapse
            # into a correctly decoded shot and biasing the failure rate
            # downward, invisibly when silent=True. Report the failure instead.
            if not silent:
                logging.warning(
                    "The logical posterior collapsed to zero at chi_max=%d; this "
                    "shot carries no information and is scored as a failure.",
                    chi_max,
                )
            return logical_dense, 0.0
        if most_negative < -1e-12 * max(peak, 1.0) and not silent:
            logging.warning(
                "Negative logical amplitude %.3e (%.1f%% of the peak): chi_max=%d "
                "is not converged for this instance.",
                most_negative,
                100.0 * abs(most_negative) / peak,
                chi_max,
            )

        # Normalise to the peak so that tie tolerances are scale-independent.
        # Partially underflowed vectors (peak ~1e-200) would otherwise pass the
        # collapse guard but have every entry within the fixed 1e-12 absolute
        # tolerance of the maximum, marking all classes as tied.
        logical_normed = logical_dense / peak

        # find global maximum amplitude (always 1.0 after normalisation)
        max_amp = np.max(logical_normed)

        # treat identity logical as success if it is among the maximisers
        # (within some numerical tolerance)
        # Same tolerance as decode_css, so both decoders call a tie the same way.
        eps = max(1e-9 * max_amp, 1e-12)
        is_map_identity = logical_normed[0] >= max_amp - eps
        degeneracy = int(np.count_nonzero(logical_normed >= max_amp - eps))
        score = _score_tie(is_map_identity, degeneracy, tie_policy)

        if degeneracy > 1 and not silent:
            logging.warning(
                "The MAP set is %d-fold degenerate; scored under the '%s' "
                "policy as %.4f.",
                degeneracy,
                tie_policy,
                score,
            )

        result = logical_dense, score
        return result
        # Encoding: 0 -> I, 1 -> X, 2 -> Z, 3 -> Y, where the number is np.argmax(logical_dense).

    if optimiser == "Optima TT":
        raise NotImplementedError("Optima TT is not implemented yet.")
    if optimiser != "Dephasing DMRG":
        raise ValueError("Invalid optimiser chosen.")
    if tie_policy != "optimistic":
        raise NotImplementedError(
            f"tie_policy={tie_policy!r} is not supported on the Dephasing DMRG "
            "path; the DMRG readout does not enumerate all tied classes and "
            "therefore cannot implement 'fractional' or 'pessimistic' scoring. "
            "Use tie_policy='optimistic' or reduce dense_readout_max_sites so "
            "that the dense branch is taken instead."
        )

    if not silent:
        logging.info("Reading out the logical class.")
    engine, amplitude_found, certified = _logical_readout(
        logical_mps,
        num_logical_sites,
        chi_max,
        cut,
        num_runs,
        num_restarts,
        silent,
    )

    # DMRG returns a single basis state, so on a degenerate posterior it lands on
    # whichever tied class its sweep reached first. Comparing only that state
    # with the identity would then score a tie as a failure, while the dense
    # branch above scores the same shot as a success. Instead ask the question
    # dense readout asks -- is the identity among the maximisers? -- by pulling
    # both amplitudes out of the logical MPS directly, which costs O(k chi^2) and
    # needs no enumeration of the 4^k classes.
    mps_dmrg_target = create_simple_product_state(num_logical_sites, which="0")
    amplitude_identity = abs(inner_product(mps_dmrg_target, logical_mps))

    # Collapse check: treat zero or non-finite amplitudes as posterior collapse,
    # mirroring the dense branch.
    if not np.isfinite(amplitude_found) or amplitude_found == 0.0:
        if not silent:
            logging.warning(
                "The logical posterior collapsed to zero at chi_max=%d; this shot "
                "carries no information and is scored as a failure.",
                chi_max,
            )
        return engine, 0

    # Compare amplitudes on a unit scale (normalised by the DMRG maximum) so
    # that the fixed tolerance is meaningful regardless of overall scale.
    normed_identity = amplitude_identity / amplitude_found
    eps = 1e-9
    is_map_identity = normed_identity >= 1.0 - eps

    bound = max_amplitude_bound(logical_mps)
    if not silent and not certified:
        if amplitude_found < bound * (1 - 1e-6):
            logging.warning(
                "Dephasing DMRG reached %.6e but the maximum is at most %.6e; "
                "the sweep may have stopped at a local optimum. Consider raising "
                "num_restarts or num_runs.",
                amplitude_found,
                bound,
            )
        if is_map_identity and amplitude_identity < bound * (1 - 1e-6):
            logging.warning(
                "Success here rests on DMRG's estimate: the identity amplitude "
                "%.6e clears |<s*|psi>| but not the upper bound %.6e.",
                amplitude_identity,
                bound,
            )

    if not silent:
        logging.info(
            "Dephasing DMRG finished: |<s*|psi>| = %.6e, |<0|psi>| = %.6e, "
            "identity in the MAP set: %s",
            amplitude_found,
            amplitude_identity,
            bool(is_map_identity),
        )
    return engine, int(is_map_identity)
