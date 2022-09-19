"""
The ``mdopt.mps.utils`` module.
===========================================
This module contains various MPS utilities.
"""

from typing import Union, Optional
from functools import reduce
import numpy as np
from opt_einsum import contract
from mdopt.utils.utils import svd
from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.explicit import ExplicitMPS


def create_state_vector(num_sites: np.int16, phys_dim: np.int16 = 2) -> np.ndarray:
    """
    Creates a uniform random complex quantum state in the form of a state vector.

    Parameters
    ----------
    num_sites : np.int16
        Number of degrees of freedom.
    phys_dim : np.int16
        Number of dimensions of each degree of freedom.

    Returns
    -------
    state_vector
        The resulting state vector.

    """

    state_vector = np.random.uniform(
        size=(phys_dim**num_sites)
    ) + 1j * np.random.uniform(size=(phys_dim**num_sites))
    state_vector /= np.linalg.norm(state_vector)

    return state_vector


def find_orth_centre(
    mps: CanonicalMPS, return_orth_flags: bool = False, tolerance: np.float64 = 1e-12
):
    """
    Returns a list of integers corresponding to positions of orthogonality centres of an MPS.

    Parameters
    ----------
    mps : CanonicalMPS
        The MPS to find the orthogonality centre(s) in.
    return_orth_flags : bool
        Whether to return if each tensor is a right or a left isometry.
    tolerance : np.float64
        Numerical tolerance for checking the isometry property.

    Returns
    -------
    orth_centres
        Indices of sites at which tensors are orthogonality centres.
    orth_flags_left
        Boolean variables for each tensor corresponding to being a left isometry.
    orth_flags_right
        Boolean variables for each tensor corresponding to being a right isometry.

    Raises
    ------
    ValueError
        If an :class:`ExplicitMPS' instance is passed as an input.
        These do not have orthogonality centres by definition.
    """

    if isinstance(mps, ExplicitMPS):
        raise ValueError(
            "Orthogonality centre is undefined for an Explicit MPS instance."
        )

    num_sites = len(mps)

    orth_flags_left = []
    orth_flags_right = []
    orth_centres = []

    for i, tensor in enumerate(mps.tensors):

        to_be_identity_left = contract(
            "ijk, ijl -> kl", tensor, np.conjugate(tensor), optimize=[(0, 1)]
        )
        to_be_identity_right = contract(
            "ijk, ljk -> il", tensor, np.conjugate(tensor), optimize=[(0, 1)]
        )

        identity_left = np.identity(to_be_identity_left.shape[0])
        identity_right = np.identity(to_be_identity_right.shape[0])

        norm_left = np.linalg.norm(to_be_identity_left - identity_left)
        norm_right = np.linalg.norm(to_be_identity_right - identity_right)

        orth_flags_left.append(np.isclose(norm_left, 0, atol=tolerance))
        orth_flags_right.append(np.isclose(norm_right, 0, atol=tolerance))

        if not (np.isclose(norm_left, 0, atol=tolerance)) and not (
            np.isclose(norm_right, 0, atol=tolerance)
        ):
            orth_centres.append(i)

    # Handling exceptions, right- and left-canonical forms, and cases
    # when the orthogonality centre might be left- or right- isometry at
    # the boundaries while all the other tensors are the opposite isometries.
    if orth_flags_left in ([True] + [False] * (num_sites - 1), [False] * num_sites):
        if orth_flags_right == [not flag for flag in orth_flags_left]:
            orth_centres.append(0)

    if orth_flags_left in ([True] * (num_sites - 1) + [False], [True] * num_sites):
        if orth_flags_right == [not flag for flag in orth_flags_left]:
            orth_centres.append(num_sites - 1)

    # Handling a product state.
    if (orth_flags_left == [True] * num_sites) and (
        orth_flags_right == [True] * num_sites
    ):
        orth_centres.append(0)

    if return_orth_flags:
        return orth_centres, orth_flags_left, orth_flags_right

    return orth_centres


def is_canonical(mps: CanonicalMPS, tolerance: np.float64 = 1e-12):
    """Checks if the MPS is in any of the canonical forms."""

    if isinstance(mps, ExplicitMPS):
        raise ValueError(
            "Orthogonality centre is undefined for an Explicit MPS instance."
        )

    flags_left = []
    flags_right = []
    for _, tensor in enumerate(mps.tensors):

        to_be_identity_left = contract(
            "ijk, ijl -> kl", tensor, np.conjugate(tensor), optimize=[(0, 1)]
        )
        to_be_identity_right = contract(
            "ijk, ljk -> il", tensor, np.conjugate(tensor), optimize=[(0, 1)]
        )

        identity_left = np.identity(to_be_identity_left.shape[0], dtype=np.float64)
        identity_right = np.identity(to_be_identity_right.shape[0], dtype=np.float64)

        flags_left.append(
            np.isclose(
                np.linalg.norm(to_be_identity_left - identity_left), 0, atol=tolerance
            )
        )
        flags_right.append(
            np.isclose(
                np.linalg.norm(to_be_identity_right - identity_right), 0, atol=tolerance
            )
        )

    if np.array(flags_left).all() or np.array(flags_right).all():
        return True

    orth_centres = find_orth_centre(mps)

    return len(orth_centres) == 1


def inner_product(
    mps_1: Union[CanonicalMPS, ExplicitMPS], mps_2: Union[CanonicalMPS, ExplicitMPS]
) -> Union[np.float64, np.complex128]:
    """Returns an inner product between 2 matrix product states."""

    if len(mps_1) != len(mps_2):
        raise ValueError(
            f"The number of sites in the first MPS is {len(mps_1)} while "
            f"the number of sites in the second MPS is {len(mps_2)}. "
            "The MPS's must be of equal length."
        )

    num_sites = len(mps_1)

    mps_1 = mps_1.conjugate()

    if isinstance(mps_1, ExplicitMPS):
        mps_1 = mps_1.right_canonical()
    if isinstance(mps_2, ExplicitMPS):
        mps_2 = mps_2.right_canonical()

    tensors = []

    for i in range(num_sites):

        dims_1 = mps_1.tensors[i].shape
        dims_2 = mps_2.tensors[i].shape

        tensors.append(
            contract(
                "ijk, ljm -> ilkm",
                mps_1.tensors[i],
                mps_2.tensors[i],
                optimize=[(0, 1)],
            ).reshape((dims_1[0] * dims_2[0], dims_1[2] * dims_2[2]))
        )

    product = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), tensors)

    return product[0][0]


def mps_from_dense(
    state_vector: np.ndarray,
    phys_dim: np.int16 = 2,
    chi_max: np.int16 = 1e4,
    tolerance: np.float64 = 1e-12,
    form: str = "Explicit",
    orth_centre: Optional[np.int16] = None,
) -> Union[CanonicalMPS, ExplicitMPS]:
    """
    Returns the Matrix Product State,
    given a state in the dense (state-vector) form.

    Parameters
        state_vector:
            The state vector.
        phys_dim:
            Dimensionality of the local Hilbert space.
        limit_max:
            Activate an upper limit to the spectrum's size.
        chi_max:
            Maximum number of singular values to keep.
        tolerance:
            Absolute tolerance of the normalisation of the singular value spectrum at each bond.

    Returns
        mps(tensors, singular_values):
    """

    state_vector = np.copy(state_vector)

    if state_vector.flatten().shape[0] % phys_dim != 0:
        raise ValueError(
            "The dimension of the flattened state vector is incorrect "
            "(does not correspond to the product of local dimensions)."
        )

    tensors = []
    singular_values = []

    state_vector = state_vector.reshape((-1, phys_dim))

    state_vector, singular_values_local, v_r = svd(
        state_vector, chi_max=chi_max, renormalise=False
    )

    tensors.append(np.expand_dims(v_r, -1))
    singular_values.append(singular_values_local)

    while state_vector.shape[0] >= phys_dim:

        state_vector = np.matmul(state_vector, np.diag(singular_values_local))

        bond_dim = state_vector.shape[-1]
        state_vector = state_vector.reshape((-1, phys_dim * bond_dim))
        state_vector, singular_values_local, v_r = svd(
            state_vector, chi_max=chi_max, renormalise=False
        )
        v_r = v_r.reshape((-1, phys_dim, bond_dim))

        tensors.insert(0, v_r)
        singular_values.insert(0, singular_values_local)

    singular_values.append(np.array([1.0]))

    for i, _ in enumerate(tensors):

        tensors[i] = np.tensordot(
            tensors[i], np.linalg.inv(np.diag(singular_values[i + 1])), (2, 0)
        )

    if form == "Right-canonical":
        return ExplicitMPS(
            tensors, singular_values, tolerance=tolerance, chi_max=chi_max
        ).right_canonical()
    if form == "Left-canonical":
        return ExplicitMPS(
            tensors, singular_values, tolerance=tolerance, chi_max=chi_max
        ).left_canonical()
    if form == "Mixed-canonical":
        return ExplicitMPS(
            tensors, singular_values, tolerance=tolerance, chi_max=chi_max
        ).mixed_canonical(orth_centre=orth_centre)

    return ExplicitMPS(tensors, singular_values, tolerance=tolerance)


def create_simple_product_state(
    num_sites: np.int16,
    which: str = "0",
    phys_dim: np.int16 = 2,
    form: str = "Explicit",
) -> Union[CanonicalMPS, ExplicitMPS]:
    """
    Creates |0...0>/|1...1>/|+...+> as an MPS.
    """

    tensor = np.zeros((phys_dim,))
    if which == "0":
        tensor[0] = 1.0
    if which == "1":
        tensor[-1] = 1.0
    if which == "+":
        for i in range(phys_dim):
            tensor[i] = 1 / np.sqrt(phys_dim)

    tensors = [tensor.reshape((1, phys_dim, 1)) for _ in range(num_sites)]
    singular_values = [[1.0] for _ in range(num_sites + 1)]

    if form == "Right-canonical":
        return ExplicitMPS(tensors, singular_values).right_canonical()
    if form == "Left-canonical":
        return ExplicitMPS(tensors, singular_values).left_canonical()
    if form == "Mixed-canonical":
        raise ValueError("Mixed-canonical form is not defined for a product state.")

    return ExplicitMPS(tensors, singular_values)


def create_custom_product_state(
    string: str, phys_dim: np.int16 = 2, form: str = "Explicit"
) -> Union[CanonicalMPS, ExplicitMPS]:
    """
    Creates a custom product state defined by the `string` argument as an MPS.
    """

    num_sites = len(string)
    tensors = []

    for k in string:

        tensor = np.zeros((phys_dim,))
        if k == "0":
            tensor[0] = 1.0
        elif k == "1":
            tensor[-1] = 1.0
        elif k == "+":
            for i in range(phys_dim):
                tensor[i] = 1 / np.sqrt(phys_dim)
        else:
            raise ValueError(
                f"The string argument accepts only 0, 1 or + as elements, given {k}."
            )
        tensors.append(tensor)

    tensors = [tensor.reshape((1, phys_dim, 1)) for tensor in tensors]
    singular_values = [[1.0] for _ in range(num_sites + 1)]

    if form == "Right-canonical":
        return ExplicitMPS(tensors, singular_values).right_canonical()
    if form == "Left-canonical":
        return ExplicitMPS(tensors, singular_values).left_canonical()
    if form == "Mixed-canonical":
        raise ValueError("Mixed-canonical form is not defined for a product state.")

    return ExplicitMPS(tensors, singular_values)
