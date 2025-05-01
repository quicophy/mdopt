"""
This module contains various MPS utilities.
"""

from typing import Union, Optional, cast
from functools import reduce
import numpy as np
from opt_einsum import contract

from mdopt.utils.utils import svd
from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.explicit import ExplicitMPS


def create_state_vector(num_sites: int, phys_dim: int = int(2)) -> np.ndarray:
    """
    Creates a random uniform complex-valued vector of norm 1.

    Parameters
    ----------
    num_sites : int
        Number of degrees of freedom.
    phys_dim : int
        Number of dimensions of each degree of freedom.

    Returns
    -------
    state_vector : np.ndarray
        The resulting state vector.
    """

    state_vector = np.random.uniform(
        size=(phys_dim**num_sites)
    ) + 1j * np.random.uniform(size=phys_dim**num_sites)
    state_vector /= np.linalg.norm(state_vector)

    return state_vector


def find_orth_centre(
    mps: CanonicalMPS, return_orth_flags: bool = False, tolerance: float = 1e-12
):
    """
    Returns a list of integers corresponding to positions of orthogonality centres of an MPS.

    Parameters
    ----------
    mps : CanonicalMPS
        The MPS to find the orthogonality centre(s) in.
    return_orth_flags : bool
        Whether to return if each tensor is a right or a left isometry.
    tolerance : float
        Numerical tolerance for checking the isometry property.

    Returns
    -------
    orth_centres : List[int]
        Indices of sites at which tensors are orthogonality centres.
    orth_flags_left : Optional[List[bool]]
        Boolean variables for each tensor corresponding to being a left isometry.
    orth_flags_right : Optional[List[bool]]
        Boolean variables for each tensor corresponding to being a right isometry.

    Raises
    ------
    ValueError
        If an :class:`ExplicitMPS` instance is passed as an input.
        They do not have orthogonality centres by definition.
    """

    if isinstance(mps, ExplicitMPS):
        raise ValueError(
            "Orthogonality centre is undefined for an Explicit MPS instance."
        )

    num_sites = mps.num_sites

    orth_flags_left = []
    orth_flags_right = []
    orth_centres = []

    for i, tensor in enumerate(mps.tensors):
        to_be_identity_left = np.asarray(
            contract("ijk, ijl -> kl", tensor, np.conjugate(tensor), optimize=[(0, 1)])
        )
        to_be_identity_right = np.asarray(
            contract("ijk, ljk -> il", tensor, np.conjugate(tensor), optimize=[(0, 1)])
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


def is_canonical(mps: CanonicalMPS, tolerance: float = 1e-12):
    """
    Checks if the MPS is in any of the canonical forms.

    Parameters
    ----------
    mps : CanonicalMPS
        The MPS to check the canonical form of.
    tolerance : float
        Numerical tolerance for checking the isometry property.

    Returns
    -------
    if_canonical : bool
        ``True`` if the MPS is in any of the canonical forms.

    Raises
    ------
    ValueError
        If an :class:`ExplicitMPS` instance is passed as an input.
        They do not have orthogonality centres by definition.
    """

    if isinstance(mps, ExplicitMPS):
        raise ValueError(
            "Orthogonality centre is undefined for an Explicit MPS instance."
        )

    flags_left = []
    flags_right = []
    for _, tensor in enumerate(mps.tensors):
        to_be_identity_left = np.asarray(
            contract("ijk, ijl -> kl", tensor, np.conjugate(tensor), optimize=[(0, 1)])
        )
        to_be_identity_right = np.asarray(
            contract("ijk, ljk -> il", tensor, np.conjugate(tensor), optimize=[(0, 1)])
        )

        identity_left = np.identity(to_be_identity_left.shape[0], dtype=float)
        identity_right = np.identity(to_be_identity_right.shape[0], dtype=float)

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
    if_canonical = len(orth_centres) == 1

    return if_canonical


def inner_product(
    mps_1: Union[ExplicitMPS, CanonicalMPS], mps_2: Union[ExplicitMPS, CanonicalMPS]
) -> Union[float, np.complex128]:
    """
    Returns an inner product between 2 Matrix Product States.

    Parameters
    ----------
    mps_1 : Union[ExplicitMPS, CanonicalMPS]
        The first MPS in the inner product.
    mps_1 : Union[ExplicitMPS, CanonicalMPS]
        The second MPS in the inner product.

    Returns
    -------
    product : Union[float, np.complex128]
        The value of the inner product.

    Raises
    ------
    ValueError
        If the Matrix Product States are of different length.
    """

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

    product = reduce(lambda a, b: np.tensordot(a, b, (-1, 0)), tensors)[0][0]

    if np.isclose(np.imag(product), 0):
        return float(np.real(product))

    return np.complex128(product)


def mps_from_dense(
    state_vector: np.ndarray,
    phys_dim: int = int(2),
    chi_max: int = int(1e4),
    tolerance: float = float(1e-12),
    form: str = "Explicit",
    orth_centre: Optional[int] = None,
) -> Union[ExplicitMPS, CanonicalMPS]:
    """
    Builds an MPS from a dense (state-vector) from.

    Parameters
    ----------
    state_vector : np.ndarray
        The initial state vector.
    phys_dim : int
        Dimensionality of the local Hilbert space, i.e.,
        the dimension of each physical leg of the MPS.
    chi_max : int
        Maximum number of singular values to keep.
    tolerance : float
        Absolute tolerance of the normalisation of the singular value spectrum at each bond.
    form : str
        The form of the MPS. Available options:
            | ``Explicit`` : The :class:`ExplicitMPS` form (by default).
            | ``Right-canonical`` : The :class:`CanonicalMPS` right-canonical form.
            | ``Left-canonical`` : The :class:`CanonicalMPS` left-canonical form.
            | ``Mixed-canonical`` : The :class:`CanonicalMPS` mixed-canonical form.
    orth_centre : Optional[int]
        The orthogonal centre position for the mixed-canonical form.

    Returns
    -------
    mps : Union[ExplicitMPS, CanonicalMPS]
        The resulting MPS.

    Raises
    ------
    ValueError
        If the vector's length does not correspond to the physical dimension.
    """

    if form not in ["Explicit", "Right-canonical", "Left-canonical", "Mixed-canonical"]:
        raise ValueError(
            "Wrong value of the form option. "
            "Available options: Explicit, Right-canonical, Left-canonical, Mixed-canonical"
        )

    state_vector = np.copy(state_vector)

    if state_vector.flatten().shape[0] % phys_dim != 0:
        raise ValueError(
            "The dimension of the flattened state vector is incorrect "
            "(does not correspond to the product of local dimensions)."
        )

    tensors = []
    singular_values = []

    state_vector = state_vector.reshape((-1, phys_dim))

    state_vector, singular_values_local, v_r, _ = svd(
        state_vector, chi_max=chi_max, renormalise=False
    )

    tensors.append(np.expand_dims(v_r, -1))
    singular_values.append(singular_values_local)

    while state_vector.shape[0] >= phys_dim:
        state_vector = np.matmul(state_vector, np.diag(singular_values_local))

        bond_dim = state_vector.shape[-1]
        state_vector = state_vector.reshape((-1, phys_dim * bond_dim))
        state_vector, singular_values_local, v_r, _ = svd(
            state_vector, chi_max=chi_max, renormalise=False
        )
        v_r = v_r.reshape((-1, phys_dim, bond_dim))

        tensors.insert(0, v_r)
        singular_values.insert(0, singular_values_local)

    singular_values.append([1.0])

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
        orth_centre = cast(int, orth_centre)
        return ExplicitMPS(
            tensors, singular_values, tolerance=tolerance, chi_max=chi_max
        ).mixed_canonical(orth_centre=orth_centre)

    return ExplicitMPS(tensors, singular_values, tolerance=tolerance)


def create_simple_product_state(
    num_sites: int,
    which: str = "0",
    phys_dim: int = 2,
    form: str = "Right-canonical",
    tolerance: float = float(1e-12),
) -> Union[ExplicitMPS, CanonicalMPS]:
    r"""
    Creates a simple product-state MPS.

    Parameters
    ----------
    num_sites : int
        The number of sites.
    which : str
        The form of the MPS, for explanation see the notes. Available options:
            | ``0`` : The :math:`|0...0>` state.
            | ``1`` : The :math:`|1...1>` state.
            | ``+`` : The :math:`|+...+>` state.
    phys_dim : int
        Dimensionality of the local Hilbert space, i.e.,
        the dimension of each physical leg of the MPS.
    form : str
        The form of the MPS. Available options:
            | ``Explicit`` : The :class:`ExplicitMPS` form (by default).
            | ``Right-canonical`` : The :class:`CanonicalMPS` right-canonical form.
            | ``Left-canonical`` : The :class:`CanonicalMPS` left-canonical form.
    tolerance : float
        For the Explicit form:
        absolute tolerance of the normalisation of the singular value spectrum at each bond.
        For the Canonical form:
        numerical tolerance to zero out the singular values in Singular Value Decomposition.

    Returns
    -------
    mps : Union[ExplicitMPS, CanonicalMPS]
        The resulting MPS.

    Raises
    ------
    ValueError
        If the chosen form is mixed-canonical. (This form is not available for product states.)

    Notes
    -----
    Produces a Matrix Product State consisting of tensors with bond dimenstions equal to 1.
    The tensors are defined as follows:
        | :math:`| 0 \rangle = \underbrace{(1, 0, ..., 0, 0)}_{\text{phys_dim}}`,
        | :math:`| 1 \rangle = \underbrace{(0, 0, ..., 0, 1)}_{\text{phys_dim}}`,
        | :math:`| + \rangle = \underbrace{(\frac{1}{\sqrt{\text{phys_dim}}}, ..., \frac{1}{\sqrt{\text{phys_dim}}})}_{\text{phys_dim}}`.

    The returned state is normalised.

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
        mps = ExplicitMPS(tensors, singular_values).right_canonical()
        mps.orth_centre = None
        return mps
    if form == "Left-canonical":
        mps = ExplicitMPS(tensors, singular_values).left_canonical()
        mps.orth_centre = None
        return mps
    if form == "Mixed-canonical":
        raise ValueError("Mixed-canonical form is not defined for a product state.")

    return ExplicitMPS(tensors, singular_values, tolerance=tolerance)


def create_custom_product_state(
    string: str,
    phys_dim: int = 2,
    form: str = "Right-canonical",
    tolerance: float = float(1e-12),
) -> Union[ExplicitMPS, CanonicalMPS]:
    r"""
    Creates a custom product-state MPS defined by the ``string`` argument.

    Parameters
    ----------
    string : str
        The string defining the product-state MPS. Available characters: ``0``, ``1``, ``+``.
    phys_dim : int
        Dimensionality of the local Hilbert space, i.e., the dimension of each physical leg of the MPS.
    form : str
        The form of the MPS. Available options:
        | ``Explicit`` : The :class:`ExplicitMPS` form (by default).
        | ``Right-canonical`` : The :class:`CanonicalMPS` right-canonical form.
        | ``Left-canonical`` : The :class:`CanonicalMPS` left-canonical form.
    tolerance : float
        For the Explicit form:
        Absolute tolerance of the normalization of the singular value spectrum at each bond.
        For the Canonical form:
        Numerical tolerance to zero out the singular values in Singular Value Decomposition.

    Returns
    -------
    mps : Union[ExplicitMPS, CanonicalMPS]
        The resulting MPS.

    Raises
    ------
    ValueError
        A symbol inside the ``string`` argument does not belong to the available set
        or the chosen form is mixed-canonical. (This form is not available for product states.)

    Notes
    -----
    Produces a Matrix Product State consisting of tensors with bond dimensions equal to 1.
    The tensors are defined as follows:
        | :math:`|0\rangle = \underbrace{(1, 0, ..., 0, 0)}_{\text{phys_dim}}`,
        | :math:`|1\rangle = \underbrace{(0, 0, ..., 0, 1)}_{\text{phys_dim}}`,
        | :math:`|+\rangle = \underbrace{(\frac{1}{\sqrt{\text{phys_dim}}}, ..., \frac{1}{\sqrt{\text{phys_dim}}})}_{\text{phys_dim}}`.

    The state is renormalised at the end.

    """

    num_sites = len(string)
    tensors = []

    for symbol in string:
        tensor = np.zeros((phys_dim,))
        if symbol == "0":
            tensor[0] = 1.0
        elif symbol == "1":
            tensor[-1] = 1.0
        elif symbol == "+":
            for i in range(phys_dim):
                tensor[i] = 1 / np.sqrt(phys_dim)
        else:
            raise ValueError(
                f"The string argument accepts only 0, 1 or + as elements, given {symbol}."
            )
        tensors.append(tensor)

    tensors = [tensor.reshape((1, phys_dim, 1)) for tensor in tensors]
    singular_values = [[1.0] for _ in range(num_sites + 1)]

    if form == "Right-canonical":
        mps = ExplicitMPS(
            tensors, singular_values, tolerance=tolerance
        ).right_canonical()
        mps.orth_centre = None
        return mps
    if form == "Left-canonical":
        mps = ExplicitMPS(
            tensors, singular_values, tolerance=tolerance
        ).left_canonical()
        mps.orth_centre = None
        return mps
    if form == "Mixed-canonical":
        raise ValueError("Mixed-canonical form is not defined for a product state.")

    return ExplicitMPS(tensors, singular_values, tolerance=tolerance)
