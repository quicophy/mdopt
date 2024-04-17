"""
Below, we define some functions we will be using for random circuit simulation.
"""

from typing import List
import numpy as np


def create_mpo(gate: np.ndarray, phys_dim: int = 2) -> List[np.ndarray]:
    """
    Creates a two-site MPO from a two-qubit gate by virtue of QR decomposition.

    Parameters
    ----------
    gate : np.ndarray
        The input gate.
    phys_dim: int
        The physical dimension of the system.

    Returns
    -------
        Two one-body gates representing the input gate.

    Raises
    ------
    ValueError
        If the gate is not a two-dimensional matrix.

    Notes
    -----
    Note that this operation does not make sense for a unitary gate.
    """

    if len(gate.shape) != 2:
        raise ValueError(
            f"The number of dimensions of the gate is {len(gate.shape)},"
            "but the number of dimensions expected is 2."
        )

    q, r = np.linalg.qr(gate)
    q_shape = q.shape
    r_shape = r.shape
    q = q.reshape((1, q_shape[1], phys_dim, phys_dim))
    r = r.reshape((r_shape[0], 1, phys_dim, phys_dim))

    return [q, r]


def create_mpo_from_layer(
    layer: List[np.ndarray], phys_dim: int = 2
) -> List[np.ndarray]:
    """
    Creates an MPO from a layer of two-site MPOs.

    Parameters
    ----------
    layer : List[np.ndarray]
        The input layer of gates.
    phys_dim: int
        The physical dimension of the system.

    Returns
    -------
        The resulting MPO.
    """

    mpo = []
    for _, gate in enumerate(layer):
        mpo += create_mpo(gate=gate, phys_dim=phys_dim)

    return mpo
