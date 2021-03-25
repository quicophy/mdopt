"""
    This module contains some commonly-used tensors.
    Written by Alex Berezutskii inspired by TenPy in 2020-2021.
"""

import numpy as np

# The Pauli matrices together with the identity matrix

I = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex64)

X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)

Y = np.asarray([[0.0, 1.0j], [1.0j, 0.0]], dtype=np.complex64)

Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)
