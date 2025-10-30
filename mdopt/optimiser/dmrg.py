"""
This module contains the :class:`DMRG` and the :class:`EffectiveOperator` classes.
The class structure is inspired by TenPy.
"""

from typing import Union, List, cast
import numpy as np
import scipy.sparse
from opt_einsum import contract
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.explicit import ExplicitMPS
from mdopt.utils.utils import split_two_site_tensor


class EffectiveOperator(scipy.sparse.linalg.LinearOperator):
    r"""Class to store an effective two-site operator.

    In order to take more advantage of the ``scipy.sparse.linalg`` module,
    we make a special class for local effective operators.
    It allows us to compute eigenvectors more efficiently.

    Such effective operator is to be diagonalised in the
    ``update_bond`` method of the :class:`DMRG` class::

        ---uL                    uR---
        |        i          j        |
        |  vL    |          |    vR  |
        (L)----(mpo_l)----(mpo_r)----(R)
        |        |          |        |
        |        k          l        |
        ---dL                    dR---

    In our convention, the legs of left/right environments (tensors ``L``/``R`` in the cartoon)
    are ordered as follows: ``(uL/uR, vL/vR, dL/dR)`` which means "(up, virtual, down)".
    """

    _EINSUM = "ijkl, mni, nopj, oqrk, sql -> mprs"

    def __init__(
        self,
        left_environment: np.ndarray,
        mpo_tensor_left: np.ndarray,
        mpo_tensor_right: np.ndarray,
        right_environment: np.ndarray,
    ) -> None:
        if left_environment.ndim != 3:
            raise ValueError(
                "A valid left environment tensor must have 3 legs "
                f"while the one given has {left_environment.ndim}."
            )
        if mpo_tensor_left.ndim != 4:
            raise ValueError(
                "A valid mpo left tensor must have 4 legs "
                f"while the one given has {mpo_tensor_left.ndim}."
            )
        if mpo_tensor_right.ndim != 4:
            raise ValueError(
                "A valid mpo right tensor must have 4 legs "
                f"while the one given has {mpo_tensor_right.ndim}."
            )
        if right_environment.ndim != 3:
            raise ValueError(
                "A valid right environment tensor must have 3 legs "
                f"while the one given has {right_environment.ndim}."
            )

        self.left_environment = left_environment
        self.right_environment = right_environment
        self.mpo_tensor_left = mpo_tensor_left
        self.mpo_tensor_right = mpo_tensor_right

        chi_1 = left_environment.shape[2]
        chi_2 = right_environment.shape[2]
        d_1 = mpo_tensor_left.shape[3]
        d_2 = mpo_tensor_right.shape[3]

        self.x_shape = (chi_1, d_1, d_2, chi_2)
        self.shape = (chi_1 * d_1 * d_2 * chi_2, chi_1 * d_1 * d_2 * chi_2)
        self.dtype = mpo_tensor_left.dtype
        super().__init__(shape=self.shape, dtype=self.dtype)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Performs matrix-vector multiplication: (effective operator) @ vec(x)."""
        two_site_tensor = np.reshape(x, self.x_shape)

        if two_site_tensor.ndim != 4:
            raise ValueError(
                f"A valid two-site tensor must have 4 legs "
                f"while the one given has {two_site_tensor.ndim}."
            )

        y = contract(
            self._EINSUM,
            two_site_tensor,
            self.left_environment,
            self.mpo_tensor_left,
            self.mpo_tensor_right,
            self.right_environment,
            optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
            use_blas=True,
        )

        return np.reshape(y, self.shape[0])


class DMRG:
    """
    Class holding the Density Matrix Renormalisation Group algorithm with two-site updates
    for a finite-size system with open-boundary conditions.
    """

    def __init__(
        self,
        mps: Union[ExplicitMPS, CanonicalMPS],
        mpo: List[np.ndarray],
        chi_max: int = int(1e4),
        cut: float = float(1e-17),
        mode: str = "SA",
        silent: bool = False,
        copy: bool = True,
    ) -> None:
        if len(mps) != len(mpo):
            raise ValueError(
                f"The MPS has length {len(mps)}, the MPO has length {len(mpo)}, "
                "but the lengths should be equal."
            )
        for i, tensor in enumerate(mpo):
            if tensor.ndim != 4:
                raise ValueError(
                    f"A valid MPO tensor must have 4 legs while tensor {i} has {tensor.ndim}."
                )
        if mode not in ["SA", "LA", "SM", "LM"]:
            raise ValueError("Invalid eigensolver mode given.")

        self.mps = mps.copy() if copy else mps
        if isinstance(self.mps, CanonicalMPS):
            self.mps = self.mps.right_canonical()

        L = len(mps)
        dtype = self.mps.tensors[0].dtype

        self.left_environments = [np.zeros(shape=(1,), dtype=dtype)] * L
        self.right_environments = [np.zeros(shape=(1,), dtype=dtype)] * L
        self.mpo = mpo
        self.chi_max = chi_max
        self.cut = cut
        self.mode = mode
        self.silent = silent

        start_bond_dim = self.mpo[0].shape[0]
        chi = self.mps.tensors[0].shape[0]
        left_environment = np.zeros((chi, start_bond_dim, chi), dtype=dtype)
        right_environment = np.zeros((chi, start_bond_dim, chi), dtype=dtype)
        left_environment[:, 0, :] = np.eye(chi, dtype=dtype)
        right_environment[:, start_bond_dim - 1, :] = np.eye(chi, dtype=dtype)
        self.left_environments[0] = left_environment
        self.right_environments[-1] = right_environment

        # Build right environments from right to left
        for i in reversed(range(1, L)):
            self.update_right_environment(i)

    def sweep(self) -> None:
        """One full DMRG sweep (left→right, then right→left)."""
        for i in range(self.mps.num_sites - 1):
            self.update_bond(i)
        for i in reversed(range(self.mps.num_sites - 1)):
            self.update_bond(i)

    def update_bond(self, i: int) -> None:
        """Update the bond between sites ``i`` and ``i+1``."""
        j = i + 1

        effective_hamiltonian = EffectiveOperator(
            self.left_environments[i],
            self.mpo[i],
            self.mpo[j],
            self.right_environments[j],
        )

        if isinstance(self.mps, ExplicitMPS):
            initial_guess = self.mps.two_site_right_iso(i).reshape(
                effective_hamiltonian.shape[0]
            )
        else:  # CanonicalMPS
            self.mps = cast(CanonicalMPS, self.mps.move_orth_centre(i))
            initial_guess = self.mps.two_site_tensor_next(i).reshape(
                effective_hamiltonian.shape[0]
            )

        _, eigenvectors = eigsh(
            effective_hamiltonian,
            k=1,
            which=self.mode,
            return_eigenvectors=True,
            v0=initial_guess,
            tol=1e-8,
        )
        x = eigenvectors[:, 0].reshape(effective_hamiltonian.x_shape)

        left_iso_i, singular_values_j, right_iso_j, _ = split_two_site_tensor(
            x,
            chi_max=self.chi_max,
            cut=self.cut,
            renormalise=True,
            return_truncation_error=True,
        )

        s = np.asarray(singular_values_j, dtype=self.mps.tensors[i].dtype)

        if isinstance(self.mps, CanonicalMPS):
            # mps[i] = left_iso_i @ diag(s)  -> scale vR axis by s
            self.mps.tensors[i] = left_iso_i * s[None, None, :]
            self.mps.orth_centre = i
            self.mps.tensors[j] = right_iso_j

        else:  # ExplicitMPS
            # Left site: inv(diag(Λ_i)) @ left_iso_i  -> divide vL axis by Λ_i
            sL = np.asarray(self.mps.singular_values[i], dtype=left_iso_i.dtype)
            sL_safe = np.where(sL != 0.0, sL, 1.0)
            self.mps.tensors[i] = left_iso_i / sL_safe[:, None, None]

            # Right site: right_iso_j @ inv(diag(Λ_{j+1})) -> divide vR axis by Λ_{j+1}
            sR = np.asarray(self.mps.singular_values[j + 1], dtype=right_iso_j.dtype)
            sR_safe = np.where(sR != 0.0, sR, 1.0)
            self.mps.tensors[j] = right_iso_j / sR_safe[None, None, :]

            # Update middle singular values
            self.mps.singular_values[j] = singular_values_j

        self.update_left_environment(i)
        self.update_right_environment(j)

    def update_right_environment(self, i: int) -> None:
        """
        Compute the ``right_environment`` right of site ``i-1``
        from the ``right_environment`` right of site ``i``.
        """
        right_environment = self.right_environments[i]

        if isinstance(self.mps, ExplicitMPS):
            right_iso = self.mps.one_site_right_iso(i)
        else:  # CanonicalMPS
            self.mps = cast(CanonicalMPS, self.mps.move_orth_centre(i - 1))
            right_iso = self.mps.one_site_tensor(i)

        right_environment = contract(
            "ijk, lnjm, omp, knp -> ilo",
            right_iso,
            self.mpo[i],
            np.conjugate(right_iso),
            right_environment,
            optimize=[(0, 3), (0, 2), (0, 1)],
        )
        self.right_environments[i - 1] = right_environment

    def update_left_environment(self, i: int) -> None:
        """
        Compute the ``left_environment`` left of site ``i+1``
        from the ``left_environment`` left of site ``i``.
        """
        left_environment = self.left_environments[i]

        if isinstance(self.mps, ExplicitMPS):
            left_iso = self.mps.one_site_left_iso(i)
        else:  # CanonicalMPS
            self.mps = cast(CanonicalMPS, self.mps.move_orth_centre(i + 1))
            left_iso = self.mps.one_site_tensor(i)

        left_environment = contract(
            "ijk, lnjm, omp, ilo -> knp",
            left_iso,
            self.mpo[i],
            np.conjugate(left_iso),
            left_environment,
            optimize=[(0, 3), (0, 2), (0, 1)],
        )
        self.left_environments[i + 1] = left_environment

    def run(self, num_iter: int = 1) -> None:
        """Run the algorithm, i.e., run ``sweep`` for ``num_iter`` iterations."""
        for _ in tqdm(range(num_iter), disable=self.silent):
            self.sweep()
