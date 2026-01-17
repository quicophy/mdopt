"""
This module contains the :class:`DephasingDMRG` and the :class:`EffectiveDensityOperator` classes.

This algorithm's main feature is that it restricts the target-state search to
the computational basis states domain.
In particular, we use it to find the main component of a Matrix Density Product Operator (MDPO),
i.e., a computational basis state contributing the largest amplitude.

In our notation, MDPO for ``n`` sites denotes the following object::

         |      |               |       |
         |      |               |       |
    ----(0*)---(1*)--- ... ---(n-2*)--(n-1*)---
    ----(0)----(1)---- ... ---(n-2)---(n-1)----
         |      |               |       |
         |      |               |       |

An MDPO is formed by an MPS and its complex-conjugated version.
The main idea is to find the main component of this object without
performing the kronecker product explicitly.
"""

from typing import Union, cast
import numpy as np
import scipy.sparse
from opt_einsum import contract
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.explicit import ExplicitMPS
from mdopt.utils.utils import split_two_site_tensor


class EffectiveDensityOperator(scipy.sparse.linalg.LinearOperator):
    """
    Class to store an effective two-site density operator.

    To take more advantage of the ``scipy.sparse.linalg`` module, we make a special class
    for local effective density operators extending the analogy from local effective operators.
    It allows us to compute eigenvectors more effeciently.

    The diagram displaying the contraction can be found in the supplementary notes.
    """

    # Single compiled einsum route kept for readability/reuse
    _EINSUM = "ustw, ailu, ifj, jhk, bef, cgh, lom, mpn, eso, gtp, dknw -> abcd"

    def __init__(
        self,
        left_environment: np.ndarray,
        mps_target_1: np.ndarray,
        mps_target_2: np.ndarray,
        right_environment: np.ndarray,
    ) -> None:
        if left_environment.ndim != 4:
            raise ValueError(
                "A valid left environment tensor must have 4 legs "
                f"while the one given has {left_environment.ndim}."
            )
        if mps_target_1.ndim != 3:
            raise ValueError(
                "A valid target MPS tensor must have 3 legs "
                f"while the one given has {mps_target_1.ndim}."
            )
        if mps_target_2.ndim != 3:
            raise ValueError(
                "A valid target MPS tensor must have 3 legs "
                f"while the one given has {mps_target_2.ndim}."
            )
        if right_environment.ndim != 4:
            raise ValueError(
                "A valid right environment tensor must have 4 legs "
                f"while the one given has {right_environment.ndim}."
            )

        self.left_environment = left_environment
        self.right_environment = right_environment
        self.mps_target_1 = mps_target_1
        self.mps_target_2 = mps_target_2

        chi_1 = left_environment.shape[3]
        chi_2 = right_environment.shape[3]
        d_1 = mps_target_1.shape[1]
        d_2 = mps_target_2.shape[1]

        self.x_shape = (chi_1, d_1, d_2, chi_2)
        self.shape = (chi_1 * d_1 * d_2 * chi_2, chi_1 * d_1 * d_2 * chi_2)
        self.dtype = mps_target_1.dtype
        super().__init__(shape=self.shape, dtype=self.dtype)

        # Precompute the 3-way copy tensor δ_{i,j,k} once (dtype-safe)
        self._copy = np.zeros((2, 2, 2), dtype=self.dtype)
        self._copy[0, 0, 0] = 1
        self._copy[1, 1, 1] = 1

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Performs matrix-vector multiplication.

        Computes ``effective_density_operator * |x> = |x'>``.
        This function is used by ``scipy.sparse.linalg.eigsh`` to diagonalise
        the effective density operator with the Lanczos method, withouth generating the full matrix.

        Parameters
        ----------
        x : np.ndarray
            The two-site tensor to be acted on by an effective density operator.
        """
        two_site_tensor = np.reshape(x, self.x_shape)
        if two_site_tensor.ndim != 4:
            raise ValueError(
                f"A valid two-site tensor must have 4 legs while the one given has {two_site_tensor.ndim}."
            )

        y = contract(
            self._EINSUM,
            two_site_tensor,
            self.left_environment,
            np.conjugate(self.mps_target_1),
            np.conjugate(self.mps_target_2),
            self._copy,
            self._copy,
            self.mps_target_1,
            self.mps_target_2,
            self._copy,
            self._copy,
            self.right_environment,
            optimize=[
                (0, 8),
                (0, 1),
                (0, 6),
                (0, 5),
                (1, 2),
                (2, 3),
                (3, 4),
                (2, 3),
                (1, 2),
                (0, 1),
            ],
            use_blas=True,
        )

        return np.reshape(y, self.shape[0])


class DephasingDMRG:
    """
    Class holding the Dephasing Density Matrix Renormalisation Group algorithm with two-site updates
    for a finite-size system with open-boundary conditions.

    Attributes
    ----------
    mps : Union[ExplicitMPS, CanonicalMPS]
        MPS serving as a current approximation of the target state.
    mps_target : Union[ExplicitMPS, CanonicalMPS]
        The target MPS in the right-canonical form.
        This MPS is used to construct the dephased MDPO.
    chi_max : int
        The highest bond dimension of an MPS allowed.
    mode : str
        The eigensolver mode. Available options:
            | ``LM`` : Largest (in magnitude) eigenvalues.
            | ``SM`` : Smallest (in magnitude) eigenvalues.
            | ``LA`` : Largest (algebraic) eigenvalues.
            | ``SA`` : Smallest (algebraic) eigenvalues.
    cut : float
        The lower boundary of the spectrum, i.e., all
        the singular values smaller than that will be discarded.
    silent : bool
        Whether to show/hide the progress bar.
    """

    def __init__(
        self,
        mps: Union[ExplicitMPS, CanonicalMPS],
        mps_target: Union[ExplicitMPS, CanonicalMPS],
        chi_max: int = int(1e4),
        cut: float = float(1e-17),
        mode: str = "SA",
        silent: bool = False,
        copy: bool = True,
    ) -> None:
        """
        Raises
        ------
        ValueError
            If the current MPS and the target MPS do not have the same lengths.
        """
        if len(mps) != len(mps_target):
            raise ValueError(
                f"The MPS has length {len(mps)}, the target MPS has length {len(mps_target)}, "
                "but the lengths should be equal."
            )

        self.mps = mps.copy() if copy else mps
        self.mps_target = mps_target.right_canonical()
        self.chi_max = chi_max
        self.cut = cut
        self.mode = mode
        self.silent = silent

        L = len(mps)
        self.left_environments = [
            np.zeros(shape=(1,), dtype=self.mps.tensors[0].dtype) for _ in range(L)
        ]
        self.right_environments = [
            np.zeros(shape=(1,), dtype=self.mps.tensors[0].dtype) for _ in range(L)
        ]

        # dtype-safe envs (complex if needed)
        dtype = self.mps.tensors[0].dtype
        start_bond_dim = self.mps_target.tensors[0].shape[0]
        chi = self.mps.tensors[0].shape[0]

        left_environment = np.zeros(
            (chi, start_bond_dim, start_bond_dim, chi), dtype=dtype
        )
        right_environment = np.zeros(
            (chi, start_bond_dim, start_bond_dim, chi), dtype=dtype
        )

        left_environment[:, 0, 0, :] = np.eye(chi, dtype=dtype)
        right_environment[:, start_bond_dim - 1, start_bond_dim - 1, :] = np.eye(
            chi, dtype=dtype
        )

        self.left_environments[0] = left_environment
        self.right_environments[-1] = right_environment

        # Build right environments (right-to-left)
        for i in reversed(range(1, L)):
            self.update_right_environment(i)

    @staticmethod
    def _snap_to_computational_basis(x: np.ndarray) -> np.ndarray:
        """
        Project a two-site tensor onto a single computational-basis configuration
        (one-hot in the flattened basis). This is essential in degenerate cases
        (e.g., maximally mixed / large maximum a posteriori degeneracy), where eigensolvers may
        return arbitrary superpositions inside the top eigenspace.

        The tie-break is still deterministic: argmax on |x_k| chooses the first maximal index.
        """
        x_flat = x.reshape(-1)
        idx = int(np.argmax(np.abs(x_flat)))
        x_snapped = np.zeros_like(x_flat)
        x_snapped[idx] = np.array(1, dtype=x_flat.dtype)
        return x_snapped.reshape(x.shape)

    def sweep(self) -> None:
        """One full Dephasing DMRG sweep (left→right, then right→left)."""
        for i in range(self.mps.num_sites - 1):
            self.update_bond(i)
        for i in reversed(range(self.mps.num_sites - 1)):
            self.update_bond(i)

    def update_bond(self, i: int) -> None:
        """Update the bond between sites i and i+1."""
        j = i + 1

        effective_density_operator = EffectiveDensityOperator(
            self.left_environments[i],
            self.mps_target.tensors[i],
            self.mps_target.tensors[j],
            self.right_environments[j],
        )

        if isinstance(self.mps, CanonicalMPS):
            self.mps = cast(CanonicalMPS, self.mps.move_orth_centre(i))
            initial_guess = self.mps.two_site_tensor_next(i).reshape(
                effective_density_operator.shape[0]
            )
        else:
            initial_guess = self.mps.two_site_right_iso(i).reshape(
                effective_density_operator.shape[0]
            )

        _, eigenvectors = eigsh(
            effective_density_operator,
            k=1,
            which=self.mode,
            return_eigenvectors=True,
            v0=initial_guess,
            tol=1e-8,
        )
        x = eigenvectors[:, 0].reshape(effective_density_operator.x_shape)

        # Enforce the search domain: computational-basis bitstrings only.
        # Without this, degenerate top eigenspaces lead to coherence/entanglement leakage.
        x = self._snap_to_computational_basis(x)

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

        if isinstance(self.mps, ExplicitMPS):
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
        """Compute right_environment right of site i-1 from right of site i."""

        right_environment = self.right_environments[i]

        if isinstance(self.mps, CanonicalMPS):
            self.mps = cast(CanonicalMPS, self.mps.move_orth_centre(i - 1))
            right_iso = self.mps.one_site_tensor(i)
        else:
            right_iso = self.mps.one_site_right_iso(i)

        right_environment = contract(
            "ijkl, omi, pmj, qnk, rnl -> opqr",
            right_environment,
            right_iso,
            np.conjugate(self.mps_target.tensors[i]),
            self.mps_target.tensors[i],
            np.conjugate(right_iso),
            optimize=[(0, 2), (0, 1), (0, 1), (0, 1)],
        )
        self.right_environments[i - 1] = right_environment

    def update_left_environment(self, i: int) -> None:
        """Compute left_environment left of site i+1 from left of site i."""

        left_environment = self.left_environments[i]

        if isinstance(self.mps, CanonicalMPS):
            self.mps = cast(CanonicalMPS, self.mps.move_orth_centre(i + 1))
            left_iso = self.mps.one_site_tensor(i)
        else:
            left_iso = self.mps.one_site_left_iso(i)

        left_environment = contract(
            "ijkl, imo, jmp, knq, lnr -> opqr",
            left_environment,
            left_iso,
            np.conjugate(self.mps_target.tensors[i]),
            self.mps_target.tensors[i],
            np.conjugate(left_iso),
            optimize=[(0, 2), (0, 1), (0, 1), (0, 1)],
        )
        self.left_environments[i + 1] = left_environment

    def run(self, num_iter: int = 1) -> None:
        """Run the algorithm for `num_iter` full sweeps."""
        for _ in tqdm(range(num_iter), disable=self.silent):
            self.sweep()
