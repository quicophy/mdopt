"""
This module contains the :class:`DephasingDMRG` and the :class:`EffectiveDensityOperator` classes.

This algorithm's main feature is that it restricts the target-state search to
the computational basis states domain.
In particular, we use it to find the main component of a Matrix Product Density Operator (MPDO),
i.e., a computational basis state contributing the largest amplitude.

In our notation, MPDO for ``n`` sites denotes the following object::

         |      |               |       |
         |      |               |       |
    ----(0*)---(1*)--- ... ---(n-2*)--(n-1*)---
    ----(0)----(1)---- ... ---(n-2)---(n-1)----
         |      |               |       |
         |      |               |       |

An MPDO is formed by an MPS and its complex-conjugated version.
The main idea is to find the main component of this object without
performing the kronecker product explicitly.
"""

from typing import Union, cast
import os
import pathlib
import pickle

import numpy as np
import scipy.sparse
from opt_einsum import contract
from scipy.sparse.linalg import ArpackError, eigsh
from tqdm import tqdm

from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.explicit import ExplicitMPS
from mdopt.utils.utils import split_two_site_tensor


def _dump_arpack_failure(
    operator, start_vector, left_env, right_env, target_i, target_j, bond
):
    """Write the failing eigensolve to disk when MDOPT_ARPACK_DUMP is set.

    A no-op otherwise, so production runs are unaffected. Records the operator's
    action on several probes as well as its inputs, because the question is
    exactly what ARPACK saw that made it refuse to start.
    """
    destination = os.environ.get("MDOPT_ARPACK_DUMP")
    if not destination:
        return
    path = pathlib.Path(destination)
    if path.exists():  # keep the first failure, not the last
        return

    generator = np.random.default_rng(0)
    dimension = operator.shape[0]
    probes = {"start": start_vector}
    for index in range(3):
        probes[f"random{index}"] = generator.normal(size=dimension).astype(
            start_vector.dtype
        )
    images = {}
    for name, probe in probes.items():
        image = operator.matvec(probe)
        images[name] = {
            "norm": float(np.linalg.norm(image)),
            "max_abs": float(np.max(np.abs(image))) if image.size else 0.0,
            "all_finite": bool(np.all(np.isfinite(image))),
            "nan_count": int(np.sum(np.isnan(image))),
        }

    payload = {
        "bond": bond,
        "dimension": dimension,
        "dtype": str(start_vector.dtype),
        "start_vector": {
            "norm": float(np.linalg.norm(start_vector)),
            "max_abs": float(np.max(np.abs(start_vector))),
            "all_finite": bool(np.all(np.isfinite(start_vector))),
        },
        "images": images,
        "left_env": left_env,
        "right_env": right_env,
        "target_i": target_i,
        "target_j": target_j,
        "start_raw": start_vector,
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _nonzero_start_vector(guess: np.ndarray, dimension: int) -> np.ndarray:
    """Return a starting vector ARPACK will accept.

    ``eigsh`` raises "ARPACK error -9: Starting vector is zero" when ``v0`` is
    all zeros, taking the whole run down. The two-site tensor can underflow to
    zero on a hard instance -- that is what ended a 4.6h full-scale
    classical_ldpc run at chi_max=128, so this is not confined to the tiny bond
    dimensions where it was first seen. Any unit vector is a valid place for the
    iteration to start, so fall back to a uniform one rather than failing.
    """
    norm = float(np.linalg.norm(guess))
    if np.isfinite(norm) and norm > 0.0:
        return guess
    return np.full(dimension, 1.0 / np.sqrt(dimension), dtype=guess.dtype)


def _restart_from_operator_range(
    operator: scipy.sparse.linalg.LinearOperator,
    start_vector: np.ndarray,
    mode: str,
    attempts: int = 8,
):
    """Retry the eigensolve from a vector inside the operator's range.

    ARPACK reports "error -9: Starting vector is zero" when ``A @ v0`` is exactly
    zero, which happens when v0 lies in the operator's nullspace -- the operator
    itself can be perfectly healthy. That is the case this recovers: ``A @ x`` is
    in the range by construction, so it cannot be annihilated unless ``A`` is
    zero everywhere.

    Returns the eigenvectors, or None if no usable restart exists (the operator
    annihilates or corrupts every probe). Deciding which of those two it is is
    left to the caller.
    """
    generator = np.random.default_rng(0)
    dimension = operator.shape[0]
    for _ in range(attempts):
        probe = generator.normal(size=dimension).astype(start_vector.dtype)
        candidate = operator.matvec(probe)
        norm = float(np.linalg.norm(candidate))
        if not np.isfinite(norm) or norm == 0.0:
            continue
        try:
            _, eigenvectors = eigsh(
                operator,
                k=1,
                which=mode,
                return_eigenvectors=True,
                v0=candidate / norm,
                tol=1e-8,
            )
            return eigenvectors
        except ArpackError:
            continue
    return None


def _operator_is_numerically_zero(
    operator: scipy.sparse.linalg.LinearOperator, first_probe: np.ndarray
) -> bool:
    """Whether ``operator`` annihilates every vector it is shown.

    Used only after ARPACK has already failed, to tell a genuinely zero operator
    from one that merely has the starting vector in its nullspace. Random probes
    make the second case vanishingly unlikely to be misread.

    A non-finite image counts as "not zero": NaN or Inf means the state is
    corrupted, which the caller must not quietly skip past.
    """
    generator = np.random.default_rng(0)
    dimension = operator.shape[0]
    probes = [first_probe] + [
        generator.normal(size=dimension).astype(first_probe.dtype) for _ in range(3)
    ]
    for probe in probes:
        image = operator.matvec(probe)
        if not np.all(np.isfinite(image)):
            # NaN or Inf is corrupted state, not a zero operator. Reporting it as
            # "zero" would skip the bond and bury the corruption; say no so the
            # original ArpackError propagates and the run stops loudly.
            return False
        if np.linalg.norm(image) > 0.0:
            return False
    return True


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
        This MPS is used to construct the dephased MPDO.
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

        start_vector = _nonzero_start_vector(
            initial_guess, effective_density_operator.shape[0]
        )

        try:
            _, eigenvectors = eigsh(
                effective_density_operator,
                k=1,
                which=self.mode,
                return_eigenvectors=True,
                v0=start_vector,
                tol=1e-8,
            )
        except ArpackError:
            # Opt-in forensics: set MDOPT_ARPACK_DUMP to a path and the failing
            # operator is written there before any interpretation of it. Four
            # hypotheses about this error have been wrong, so capture the object
            # rather than reason about it from a distance.
            # "ARPACK error -9: Starting vector is zero" has two causes and names
            # only one. v0 is guarded above, which leaves an operator that
            # annihilates whatever it is given: the effective density operator is
            # built from the target MPS tensors, so once those underflow it is
            # numerically zero and the Krylov space collapses however the
            # iteration starts. That ended full-scale classical_ldpc runs twice.
            #
            # The usual cause is v0 sitting exactly in the operator's
            # nullspace -- _snap_to_computational_basis pins the state to one
            # basis configuration, and if that configuration has no amplitude in
            # the target at this bond, the effective operator annihilates it
            # exactly while remaining perfectly healthy elsewhere. Restarting
            # from inside the operator's range recovers those.
            eigenvectors = _restart_from_operator_range(
                effective_density_operator, start_vector, self.mode
            )
            if eigenvectors is None:
                # No restart worked. Either the operator is genuinely zero, in
                # which case there is nothing to optimise here, or it is
                # corrupted -- and corruption must not be skipped silently.
                if not _operator_is_numerically_zero(
                    effective_density_operator, start_vector
                ):
                    _dump_arpack_failure(
                        effective_density_operator,
                        start_vector,
                        self.left_environments[i],
                        self.right_environments[j],
                        self.mps_target.tensors[i],
                        self.mps_target.tensors[j],
                        i,
                    )
                    raise
                # The environments must still advance: the sweep reads
                # left_environments[i + 1] on the next bond, and skipping the
                # updates would leave it at the placeholder built in __init__.
                self.update_left_environment(i)
                self.update_right_environment(j)
                return
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
