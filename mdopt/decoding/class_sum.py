"""Class sums of a detector error model on an active set of mechanisms.

:func:`~mdopt.decoding.dem.decode_dem` contracts every mechanism of the model.
For quantum LDPC memory circuits that chain is thousands of sites long and its
detector constraints have long spans, so a small bond dimension truncates away
the answer. :func:`active_set_class_masses` contracts a much smaller problem
instead: the caller names an *active set* of mechanisms, every other mechanism
is pinned to "did not fire", and the function returns the probability of each
logical class inside that set.

1. **Kernel picture.** Gauss-Jordan elimination of the detector matrix
   restricted to the active set picks one pivot mechanism per independent
   detector. Every assignment of the remaining *free* mechanisms is then a
   valid solution: a pivot fires if and only if its base bit and the parity
   of the free mechanisms in its reduced row disagree. No hard syndrome
   constraint is left, so truncation can only lose probability mass of valid
   solutions.
2. **Class sum.** An MPS over one register site per observable followed by
   one site per free mechanism holds the square roots of the probabilities.
   Each pivot is a soft parity factor, each observable a hard parity with its
   register site. The squared norm of each register value is the total
   probability of that logical class inside the active set.

The contraction (:class:`ParityChain`) applies each parity factor exactly (the
bonds it crosses double) and then compresses with a canonical SVD sweep, and
the bond cut never splits a degenerate multiplet of singular values. Class
masses are squared norms, so they are non-negative by construction.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import svd as scipy_svd
from scipy.sparse import csr_matrix

from mdopt.decoding.dem import DemProblem, _validated_syndrome


#: Singular values below this fraction of the largest one count as zero.
ZERO_CUTOFF = 1e-13


def _svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD that falls back to the slower, more robust LAPACK driver."""
    try:
        return np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return scipy_svd(matrix, full_matrices=False, lapack_driver="gesvd")


def multiplet_safe_rank(
    singular_values: np.ndarray, chi_max: int, tolerance: float = 1e-7
) -> int:
    """Number of singular values to keep without splitting a multiplet.

    Parity constraints produce exactly degenerate singular values. Cutting
    through such a multiplet keeps an arbitrary rotation of it and breaks the
    symmetry between logical classes, so the cut is moved to the nearest
    multiplet boundary: up if that costs at most ``max(4, chi_max // 4)``
    extra values (or if the multiplet starts at the top of the spectrum),
    down otherwise.

    The multiplet is the one the requested cut falls in: the values above
    the cut within ``tolerance`` of the first value below it, and the values
    below the cut within ``tolerance`` of the last value above it. Two values
    whose relative difference is exactly ``tolerance`` count as different.
    The window is anchored at the cut and is not extended from neighbor to
    neighbor, so a long run of values that drift apart by less than
    ``tolerance`` per step is one multiplet only as far as it stays within
    ``tolerance`` of the cut. Exact multiplets, which differ by rounding
    alone, are always kept or dropped whole. A multiplet at the top of the
    spectrum is kept whole whatever its size, since dropping it would leave
    nothing: the bond can then exceed ``chi_max`` by more than the allowance.

    Parameters
    ----------
    singular_values : np.ndarray
        Singular values in descending order.
    chi_max : int
        The requested bond dimension.
    tolerance : float
        Relative difference below which two singular values count as equal.

    Returns
    -------
    int
        The number of singular values to keep.
    """
    if singular_values.size == 0 or singular_values[0] <= 0:
        return 0
    nonzero = int(np.sum(singular_values > singular_values[0] * ZERO_CUTOFF))
    keep = min(chi_max, nonzero)
    if not 0 < keep < nonzero:
        return keep
    if singular_values[keep] <= singular_values[keep - 1] * (1 - tolerance):
        return keep
    low = keep
    while (
        low > 0 and singular_values[low - 1] * (1 - tolerance) < singular_values[keep]
    ):
        low -= 1
    high = keep
    while high < nonzero and singular_values[high] > singular_values[keep - 1] * (
        1 - tolerance
    ):
        high += 1
    if high <= chi_max + max(4, chi_max // 4) or low == 0:
        return high
    return low


class ParityChain:
    """A real MPS of bits that is multiplied by functions of parities.

    The chain is kept in mixed canonical form and normalized; the logarithm
    of the accumulated norm is tracked separately.

    Parameters
    ----------
    vectors : Sequence[np.ndarray]
        One length-2 vector per site: the initial product state.

    Attributes
    ----------
    tensors : List[np.ndarray]
        Site tensors with index order ``(left bond, bit, right bond)``.
    center : int
        The orthogonality center.
    log_norm : float
        Logarithm of the norm that has been divided out so far.
    lost : float
        Sum over all truncations of the discarded fraction of the squared
        norm.
    """

    def __init__(self, vectors: Sequence[np.ndarray]):
        if len(vectors) == 0:
            raise ValueError("The chain needs at least one site.")
        self.tensors: List[np.ndarray] = []
        self.center = 0
        self.log_norm = 0.0
        self.lost = 0.0
        # Every site is normalized by itself and the logarithms are summed:
        # the norm of the whole product can be far outside float64 while
        # its logarithm is an ordinary number. Normalized rank-one tensors
        # are canonical for any center, so no sweep is needed.
        for vector in vectors:
            tensor = np.asarray(vector, dtype=float).reshape(1, 2, 1)
            peak = float(np.abs(tensor).max())
            if peak == 0 or not np.isfinite(peak):
                raise ArithmeticError("The product state has a zero or infinite site.")
            # The norm itself is never formed: for entries near the top of
            # float64 it would be infinite while its logarithm is not.
            scaled = tensor / peak
            scaled_norm = float(np.linalg.norm(scaled))
            self.tensors.append(scaled / scaled_norm)
            self.log_norm += float(np.log(peak) + np.log(scaled_norm))

    def __len__(self) -> int:
        return len(self.tensors)

    def _normalize(self) -> None:
        # Scaled by the largest entry first: amplitudes of 1e-200 are not
        # zero, but their squares are.
        center = self.tensors[self.center]
        peak = float(np.abs(center).max()) if center.size else 0.0
        if peak == 0 or not np.isfinite(peak):
            raise ArithmeticError("The constraint annihilated the state.")
        scaled = center / peak
        scaled_norm = float(np.linalg.norm(scaled))
        self.tensors[self.center] = scaled / scaled_norm
        self.log_norm += float(np.log(peak) + np.log(scaled_norm))

    def _orthogonalize_left(self, site: int) -> None:
        """Make ``site`` left-canonical and push the remainder to its right."""
        left, phys, right = self.tensors[site].shape
        isometry, rest = np.linalg.qr(self.tensors[site].reshape(left * phys, right))
        self.tensors[site] = isometry.reshape(left, phys, -1)
        self.tensors[site + 1] = np.tensordot(rest, self.tensors[site + 1], axes=(1, 0))

    def _orthogonalize_right(self, site: int) -> None:
        """Make ``site`` right-canonical and push the remainder to its left."""
        left, phys, right = self.tensors[site].shape
        isometry, rest = np.linalg.qr(self.tensors[site].reshape(left, phys * right).T)
        self.tensors[site] = isometry.T.reshape(-1, phys, right)
        self.tensors[site - 1] = np.tensordot(
            self.tensors[site - 1], rest.T, axes=(2, 0)
        )

    def move_center(self, site: int) -> None:
        """Move the orthogonality center to ``site`` without truncation."""
        while self.center < site:
            self._orthogonalize_left(self.center)
            self.center += 1
        while self.center > site:
            self._orthogonalize_right(self.center)
            self.center -= 1

    def project_bit(self, site: int, value: int) -> None:
        """Project the bit at ``site`` onto ``value`` and renormalize."""
        self.move_center(site)
        self.tensors[site][:, 1 - value, :] = 0.0
        self._normalize()

    def apply_parity(
        self,
        sites: Sequence[int],
        chi_max: int,
        weights: Union[Sequence[float], np.ndarray] = (1.0, 0.0),
    ) -> None:
        """Multiply the state by ``weights[parity of the bits at sites]``.

        The factor is applied exactly, which doubles the bonds between the
        first and the last site involved, and the chain is then compressed
        back to ``chi_max`` by one canonical SVD sweep. The default weights
        project onto even parity.

        Parameters
        ----------
        sites : Sequence[int]
            The sites whose parity enters the factor, each named once.
        chi_max : int
            The bond dimension to compress to.
        weights : Union[Sequence[float], np.ndarray]
            The factor for even and for odd parity.

        Raises
        ------
        ValueError
            If no site is given, a site is named twice or lies outside the
            chain, ``chi_max`` is not positive, or ``weights`` does not hold
            two numbers.
        """
        listed = [int(site) for site in sites]
        support = set(listed)
        if not support:
            raise ValueError("A parity factor needs at least one site.")
        if len(support) != len(listed):
            # Over GF(2) a site named twice cancels; silently keeping it once
            # would apply a different factor from the one asked for.
            raise ValueError("A parity factor names a site more than once.")
        if chi_max < 1:
            raise ValueError("The bond dimension must be positive.")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (2,):
            raise ValueError("A parity factor has two weights: even and odd.")
        first, last = min(support), max(support)
        if first < 0 or last >= len(self.tensors):
            raise ValueError("A parity factor acts on a site outside the chain.")
        self.move_center(first)
        if first == last:
            self.tensors[first] = self.tensors[first] * weights[None, :, None]
            self._normalize()
            return
        for site in range(first, last + 1):
            self._thread_parity(
                site, site in support, site == first, site == last, weights
            )
        for site in range(first, last):
            self._orthogonalize_left(site)
        self.center = last
        self._normalize()
        for site in range(last, first, -1):
            self._truncate_bond(site, chi_max)
        self.center = first
        self._normalize()

    def _thread_parity(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        site: int,
        counted: bool,
        is_first: bool,
        is_last: bool,
        weights: Union[Sequence[float], np.ndarray],
    ) -> None:
        """Thread the running parity through ``site`` as an extra bond index.

        The parity enters from the left (trivially at the first site), picks
        up the bit of the site if it is ``counted``, and leaves to the right;
        at the last site it is contracted with ``weights`` instead.
        """
        tensor = self.tensors[site]
        left, _, right = tensor.shape
        parity_in = 1 if is_first else 2
        parity_out = 1 if is_last else 2
        new = np.zeros((left, parity_in, 2, right, parity_out))
        for incoming in range(parity_in):
            for bit in range(2):
                outgoing = incoming ^ (bit if counted else 0)
                if is_last:
                    new[:, incoming, bit, :, 0] = tensor[:, bit, :] * weights[outgoing]
                else:
                    new[:, incoming, bit, :, outgoing] = tensor[:, bit, :]
        self.tensors[site] = np.reshape(new, (left * parity_in, 2, right * parity_out))

    def _truncate_bond(self, site: int, chi_max: int) -> None:
        """Truncate the bond to the left of ``site``; the center moves left."""
        left, phys, right = self.tensors[site].shape
        left_vectors, singular_values, right_vectors = _svd(
            self.tensors[site].reshape(left, phys * right)
        )
        keep = multiplet_safe_rank(singular_values, chi_max)
        if keep == 0:
            raise ArithmeticError("The state vanished during the contraction.")
        if keep < singular_values.size:
            # Relative to the largest value, so that the fraction is right
            # for a spectrum whose squares would underflow.
            scaled = singular_values / singular_values[0]
            self.lost += float(np.sum(scaled[keep:] ** 2) / np.sum(scaled**2))
        self.tensors[site] = right_vectors[:keep].reshape(keep, phys, right)
        self.tensors[site - 1] = np.tensordot(
            self.tensors[site - 1],
            left_vectors[:, :keep] * singular_values[:keep],
            axes=(2, 0),
        )

    def max_bond(self) -> int:
        """The largest bond dimension of the chain."""
        return max(tensor.shape[2] for tensor in self.tensors)


def problem_matrices(problem: DemProblem) -> Tuple[csr_matrix, csr_matrix]:
    """Sparse detector and observable matrices of a detector error model.

    Returns
    -------
    (check_matrix, observable_matrix)
        ``check_matrix[d, i]`` is 1 if mechanism ``i`` flips detector ``d``
        and ``observable_matrix[j, i]`` is 1 if it flips observable ``j``.
    """
    num_mech = problem.num_mechanisms

    def build(rows: List[List[int]]) -> csr_matrix:
        row_index = [r for r, row in enumerate(rows) for _ in row]
        column_index = [i for row in rows for i in row]
        data = np.ones(len(row_index), dtype=np.uint8)
        matrix = csr_matrix(
            (data, (row_index, column_index)), shape=(len(rows), num_mech)
        )
        # The constructor sums the entries of a mechanism that one row
        # lists more than once; over GF(2) every pair of them cancels.
        matrix.data %= 2
        matrix.eliminate_zeros()
        return matrix

    return build(problem.detector_rows), build(problem.observable_rows)


def _gauss_jordan(
    check_matrix: np.ndarray, syndrome: np.ndarray, columns: Sequence[int]
) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    """Gauss-Jordan elimination over GF(2) restricted to ``columns``.

    Pivots are chosen in the order of ``columns``. Returns the pivot columns,
    the row of each pivot, the reduced matrix augmented by the reduced
    syndrome as its last column, and a mask of the rows that hold a pivot.
    """
    work = np.concatenate([check_matrix, syndrome[:, None]], axis=1).astype(np.uint8)
    used = np.zeros(check_matrix.shape[0], dtype=bool)
    pivots: List[int] = []
    pivot_rows: List[int] = []
    for column in columns:
        candidates = np.flatnonzero(work[:, column] & ~used)
        if candidates.size == 0:
            continue
        row = int(candidates[0])
        used[row] = True
        others = np.flatnonzero(work[:, column])
        others = others[others != row]
        work[others] ^= work[row]
        pivots.append(int(column))
        pivot_rows.append(row)
    return pivots, pivot_rows, work, used


@dataclass
class _KernelPicture:
    """The solutions of a syndrome inside an active set, in free variables.

    Attributes
    ----------
    pivots : List[int]
        Pivot mechanisms, one per independent detector of the active set.
    base : np.ndarray
        For each pivot, whether it fires when every free mechanism is off.
    free : List[int]
        The free mechanisms, in chain order.
    supports : List[List[int]]
        For each pivot, the positions (in ``free``) of the free mechanisms
        that toggle it.
    observable_flips : np.ndarray
        ``observable_flips[j, k]`` is 1 if switching free mechanism ``k`` on
        (together with the pivots it toggles) flips observable ``j``.
    base_class : int
        Observable pattern of the base solution.
    """

    pivots: List[int]
    base: np.ndarray
    free: List[int]
    supports: List[List[int]]
    observable_flips: np.ndarray
    base_class: int


def _kernel_picture(
    check_matrix: np.ndarray,
    observable_matrix: np.ndarray,
    syndrome: np.ndarray,
    columns: List[int],
) -> _KernelPicture:
    """Eliminate the detectors inside the active set ``columns``."""
    pivots, pivot_rows, work, used = _gauss_jordan(check_matrix, syndrome, columns)
    if work[~used, -1].any():
        raise ValueError("The syndrome has no solution inside the active set.")
    pivot_set = set(pivots)
    free = [column for column in columns if column not in pivot_set]
    # Pivot r fires in the base solution (all free mechanisms off) iff its
    # reduced syndrome bit is set; switching free mechanisms on toggles it
    # through the reduced row.
    base = work[pivot_rows, -1]
    reduced = work[np.ix_(pivot_rows, free)].astype(np.int64)
    supports = [[int(k) for k in np.flatnonzero(row)] for row in reduced]
    pivot_observables = observable_matrix[:, pivots].astype(np.int64)
    return _KernelPicture(
        pivots=pivots,
        base=base,
        free=free,
        supports=supports,
        observable_flips=(observable_matrix[:, free] + pivot_observables @ reduced) % 2,
        base_class=int(
            ((pivot_observables @ base) % 2)
            @ (1 << np.arange(observable_matrix.shape[0]))
        ),
    )


def _register_masses(chain: ParityChain, num_obs: int) -> Tuple[np.ndarray, float]:
    """Squared norm of every value of the leading ``num_obs`` register sites.

    Returns the unnormalized masses (index bit ``j`` is register site ``j``)
    and the logarithm of the scale divided out of them.
    """
    environment = np.ones((1, 1))
    log_scale = 0.0
    for site in range(len(chain) - 1, num_obs - 1, -1):
        tensor = chain.tensors[site]
        environment = np.einsum(
            "lpr,rs,mps->lm", tensor, environment, tensor, optimize=True
        )
        scale = float(np.abs(environment).max())
        if not np.isfinite(scale) or scale <= 0:
            raise ArithmeticError(
                "The state vanished or overflowed during the contraction."
            )
        environment = environment / scale
        log_scale += float(np.log(scale))
    register = np.ones((1, 1))
    for site in range(num_obs):
        tensor = chain.tensors[site]
        register = np.concatenate(
            [register @ tensor[:, 0, :], register @ tensor[:, 1, :]], axis=0
        )
    return np.einsum("cr,rs,cs->c", register, environment, register), log_scale


def active_set_class_masses(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    problem: DemProblem,
    syndrome: np.ndarray,
    active: Sequence[int],
    chi_max: int = 16,
    check_matrix: Optional[np.ndarray] = None,
    observable_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Probability mass of every logical class inside an active set.

    Mechanisms outside ``active`` are pinned to "did not fire". Inside the
    active set every solution of the syndrome is summed, up to the truncation
    to ``chi_max``.

    Parameters
    ----------
    problem : DemProblem
        The detector error model.
    syndrome : np.ndarray
        One bit per detector.
    active : Sequence[int]
        The active mechanisms. Their order is the pivot preference of the
        elimination (earlier mechanisms become pivots first) and the order of
        the free mechanisms on the chain.
    chi_max : int
        The bond dimension of the compression.
    check_matrix, observable_matrix : Optional[np.ndarray]
        Dense ``uint8`` detector and observable matrices, to avoid rebuilding
        them for every syndrome.
    Returns
    -------
    (class_masses, info)
        ``class_masses[c]`` is the probability of the solutions inside the
        active set whose observable pattern is ``c`` (bit ``j`` of ``c`` is
        observable ``j``), normalized to sum to one. ``info`` holds
        ``log_mass`` (logarithm of the retained probability that the syndrome
        is produced by mechanisms of the active set alone), ``lost`` (accumulated discarded
        fraction of the squared norm), ``bond`` (largest bond dimension),
        ``free`` (number of free mechanisms) and ``rank`` (number of pivots).

    Raises
    ------
    ValueError
        If the active set holds no solution of the syndrome, or if
        ``chi_max`` is not positive.
    ArithmeticError
        If the state or every class mass vanishes during the contraction, or
        if no assignment of the active set has positive probability.
    """
    if chi_max < 1:
        raise ValueError("The bond dimension must be positive.")
    syndrome = _validated_syndrome(problem, syndrome).astype(np.uint8) % 2
    if check_matrix is None or observable_matrix is None:
        sparse_check, sparse_observable = problem_matrices(problem)
        check_matrix = np.asarray(sparse_check.todense(), dtype=np.uint8)
        observable_matrix = np.asarray(sparse_observable.todense(), dtype=np.uint8)
    num_obs = problem.num_observables
    probs = np.asarray(problem.probs, dtype=float)
    columns = list(dict.fromkeys(int(i) for i in active))
    if any(i < 0 or i >= problem.num_mechanisms for i in columns):
        raise ValueError("The active set names a mechanism outside the model.")
    kernel = _kernel_picture(check_matrix, observable_matrix, syndrome, columns)
    info: Dict[str, Any] = {"free": len(kernel.free), "rank": len(kernel.pivots)}

    # Probability of each pivot as a function of the parity of its support:
    # index 0 is the base value, index 1 the toggled one.
    pivot_probs = probs[kernel.pivots]
    pivot_weights = np.where(
        kernel.base[:, None] == 1,
        np.stack([pivot_probs, 1 - pivot_probs], axis=1),
        np.stack([1 - pivot_probs, pivot_probs], axis=1),
    )
    # Every mechanism outside the active set is off.
    inactive = np.ones(problem.num_mechanisms, dtype=bool)
    inactive[columns] = False
    with np.errstate(divide="ignore"):
        log_scale = float(np.sum(np.log1p(-probs[inactive])))
        # A pivot without free support takes its base value whatever the
        # free mechanisms do.
        log_scale += float(
            sum(
                np.log(pivot_weights[r, 0])
                for r, support in enumerate(kernel.supports)
                if not support
            )
        )
    if not np.isfinite(log_scale):
        # An inactive mechanism of probability one, or a pivot forced to a
        # value of probability zero: the restricted event cannot happen.
        raise ArithmeticError(
            "No assignment of the active set has positive probability."
        )
    if num_obs + len(kernel.free) == 0:
        info.update(log_mass=log_scale, lost=0.0, bond=1)
        return np.ones(1), info

    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    free_priors = [np.array([1 - probs[c], probs[c]]) for c in kernel.free]

    chain = ParityChain([plus] * num_obs + [np.sqrt(w) for w in free_priors])
    # Soft factors: pivots that do not fire in the base solution first, short
    # supports before long ones.
    order = sorted(
        range(len(kernel.pivots)),
        key=lambda r: (int(kernel.base[r]), len(kernel.supports[r])),
    )
    for r in order:
        weights = pivot_weights[r]
        if not kernel.supports[r]:
            continue
        log_scale += float(np.log(weights.max()))
        chain.apply_parity(
            [num_obs + k for k in kernel.supports[r]],
            chi_max,
            np.sqrt(weights / weights.max()),
        )
    # Hard factors: register site j carries the parity of the generators that
    # flip observable j relative to the base solution.
    for j in range(num_obs):
        flips = np.flatnonzero(kernel.observable_flips[j])
        if flips.size == 0:
            chain.project_bit(j, 0)
        else:
            chain.apply_parity([j] + [num_obs + int(k) for k in flips], chi_max)

    relative, log_environment = _register_masses(chain, num_obs)
    masses = np.empty_like(relative)
    masses[np.arange(relative.size) ^ kernel.base_class] = relative
    total = float(masses.sum())
    if not np.isfinite(total) or total <= 0:
        raise ArithmeticError("Every class mass vanished during the contraction.")
    info.update(
        log_mass=float(
            np.log(total)
            + log_environment
            + 2 * chain.log_norm
            + num_obs * np.log(2.0)
            + log_scale
        ),
        lost=chain.lost,
        bond=chain.max_bond(),
    )
    return masses / total, info
