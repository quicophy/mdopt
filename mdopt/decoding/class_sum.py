"""Class-summed decoding of a detector error model on a candidate active set.

:func:`~mdopt.decoding.dem.decode_dem` contracts every mechanism of the model.
For quantum LDPC memory circuits that chain is thousands of sites long and its
detector constraints have long spans, so a small bond dimension truncates away
the answer. This module contracts a much smaller problem instead, one that a
bond dimension of 16 solves almost exactly.

For one syndrome the decoder goes through four steps.

1. **Candidates.** BP-OSD gives one solution. For every observable ``j``,
   BP-OSD on the check matrix extended by the row of observable ``j``, with
   that observable forced to the opposite value, gives a solution in a
   neighboring logical class. The search is recentered on the most likely
   candidate found so far and repeated a few times.
2. **Active set.** The union of the supports of all candidates. Every other
   mechanism is pinned to "did not fire".
3. **Kernel picture.** Gauss-Jordan elimination of the detector matrix
   restricted to the active set picks one pivot mechanism per independent
   detector. Every assignment of the remaining *free* mechanisms is then a
   valid solution: a pivot fires if and only if its base bit and the parity
   of the free mechanisms in its reduced row disagree. No hard syndrome
   constraint is left, so truncation can only lose probability mass of valid
   solutions.
4. **Class sum.** An MPS over one register site per observable followed by
   one site per free mechanism holds the square roots of the probabilities.
   Each pivot is a soft parity factor, each observable a hard parity with its
   register site. The squared norm of each register value is the total
   probability of that logical class inside the active set, and the decoder
   returns the heaviest class.

The contraction applies each parity factor exactly (the bonds it crosses
double) and then compresses with a canonical SVD sweep, and the bond cut never
splits a degenerate multiplet of singular values. Class masses are squared
norms, so they are non-negative by construction.

**Tempering.** Applying every factor at full strength in one pass works as
long as nothing is truncated. On larger codes the active set has a hundred or
more free mechanisms, and the one-pass contraction then fails: a truncation
made early leaves a small error on configurations that the state has not yet
been told are unlikely, and the factors applied later can amplify that error
by ``(1 - p) / p`` each. The tempered contraction starts from the uniform
state and applies every factor (the priors of the free mechanisms and the
pivot factors) to the power ``1 / stages``, in ``stages`` passes. Each pass
truncates a state that already carries a share of every factor, so the
truncation keeps the configurations that matter. Without truncation both
contractions give the same masses. The decoder runs the one-pass contraction
first and repeats it with tempering only when that pass discarded weight.

**Hard syndromes.** The syndromes whose first pass discards weight are the
ones where the decoder still fails, and two options spend more effort on them
only. ``temper_chi_max`` gives the tempered contraction its own, larger bond
dimension. ``extra_centers`` widens the candidate search: classes that the
search has already found become centers of further class flips, which reaches
classes more than one observable flip away from the most likely candidates.

On the bivariate bicycle [[72, 12, 6]] code under circuit-level depolarizing
noise this decoder makes fewer logical errors than BP-OSD on the same shots at
1, 2 and 6 syndrome-extraction rounds. On the [[90, 8, 10]], [[108, 8, 10]]
and [[144, 12, 12]] codes of the same family the tempered pass is what keeps
it ahead: on [[144, 12, 12]] the one-pass contraction alone does worse than
picking the most likely single candidate.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import svd as scipy_svd
from scipy.sparse import csr_matrix, vstack

from mdopt.decoding.dem import DemProblem, _validated_syndrome

#: BP-OSD settings used to produce the candidates: min-sum belief propagation
#: followed by combination-sweep ordered statistics decoding of order 10.
DEFAULT_BPOSD_OPTIONS: Dict[str, Any] = {
    "max_iter": 100,
    "bp_method": "minimum_sum",
    "ms_scaling_factor": 0.9,
    "osd_method": "osd_cs",
    "osd_order": 10,
}


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
    nonzero = int(np.sum(singular_values > singular_values[0] * 1e-13))
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
        self.tensors: List[np.ndarray] = [
            np.asarray(vector, dtype=float).reshape(1, 2, 1) for vector in vectors
        ]
        self.center = 0
        self.log_norm = 0.0
        self.lost = 0.0
        for site in range(len(self.tensors) - 1, 0, -1):
            self._orthogonalize_right(site)
        self._normalize()

    def __len__(self) -> int:
        return len(self.tensors)

    def _normalize(self) -> None:
        norm = float(np.linalg.norm(self.tensors[self.center]))
        if norm == 0:
            raise ArithmeticError("The constraint annihilated the state.")
        self.tensors[self.center] = self.tensors[self.center] / norm
        self.log_norm += float(np.log(norm))

    def _orthogonalize_left(self, site: int) -> None:
        """Make ``site`` left-canonical and push the remainder to its right."""
        left, phys, right = self.tensors[site].shape
        isometry, rest = np.linalg.qr(self.tensors[site].reshape(left * phys, right))
        self.tensors[site] = isometry.reshape(left, phys, -1)
        self.tensors[site + 1] = np.einsum("ab,bdc->adc", rest, self.tensors[site + 1])

    def _orthogonalize_right(self, site: int) -> None:
        """Make ``site`` right-canonical and push the remainder to its left."""
        left, phys, right = self.tensors[site].shape
        isometry, rest = np.linalg.qr(self.tensors[site].reshape(left, phys * right).T)
        self.tensors[site] = isometry.T.reshape(-1, phys, right)
        self.tensors[site - 1] = np.einsum(
            "adb,bc->adc", self.tensors[site - 1], rest.T
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
            The sites whose parity enters the factor.
        chi_max : int
            The bond dimension to compress to.
        weights : Union[Sequence[float], np.ndarray]
            The factor for even and for odd parity.
        """
        support = {int(site) for site in sites}
        if not support:
            raise ValueError("A parity factor needs at least one site.")
        first, last = min(support), max(support)
        if first < 0 or last >= len(self.tensors):
            raise ValueError("A parity factor acts on a site outside the chain.")
        self.move_center(first)
        if first == last:
            factor = np.asarray(weights, dtype=float)[None, :, None]
            self.tensors[first] = self.tensors[first] * factor
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
        if keep < singular_values.size:
            total = float(np.sum(singular_values**2))
            self.lost += float(np.sum(singular_values[keep:] ** 2)) / total
        self.tensors[site] = right_vectors[:keep].reshape(keep, phys, right)
        self.tensors[site - 1] = np.einsum(
            "adb,bc->adc",
            self.tensors[site - 1],
            left_vectors[:, :keep] * singular_values[:keep],
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
        return csr_matrix(
            (data, (row_index, column_index)), shape=(len(rows), num_mech)
        )

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
    stages: int = 1,
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
    stages : int
        The number of tempering passes. With 1 the chain starts from the
        priors of the free mechanisms and every pivot factor is applied once
        at full strength. With more, the chain starts from the uniform state
        and every prior and pivot factor is applied to the power
        ``1 / stages`` in each of ``stages`` passes, which keeps the
        truncation faithful when the active set is large. Both give the same
        masses when nothing is truncated.

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
        If the active set holds no solution of the syndrome, or if ``stages``
        is not positive.
    """
    if stages < 1:
        raise ValueError("The number of tempering stages must be positive.")
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
    log_scale = float(np.sum(np.log1p(-probs[inactive])))
    if num_obs + len(kernel.free) == 0:
        log_scale += float(np.sum(np.log(pivot_weights[:, 0])))
        info.update(log_mass=log_scale, lost=0.0, bond=1)
        return np.ones(1), info

    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    free_priors = [np.array([1 - probs[c], probs[c]]) for c in kernel.free]

    def amplitude(weights: np.ndarray) -> np.ndarray:
        """The share of a probability factor that one pass puts on the chain."""
        return np.sqrt(weights) if stages == 1 else weights ** (0.5 / stages)

    if stages == 1:
        chain = ParityChain([plus] * num_obs + [amplitude(w) for w in free_priors])
    else:
        chain = ParityChain([plus] * num_obs + [np.ones(2)] * len(kernel.free))
    # Soft factors: pivots that do not fire in the base solution first, short
    # supports before long ones.
    order = sorted(
        range(len(kernel.pivots)),
        key=lambda r: (int(kernel.base[r]), len(kernel.supports[r])),
    )
    for _ in range(stages):
        if stages > 1:
            for k, weights in enumerate(free_priors):
                chain.apply_parity([num_obs + k], chi_max, amplitude(weights))
        for r in order:
            weights = pivot_weights[r]
            if not kernel.supports[r]:
                log_scale += float(np.log(weights[0])) / stages
                continue
            log_scale += float(np.log(weights.max())) / stages
            chain.apply_parity(
                [num_obs + k for k in kernel.supports[r]],
                chi_max,
                amplitude(weights / weights.max()),
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
    info.update(
        log_mass=float(
            np.log(max(total, 1e-300))
            + log_environment
            + 2 * chain.log_norm
            + num_obs * np.log(2.0)
            + log_scale
        ),
        lost=chain.lost,
        bond=chain.max_bond(),
    )
    return masses / total, info


class ClassSumDecoder:  # pylint: disable=too-many-instance-attributes
    """Class-summed MPS decoder seeded by BP-OSD candidates.

    Parameters
    ----------
    problem : DemProblem
        The detector error model. Every prior must be positive and at most
        1/2, so that no error is the most likely explanation of the empty
        syndrome.
    chi_max : int
        The bond dimension of the class-sum contraction.
    recenter_rounds : int
        How many times the class-flip search is recentered on the most likely
        candidate.
    temper_stages : int
        The number of passes of the tempered contraction that replaces the
        one-pass contraction when the latter truncated too much (see
        :func:`active_set_class_masses`). With 1 the decoder never tempers.
    temper_threshold : float
        The one-pass contraction is replaced by the tempered one when its
        discarded fraction of the squared norm (``lost``) reaches this value.
    temper_chi_max : Optional[int]
        The bond dimension of the tempered contraction. ``None`` uses
        ``chi_max``. The tempered pass only runs on the syndromes whose first
        pass truncated, so a larger value here costs much less than a larger
        ``chi_max``.
    extra_centers : int
        Widens the candidate search on the syndromes whose first pass
        truncated: up to this many further classes found by the search, most
        likely first, are each used once as the center of the class flips,
        and the class sum is repeated on the larger active set. Every center
        costs one BP-OSD run per observable. With 0 the search is not
        widened.
    **bposd_options
        Overrides of :data:`DEFAULT_BPOSD_OPTIONS`, passed to
        ``ldpc.BpOsdDecoder``.

    Notes
    -----
    Building the decoder constructs one BP-OSD decoder for the model and one
    per observable, so it should be built once and reused for all syndromes of
    a model.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        problem: DemProblem,
        chi_max: int = 16,
        recenter_rounds: int = 3,
        temper_stages: int = 2,
        temper_threshold: float = 1e-8,
        temper_chi_max: Optional[int] = None,
        extra_centers: int = 0,
        **bposd_options: Any,
    ):
        # Imported here so that importing mdopt does not pay for ldpc, which
        # only this decoder uses.
        from ldpc import BpOsdDecoder  # pylint: disable=import-outside-toplevel

        probs = np.asarray(problem.probs, dtype=float)
        if probs.size == 0:
            raise ValueError("The detector error model has no mechanisms.")
        if np.any(probs <= 0) or np.any(probs > 0.5):
            # Above 1/2 a mechanism is more likely to fire than not, and the
            # empty syndrome would no longer be best explained by no error.
            raise ValueError("Every prior must be positive and at most 1/2.")
        if chi_max < 1:
            raise ValueError("The bond dimension must be positive.")
        if temper_stages < 1:
            raise ValueError("The number of tempering stages must be positive.")
        if temper_threshold < 0:
            raise ValueError("The tempering threshold cannot be negative.")
        if temper_chi_max is not None and temper_chi_max < 1:
            raise ValueError("The bond dimension must be positive.")
        if extra_centers < 0:
            raise ValueError("The number of extra centers cannot be negative.")
        self.problem = problem
        self.chi_max = chi_max
        self.recenter_rounds = recenter_rounds
        self.temper_stages = temper_stages
        self.temper_threshold = temper_threshold
        self.temper_chi_max = chi_max if temper_chi_max is None else temper_chi_max
        self.extra_centers = extra_centers
        self.num_observables = problem.num_observables
        self._check, self._observables = problem_matrices(problem)
        self._dense_check = np.asarray(self._check.todense(), dtype=np.uint8)
        self._dense_observables = np.asarray(
            self._observables.todense(), dtype=np.uint8
        )
        self._llr = np.log(probs / (1 - probs))
        self._powers = 1 << np.arange(self.num_observables)
        options = {**DEFAULT_BPOSD_OPTIONS, **bposd_options}
        channel = [float(p) for p in probs]
        self._bposd = BpOsdDecoder(self._check, error_channel=channel, **options)
        self._flipped = [
            BpOsdDecoder(
                vstack([self._check, self._observables[j]]).tocsr(),
                error_channel=channel,
                **options,
            )
            for j in range(self.num_observables)
        ]

    def class_of(self, mechanisms: np.ndarray) -> int:
        """Observable pattern of a mechanism set (bit ``j`` is observable ``j``)."""
        return int(((self._observables @ mechanisms) % 2) @ self._powers)

    def _satisfies(self, mechanisms: np.ndarray, syndrome: np.ndarray) -> bool:
        return not ((self._check @ mechanisms) % 2 != syndrome).any()

    def candidates(
        self, syndrome: np.ndarray, seed: np.ndarray
    ) -> Dict[int, Tuple[float, np.ndarray]]:
        """Candidate solutions from class-flipped BP-OSD reruns.

        Parameters
        ----------
        syndrome : np.ndarray
            One bit per detector.
        seed : np.ndarray
            A solution of the syndrome, normally the BP-OSD solution.

        Returns
        -------
        Dict[int, Tuple[float, np.ndarray]]
            For each logical class reached, the log-likelihood ratio and the
            mechanisms of its most likely candidate.
        """
        center = self.class_of(seed)
        found = {center: (float(self._llr @ seed), seed.copy())}
        tried = set()
        for _ in range(self.recenter_rounds):
            if center in tried:
                break
            tried.add(center)
            self._flip_around(center, syndrome, found)
            center = max(found, key=lambda label: found[label][0])
        return found

    def _flip_around(
        self,
        center: int,
        syndrome: np.ndarray,
        found: Dict[int, Tuple[float, np.ndarray]],
    ) -> None:
        """Flip every observable of ``center`` once and record the solutions."""
        for j in range(self.num_observables):
            target = ((center >> j) & 1) ^ 1
            extended = np.concatenate([syndrome, [target]]).astype(np.uint8)
            solution = np.asarray(self._flipped[j].decode(extended), dtype=np.uint8)
            if not self._satisfies(solution, syndrome):
                continue
            label = self.class_of(solution)
            likelihood = float(self._llr @ solution)
            if label not in found or likelihood > found[label][0]:
                found[label] = (likelihood, solution.copy())

    def widen(
        self, syndrome: np.ndarray, found: Dict[int, Tuple[float, np.ndarray]]
    ) -> Dict[int, Tuple[float, np.ndarray]]:
        """Candidates of a wider search around the classes already found.

        Up to ``extra_centers`` of the classes in ``found``, most likely
        first, are each used once as the center of the class flips.

        Parameters
        ----------
        syndrome : np.ndarray
            One bit per detector.
        found : Dict[int, Tuple[float, np.ndarray]]
            The candidates of :meth:`candidates`.

        Returns
        -------
        Dict[int, Tuple[float, np.ndarray]]
            ``found`` extended by the new classes, with the most likely
            candidate of every class.
        """
        wider = dict(found)
        centers = sorted(found, key=lambda label: -found[label][0])
        for center in centers[: self.extra_centers]:
            self._flip_around(center, syndrome, wider)
        return wider

    @staticmethod
    def _active_set(found: Dict[int, Tuple[float, np.ndarray]]) -> List[int]:
        """Union of the candidate supports, most likely candidate first."""
        active: List[int] = []
        seen = set()
        for _, (_, solution) in sorted(found.items(), key=lambda item: -item[1][0]):
            for i in np.flatnonzero(solution):
                if int(i) not in seen:
                    seen.add(int(i))
                    active.append(int(i))
        return active

    def _flips(self, label: int) -> np.ndarray:
        return (label >> np.arange(self.num_observables)) & 1

    def _class_masses(
        self, syndrome: np.ndarray, active: Sequence[int], stages: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return active_set_class_masses(
            self.problem,
            syndrome,
            active,
            self.chi_max if stages == 1 else self.temper_chi_max,
            check_matrix=self._dense_check,
            observable_matrix=self._dense_observables,
            stages=stages,
        )

    def decode(  # pylint: disable=too-many-locals
        self, syndrome: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict the observable flips for one syndrome.

        Parameters
        ----------
        syndrome : np.ndarray
            One bit per detector.

        Returns
        -------
        (predicted_flips, info)
            ``predicted_flips`` has one bit per observable. ``info`` holds
            ``predicted_class`` and ``bposd_class`` (observable patterns as
            integers, bit ``j`` is observable ``j``), ``best_candidate_class``
            (the class of the most likely single candidate), ``margin``
            (difference between the two largest normalized class masses),
            ``lost`` (discarded fraction of the squared norm in the
            contraction that gave the verdict), ``tempered`` (whether that
            was the tempered contraction), ``widened`` (whether the candidate
            search was widened), ``active`` and
            ``free`` (sizes of the active set and of its free part) and
            ``candidates`` (number of logical classes reached). A tie for the
            heaviest class goes to the BP-OSD class when it is one of the
            tied classes. The empty syndrome is answered with no flips
            without a search. If BP-OSD returns a mechanism set that does not reproduce
            the syndrome, its class is returned with ``candidates`` set to 0.
        """
        syndrome = _validated_syndrome(self.problem, syndrome).astype(np.uint8) % 2
        info: Dict[str, Any] = {"predicted_class": 0, "bposd_class": 0}
        info.update(best_candidate_class=0, margin=1.0, lost=0.0, active=0, free=0)
        info.update(candidates=1, tempered=False, widened=False)
        if not syndrome.any():
            return self._flips(0), info
        seed = np.asarray(self._bposd.decode(syndrome), dtype=np.uint8)
        bposd_class = self.class_of(seed)
        info.update(predicted_class=bposd_class, bposd_class=bposd_class)
        info["best_candidate_class"] = bposd_class
        if not self._satisfies(seed, syndrome):
            info.update(margin=0.0, candidates=0)
            return self._flips(bposd_class), info

        found = self.candidates(syndrome, seed)
        active = self._active_set(found)
        masses, contraction = self._class_masses(syndrome, active, 1)
        hard = contraction["lost"] >= self.temper_threshold
        widened = False
        if hard and self.extra_centers > 0:
            found = self.widen(syndrome, found)
            active = self._active_set(found)
            masses, contraction = self._class_masses(syndrome, active, 1)
            hard = contraction["lost"] >= self.temper_threshold
            widened = True
        tempered = False
        if self.temper_stages > 1 and hard:
            try:
                masses, contraction = self._class_masses(
                    syndrome, active, self.temper_stages
                )
                tempered = True
            except ArithmeticError:
                # The tempered state underflowed: keep the one-pass verdict.
                pass
        ranking = np.argsort(masses)[::-1]
        predicted = int(ranking[0])
        margin = (
            float(masses[ranking[0]] - masses[ranking[1]]) if masses.size > 1 else 1.0
        )
        if masses[bposd_class] >= masses[predicted] - 1e-6:
            predicted = bposd_class
        info.update(
            predicted_class=predicted,
            best_candidate_class=max(found, key=lambda label: found[label][0]),
            margin=margin,
            lost=contraction["lost"],
            tempered=tempered,
            widened=widened,
            active=len(active),
            free=contraction["free"],
            candidates=len(found),
        )
        return self._flips(predicted), info
