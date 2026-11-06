"""Decode stim detector error models with an MPS marginalizer.

A detector error model (DEM) is a purely classical decoding problem: each
error mechanism is a bit ``m_i`` that fires independently with prior ``p_i``,
each detector is a parity check ``<D_d, m> = s_d`` over the mechanisms that
flip it, and each logical observable is a parity readout ``<O_j, m>``. Maximum
likelihood decoding asks for the observable-flip pattern whose total
probability mass, summed over all mechanism sets consistent with the observed
syndrome, is largest.

The tensor-network realization flattens the mechanisms onto one MPS chain --
one site per mechanism, plus one logical site per observable at the front --
and reuses the parity-check machinery of the code decoders: detectors become
XOR constraints, observables become COPY/XOR readouts, and the mechanisms are
marginalized away. Unlike the code decoders, the prior is applied as *linear*
weights, so the readout is the exact class probability mass ``sum(w)`` rather
than the amplitude form ``sum(sqrt(w))``; MAP against the literature is the
target here. Contraction is exact at large ``chi_max``; the cost of finite
``chi_max`` is governed by the spans of the detector constraints after the
mechanisms are laid out on the chain, which makes mechanism ordering the
central performance question (see arXiv:2310.10722 for the 3D-tensor-network
view of the same problem).
"""

import logging
from dataclasses import dataclass, field
from typing import cast, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import stim

from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.utils import create_custom_product_state

LOGGER = logging.getLogger(__name__)
from mdopt.optimizer.utils import (
    COPY_LEFT,
    SWAP,
    XOR_BULK,
    XOR_LEFT,
    XOR_RIGHT,
)
from mdopt.optimizer.utils import (
    TruncationCertificate,
    apply_constraints,
    optimize_qubit_order,
)


@dataclass
class DemProblem:
    """A merged, flattened detector error model.

    Attributes
    ----------
    probs : List[float]
        Prior probability of each mechanism.
    detector_rows : List[List[int]]
        For each detector, the mechanisms that flip it.
    observable_rows : List[List[int]]
        For each observable, the mechanisms that flip it.
    num_detectors, num_observables : int
        Declared counts from the model (detectors with no mechanisms are kept
        so syndrome indexing matches stim's).
    """

    probs: List[float]
    detector_rows: List[List[int]]
    observable_rows: List[List[int]]
    num_detectors: int
    num_observables: int

    @property
    def num_mechanisms(self) -> int:
        return len(self.probs)


def dem_to_problem(dem: stim.DetectorErrorModel) -> DemProblem:
    """Flatten a stim DEM and merge duplicate mechanisms.

    Mechanisms with identical detector and observable sets are merged with
    ``p = p1 (1 - p2) + p2 (1 - p1)`` -- firing both is indistinguishable from
    firing neither. Error decomposition (the ``^`` splits stim adds for
    matching decoders) must be OFF: maximum likelihood wants the undecomposed
    hyperedges.
    """
    flat = dem.flattened()
    merged: Dict[Tuple[FrozenSet[int], FrozenSet[int]], float] = {}
    for instruction in flat:
        if instruction.type != "error":
            continue
        probability = instruction.args_copy()[0]
        detectors: set = set()
        observables: set = set()
        for target in instruction.targets_copy():
            # stim XORs repeated targets within one error line (a doubled
            # detector cancels), so toggle rather than collect.
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                observables ^= {target.val}
            elif target.is_separator():
                raise ValueError(
                    "The DEM contains decomposed errors ('^' separators). "
                    "Build it with decompose_errors=False: maximum likelihood "
                    "decoding wants the undecomposed hyperedges."
                )
        key = (frozenset(detectors), frozenset(observables))
        if key in merged:
            existing = merged[key]
            merged[key] = existing * (1 - probability) + probability * (1 - existing)
        else:
            merged[key] = probability

    probs: List[float] = []
    detector_rows: List[List[int]] = [[] for _ in range(dem.num_detectors)]
    observable_rows: List[List[int]] = [[] for _ in range(dem.num_observables)]
    for (detector_key, observable_key), probability in merged.items():
        index = len(probs)
        probs.append(probability)
        for det in detector_key:
            detector_rows[det].append(index)
        for obs in observable_key:
            observable_rows[obs].append(index)
    return DemProblem(
        probs=probs,
        detector_rows=[sorted(row) for row in detector_rows],
        observable_rows=[sorted(row) for row in observable_rows],
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
    )


def _validated_syndrome(problem: DemProblem, syndrome) -> np.ndarray:
    """One bit per detector, as a 1-D int array; anything else is rejected.

    A scalar or length-one syndrome would otherwise broadcast across every
    detector and silently solve an all-equal syndrome.
    """
    array = np.asarray(syndrome)
    if array.ndim != 1 or array.shape[0] != problem.num_detectors:
        raise ValueError(
            f"The syndrome must have exactly one bit per detector "
            f"({problem.num_detectors}), given shape {array.shape}."
        )
    return array.astype(int) % 2


def solve_representative(problem: DemProblem, syndrome: np.ndarray) -> np.ndarray:
    """Any mechanism set with the observed syndrome, by GF(2) elimination.

    Which representative is returned is irrelevant to the decoded class -- the
    posterior is a function of the syndrome alone -- and
    ``test_dem_class_masses_are_representative_independent`` pins that.
    """
    num_mech = problem.num_mechanisms
    matrix = np.zeros((problem.num_detectors, num_mech + 1), dtype=int)
    for det, row in enumerate(problem.detector_rows):
        matrix[det, row] = 1
    matrix[:, num_mech] = _validated_syndrome(problem, syndrome)

    pivots: List[Tuple[int, int]] = []
    row_index = 0
    for col in range(num_mech):
        pivot = next(
            (r for r in range(row_index, matrix.shape[0]) if matrix[r, col]), None
        )
        if pivot is None:
            continue
        matrix[[row_index, pivot]] = matrix[[pivot, row_index]]
        for other in range(matrix.shape[0]):
            if other != row_index and matrix[other, col]:
                matrix[other] = (matrix[other] + matrix[row_index]) % 2
        pivots.append((row_index, col))
        row_index += 1

    for r in range(row_index, matrix.shape[0]):
        if matrix[r, num_mech]:
            raise ValueError(
                "The syndrome is inconsistent with the detector error model."
            )

    solution = np.zeros(num_mech, dtype=int)
    for r, col in pivots:
        solution[col] = matrix[r, num_mech]
    return solution


def _constraint_sites(mechanisms: List[int], offset: int) -> List[List[int]]:
    """[XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT] site lists for one detector.

    The boundary tensors are directional, so the sites are sorted here
    rather than trusting the caller: dem_to_problem emits sorted rows, but a
    hand-built DemProblem need not.
    """
    sites = sorted(m + offset for m in mechanisms)
    bulk = set(sites[1:-1])
    return [
        [sites[0]],
        sites[1:-1],
        [s for s in range(sites[0] + 1, sites[-1]) if s not in bulk],
        [sites[-1]],
    ]


def _logical_sites(
    mechanisms: List[int], logical_site: int, offset: int
) -> List[List[int]]:
    """[COPY_LEFT, XOR_BULK, SWAP, XOR_RIGHT] for one observable readout."""
    sites = sorted(m + offset for m in mechanisms)
    bulk = set(sites[:-1])
    return [
        [logical_site],
        sites[:-1],
        [s for s in range(logical_site + 1, sites[-1]) if s not in bulk],
        [sites[-1]],
    ]


def decode_dem(
    problem: DemProblem,
    syndrome: np.ndarray,
    chi_max: int = int(1e4),
    cut: float = float(1e-17),
    renormalize: bool = True,
    silent: bool = True,
    tolerance: float = float(1e-12),
    representative: Optional[np.ndarray] = None,
    prior_reweighting: Optional[str] = None,
    reweighting_floor: Optional[float] = None,
    reweighting_cap: float = 0.5,
    bp_iterations: int = 60,
    return_certificate: bool = False,
    negative_mass: str = "raise",
):
    """Maximum-likelihood observable prediction for one syndrome.

    Parameters
    ----------
    prior_reweighting : Optional[str]
        ``None`` contracts the prior as given. ``"BP"`` replaces every
        mechanism's prior by its belief-propagation posterior marginal
        (clipped relative to the prior, see ``reweighting_floor``) before the
        contraction and divides it out exactly at marginalization. The
        syndrome-consistent configurations then carry a mass fraction of
        order one instead of order ``P(syndrome)``, which is what keeps a
        finite ``chi_max`` from rebuilding the class masses out of truncation
        residue (see :func:`mdopt.decoding.decoding.decode_custom`).
    reweighting_floor : Optional[float]
        The clip of the marginals, relative to the prior: the contracted
        weight of either branch of site ``i`` stays at least
        ``floor * P_i(b)``, so every undo ratio ``P_i(b) / q_i(b)`` is at
        most ``1 / floor`` (``None`` means 0.5). Unlike the code decoders'
        absolute floor this does not underflow a chain of thousands of
        mechanisms (the clip costs the consistent configurations about
        ``exp(-floor * n * p)`` of mass), and it bounds how much a
        truncation residue is amplified when the prior is divided back in:
        at 0.1 a configuration with k reweighted branches comes back scaled
        by up to 10^k and round-off alone moved a class mass by 1e-4 on a
        13-mechanism model, while at 0.5 (2^k) the masses are exact to 1e-11
        and a d=5 color-code ladder is monotone in chi_max.
        Run the reweighted contraction with ``cut=0.0``: configurations the
        SVD cutoff drops as negligible in the contracted weights are exactly
        the ones the undo scales back up.
    reweighting_cap : float
        The upper clip of the marginal of the a-priori unlikely branch of a
        site, the mechanism firing (default 0.5). Sites BP believes fired are
        where the undo amplifies: the unfired branch is scaled back by its
        prior over its contracted weight, at most ``1 / (1 - cap)`` per such
        site, so a cap of 0.5 trades a factor two of survival per fired
        mechanism for an amplification of at most two of whatever the
        contraction got wrong there. Without the cap a truncation residue
        certified at 1e-3 came back as an order-one class mass on a d=5
        color-code circuit with about ten fired mechanisms.
    bp_iterations : int
        The number of flooding iterations.
    return_certificate : bool
        Also return the :class:`~mdopt.optimizer.utils.TruncationCertificate`
        of the contraction. It is always computed; a collapsed class-mass
        vector (nothing positive, or not finite) raises ``ArithmeticError``
        with the certificate attached as its ``certificate`` attribute. The
        certificate's ``negativity`` is the most negative class mass as a
        fraction of the peak (0 for an exact contraction).
    negative_mass : str
        What to do with a class mass below -1e-8 of the peak, which only a
        truncation artifact can produce. ``"raise"`` (default) raises
        ``ArithmeticError`` with the certificate attached, so that callers
        that escalate or reject on the exception keep working; ``"clamp"``
        clamps it to zero before the argmax, takes the verdict of the
        positive classes and leaves the caller to gate on
        ``certificate.negativity`` alongside ``bound`` and ``survival``.

    Returns
    -------
    (class_masses, predicted_flips[, certificate])
        ``class_masses[c]`` is proportional to the total probability of the
        mechanism sets whose observable pattern is ``c`` (bit ``j`` of ``c``
        is observable ``j``), and ``predicted_flips`` is the argmax pattern as
        a bit array.
    """
    if prior_reweighting not in (None, "BP"):
        raise ValueError(
            f"prior_reweighting should be None or 'BP', given {prior_reweighting!r}."
        )
    if negative_mass not in ("raise", "clamp"):
        raise ValueError(
            f"negative_mass should be 'raise' or 'clamp', given {negative_mass!r}."
        )
    if reweighting_floor is None:
        reweighting_floor = 0.5
    if not 0 < reweighting_floor <= 1:
        raise ValueError(
            f"reweighting_floor should lie in (0, 1], given {reweighting_floor}."
        )
    if not 0 < reweighting_cap < 1:
        raise ValueError(
            f"reweighting_cap should lie in (0, 1), given {reweighting_cap}."
        )
    # A supplied representative bypasses solve_representative, so the
    # shape check must live here too.
    syndrome = _validated_syndrome(problem, syndrome)
    num_obs = problem.num_observables
    num_mech = problem.num_mechanisms
    offset = num_obs

    # A detector touched by a single mechanism pins it: in the f = m XOR base
    # variables every detector constrains f to even parity, so a one-mechanism
    # row forces f_i = 0. The site stays in |0> (weighted by its prior below)
    # and the constraint itself is dropped (detector_strings keeps rows of
    # weight >= 2 only).
    pinned = set()
    for det, row in enumerate(problem.detector_rows):
        if len(row) == 0 and int(syndrome[det]) % 2 == 1:
            raise ValueError(f"Detector {det} fired but no mechanism flips it.")
        if len(row) == 1:
            pinned.add(row[0])

    if representative is None:
        # Consistent with the syndrome by construction (the solver raises
        # otherwise), so no parity recheck is needed here.
        base = solve_representative(problem, syndrome)
    else:
        base = np.asarray(representative)
        if base.ndim != 1 or base.shape[0] != num_mech:
            raise ValueError(
                f"The representative must have exactly one bit per mechanism "
                f"({num_mech}), given shape {base.shape}."
            )
        base = base.astype(int) % 2
        if np.any((_detector_parities(problem, base) - syndrome) % 2):
            raise ValueError("The supplied representative does not match the syndrome.")

    # Change of variables: the XOR constraints project onto EVEN parity, i.e.
    # the zero-syndrome sector. Writing every consistent mechanism set as
    # m = base XOR f puts the free variable f exactly in that sector, so the
    # chain carries f (initialized all-zero) and each site is weighted by the
    # prior of m_i = base_i XOR f_i:
    #     base_i = 0 ->  (1-p_i, p_i)  for f_i = (0, 1)
    #     base_i = 1 ->  (p_i, 1-p_i)
    # The observable labels read off f; the representative's own observable
    # pattern is XORed back in at the end.
    # A model with no mechanisms decodes trivially: the only consistent
    # event is "nothing happened" (a fired detector was already rejected
    # above), so all mass sits on the no-flip class. Building an MPS here
    # would need at least one site.
    if num_mech == 0 and num_obs == 0:
        if return_certificate:
            return np.array([1.0]), np.zeros(0, dtype=int), TruncationCertificate()
        return np.array([1.0]), np.zeros(0, dtype=int)

    # An observable no mechanism touches is deterministically unflipped: it
    # gets |0> instead of |+>, otherwise the readout would split its mass
    # 50/50 over a flip that cannot happen.
    state_string = (
        "".join("+" if row else "0" for row in problem.observable_rows) + "0" * num_mech
    )
    mps = cast(
        CanonicalMPS,
        create_custom_product_state(
            string=state_string, tolerance=tolerance, form="Right-canonical"
        ),
    )
    # The prior of f_i = (0, 1) per site, and the site weights actually
    # contracted: the prior itself, or the BP posterior marginal of f_i given
    # even parity on every detector, which the undo weights divide out again
    # at marginalization (prior / contracted weight, a positive diagonal that
    # commutes with every XOR and COPY tensor).
    priors = np.array(
        [
            [1.0 - probability, probability]
            if base[i] == 0
            else [probability, 1.0 - probability]
            for i, probability in enumerate(problem.probs)
        ],
        dtype=float,
    ).reshape(num_mech, 2)
    contracted = priors.copy()
    undo: List[Tuple[int, np.ndarray]] = []
    if prior_reweighting == "BP" and num_mech > 0:
        from mdopt.decoding.decoding import _bp_site_marginals

        flip = priors[:, 1].copy()
        for i in pinned:
            flip[i] = 0.0  # forced to f_i = 0
        checks = [list(row) for row in problem.detector_rows if len(row) >= 2]
        marginal = _bp_site_marginals(
            num_mech, checks, np.zeros(len(checks), dtype=int), flip, bp_iterations
        )
        # q_i(1) in [floor * P_i(1), 1 - floor * P_i(0)]: both branches keep at
        # least a fraction floor of their prior weight, so no undo ratio
        # exceeds 1 / floor.
        # Both branches keep at least the fraction `floor` of their prior
        # weight, and the a-priori unlikely branch (the mechanism firing: f=1
        # where the representative has a 0, f=0 where it has a 1) gets at
        # most `cap`, so the likely branch is never scaled back by more than
        # 1 / (1 - cap) at marginalization.
        lower = reweighting_floor * priors[:, 1]
        upper = 1.0 - reweighting_floor * priors[:, 0]
        fired_is_one = priors[:, 1] < priors[:, 0]
        upper = np.where(fired_is_one, np.minimum(upper, reweighting_cap), upper)
        lower = np.where(fired_is_one, lower, np.maximum(lower, 1.0 - reweighting_cap))
        free = np.ones(num_mech, dtype=bool)
        free[list(pinned)] = False
        infeasible = np.flatnonzero(free & (lower > upper + 1e-12))
        if infeasible.size:
            i = int(infeasible[0])
            raise ValueError(
                f"reweighting_floor={reweighting_floor} and reweighting_cap="
                f"{reweighting_cap} are incompatible for mechanism {i} with prior "
                f"{priors[i].tolist()}: the floor keeps its firing marginal in "
                f"[{reweighting_floor * priors[i, 1]:.3g}, "
                f"{1.0 - reweighting_floor * priors[i, 0]:.3g}] while the cap "
                f"requires it {'<=' if fired_is_one[i] else '>='} "
                f"{reweighting_cap if fired_is_one[i] else 1.0 - reweighting_cap:.3g}."
            )
        marginal = np.clip(marginal, lower, upper)
        for i in range(num_mech):
            if i in pinned:
                undo.append((offset + i, np.ones(2)))
                continue
            contracted[i] = [1.0 - marginal[i], marginal[i]]
            # A branch with zero prior has zero contracted weight too (the
            # floor is relative) and carries no amplitude, so its undo
            # weight is immaterial; zero keeps 0/0 out of the chain.
            undo.append(
                (
                    offset + i,
                    np.divide(
                        priors[i],
                        contracted[i],
                        out=np.zeros(2),
                        where=contracted[i] > 0.0,
                    ),
                )
            )
    for i in range(num_mech):
        if i in pinned:
            # f_i is forced to 0; only its prior's f=0 branch contributes.
            mps.tensors[offset + i] = mps.tensors[offset + i] * contracted[i, 0]
            continue
        weight_matrix = np.array(
            [[contracted[i, 0], contracted[i, 1]], [contracted[i, 1], contracted[i, 0]]]
        )
        mps.tensors[offset + i] = np.einsum(
            "ab, ibj -> iaj", weight_matrix, mps.tensors[offset + i]
        )
    certificate = TruncationCertificate()
    # The chain does not start from a unit-norm state: each mechanism site
    # carries its contracted weights (the prior, or the BP marginal), and
    # the pinned sites keep one branch only. Recording the unprojected norm
    # of those weights as the mass gain of this weighting step makes
    # ``survival`` the fraction of that mass kept by the pinned branches,
    # the constraints and the truncations together.
    # The product of thousands of site norms underflows (3,000 mechanisms at
    # probability 0.5 give 2^-1500), so the gain is accumulated as a log.
    log_weights_norm = (
        float(np.sum(np.log(np.linalg.norm(contracted, axis=1)))) if num_mech else 0.0
    )
    certificate.record(0.0, 1.0, 1.0, log_mass_gain=log_weights_norm)

    logicals_tensors = [COPY_LEFT, XOR_BULK, SWAP, XOR_RIGHT]
    constraints_tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]

    observable_strings = [
        _logical_sites(row, j, offset)
        for j, row in enumerate(problem.observable_rows)
        if row
    ]
    detector_strings = [
        _constraint_sites(row, offset) for row in problem.detector_rows if len(row) >= 2
    ]

    from mdopt.decoding.decoding import (
        LOGICAL_CONSTRAINT_GAIN,
        _marginalize_with_weights,
    )

    if observable_strings:
        mps = cast(
            CanonicalMPS,
            apply_constraints(
                mps,
                observable_strings,
                logicals_tensors,
                chi_max=chi_max,
                cut=cut,
                renormalize=renormalize,
                silent=silent,
                strategy="Optimized",
                certificate=certificate,
                gain=LOGICAL_CONSTRAINT_GAIN,
            ),
        )
    mps = cast(
        CanonicalMPS,
        apply_constraints(
            mps,
            detector_strings,
            constraints_tensors,
            chi_max=chi_max,
            cut=cut,
            renormalize=renormalize,
            silent=silent,
            strategy="Optimized",
            certificate=certificate,
        ),
    )
    certificate.finalize(log_final_norm=_log_full_norm(mps))

    if undo:
        logical_mps = _marginalize_with_weights(mps, num_obs, undo, renormalize)
    else:
        logical_mps = mps.marginal(
            sites_to_marginalize=list(range(num_obs, num_obs + num_mech)),
            renormalize=renormalize,
        ).reverse()
    dense = np.atleast_1d(
        np.real(np.asarray(logical_mps.dense(flatten=True, renormalize=False)))
    )

    # Every tensor in the exact contraction is non-negative, so a negative
    # class mass can only be a truncation artifact; taking abs() would turn
    # it into posterior weight that could even win the argmax. Its size
    # relative to the peak goes into the certificate; above 1e-8 of the peak
    # (accumulation noise over thousands of classes sits around 1e-10; 12
    # observables give 4096) it either raises, the default, or is clamped so
    # that the verdict stands on the positive classes and the caller gates
    # on the certificate. A collapsed vector, with nothing positive to choose
    # from, has no verdict to give and always raises.
    if not np.isfinite(dense).all() or float(np.max(dense)) <= 0.0:
        error = ArithmeticError(
            f"The class-mass vector collapsed at chi_max={chi_max} "
            f"(truncation-error bound {certificate.bound:.2e})."
        )
        error.certificate = certificate  # type: ignore[attr-defined]
        raise error
    peak = float(np.max(dense))
    most_negative = float(np.min(dense))
    certificate.negativity = max(0.0, -most_negative / peak)
    if certificate.negativity > 1e-8:
        if negative_mass == "raise":
            error = ArithmeticError(
                f"Negative class mass {most_negative:.3e} "
                f"({100.0 * certificate.negativity:.1f}% of the peak) at "
                f"chi_max={chi_max} (truncation-error bound "
                f"{certificate.bound:.2e}): the contraction is not converged; "
                "raise chi_max, use prior_reweighting='BP', or pass "
                "negative_mass='clamp' to take the verdict of the positive "
                "classes and gate on certificate.negativity."
            )
            error.certificate = certificate  # type: ignore[attr-defined]
            raise error
        if not silent:
            LOGGER.warning(
                "Negative class mass %.3e (%.1f%% of the peak) at chi_max=%d "
                "(truncation-error bound %.2e): the contraction is not "
                "converged; the mass is clamped and the verdict should be "
                "gated on the certificate.",
                most_negative,
                100.0 * certificate.negativity,
                chi_max,
                certificate.bound,
            )
    dense = np.clip(dense, 0.0, None)

    # Relabel from f-classes to m-classes: XOR in the representative's own
    # observable pattern.
    obs_offset = 0
    for j, row in enumerate(problem.observable_rows):
        if row and int(np.sum(base[row]) % 2):
            obs_offset |= 1 << j
    relabeled = np.empty_like(dense)
    for index in range(dense.size):
        relabeled[index ^ obs_offset] = dense[index]

    best = int(np.argmax(relabeled))
    # After reverse(), the first MPS logical site (observable 0) is the LAST
    # dense index bit, so bit j of the flat index is observable j.
    flips = np.array([(best >> j) & 1 for j in range(num_obs)], dtype=int)
    if return_certificate:
        return relabeled, flips, certificate
    return relabeled, flips


def _log_full_norm(mps: CanonicalMPS) -> float:
    """The log-norm of the whole chain, not only of its orthogonality center.

    After the sweeps every site a constraint string reached is an isometry,
    so the center carries their norm; a site no string reached (a pinned
    mechanism outside every span, or every site when no sweep was needed)
    is still a bond-dimension-one tensor carrying its own weight, and an
    isometric bond-dimension-one tensor has norm one, so multiplying in the
    norms of all such sites is exact either way. The norms are summed as
    logs: a chain of thousands of pinned sites has a norm far below the
    smallest float, which is a finite log-norm, not a collapse.
    """
    center = mps.orth_center
    norms = [] if center is None else [float(np.linalg.norm(mps.tensors[center]))]
    for site, tensor in enumerate(mps.tensors):
        if site != center and tensor.shape[0] == 1 and tensor.shape[-1] == 1:
            norms.append(float(np.linalg.norm(tensor)))
    if any(norm <= 0.0 or not np.isfinite(norm) for norm in norms):
        return -np.inf
    return float(np.sum(np.log(norms))) if norms else 0.0


def _detector_parities(problem: DemProblem, mechanisms: np.ndarray) -> np.ndarray:
    return np.array(
        [
            int(np.sum(mechanisms[row]) % 2) if row else 0
            for row in problem.detector_rows
        ],
        dtype=int,
    )


def constraint_spans(problem: DemProblem) -> np.ndarray:
    """Span of every detector constraint under the current mechanism order.

    The span (last site minus first site) counts the SWAP tensors the
    constraint drags along the chain, which is what governs the bond dimension
    a finite-``chi_max`` contraction needs. Ordering the mechanisms to shrink
    these is the central performance lever of the DEM decoder.
    """
    return np.array(
        [max(row) - min(row) for row in problem.detector_rows if len(row) >= 2],
        dtype=int,
    )


def order_mechanisms(
    problem: DemProblem, strategy: str = "bandwidth"
) -> Tuple[DemProblem, np.ndarray]:
    """Reorder the mechanisms to shrink the constraint spans.

    Returns the reordered problem and ``perm``, where ``perm[i]`` is the
    original mechanism index placed at chain position ``i``. The class
    posterior is invariant under this relabeling (pinned by
    ``test_ordering_leaves_the_posterior_invariant``); only the contraction
    cost changes.

    Strategies
    ----------
    "natural"
        Keep stim's order (a no-op, for comparison).
    "bandwidth"
        Reverse Cuthill-McKee on the mechanism interaction graph, where two
        mechanisms interact when some detector or observable contains both --
        the same bandwidth minimizer the code decoders use for qubit order.

    Measured caveat: on circuit-level DEMs, RCM makes the spans WORSE (at
    surface d=3, 9 rounds: mean span 111 natural vs 415 reordered). The
    high-degree detectors form cliques the minimizer cannot break, while
    stim's native order is time-ordered and already sits near the
    time-locality floor of roughly one round's worth of mechanisms per
    constraint. Bandwidth ordering is kept for DEMs without time structure
    (code-capacity models), where it reduces to the qubit ordering that works
    for the code decoders.
    """
    if strategy == "natural":
        return problem, np.arange(problem.num_mechanisms)
    if strategy != "bandwidth":
        raise ValueError(f"Unknown ordering strategy {strategy!r}.")
    if problem.num_mechanisms == 0:
        # Nothing to order; reverse Cuthill-McKee rejects an empty graph.
        return problem, np.arange(0)

    rows = [row for row in problem.detector_rows + problem.observable_rows if row]
    incidence = np.zeros((len(rows), problem.num_mechanisms), dtype=np.uint8)
    for r, row in enumerate(rows):
        incidence[r, row] = 1
    perm = optimize_qubit_order(incidence)

    inverse = np.argsort(perm)
    reordered = DemProblem(
        probs=[problem.probs[perm[i]] for i in range(problem.num_mechanisms)],
        detector_rows=[
            sorted(int(inverse[m]) for m in row) for row in problem.detector_rows
        ],
        observable_rows=[
            sorted(int(inverse[m]) for m in row) for row in problem.observable_rows
        ],
        num_detectors=problem.num_detectors,
        num_observables=problem.num_observables,
    )
    return reordered, perm
