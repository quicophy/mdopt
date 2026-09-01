"""Decode stim detector error models with an MPS marginaliser.

A detector error model (DEM) is a purely classical decoding problem: each
error mechanism is a bit ``m_i`` that fires independently with prior ``p_i``,
each detector is a parity check ``<D_d, m> = s_d`` over the mechanisms that
flip it, and each logical observable is a parity readout ``<O_j, m>``. Maximum
likelihood decoding asks for the observable-flip pattern whose total
probability mass, summed over all mechanism sets consistent with the observed
syndrome, is largest.

The tensor-network realisation flattens the mechanisms onto one MPS chain --
one site per mechanism, plus one logical site per observable at the front --
and reuses the parity-check machinery of the code decoders: detectors become
XOR constraints, observables become COPY/XOR readouts, and the mechanisms are
marginalised away. Unlike the code decoders, the prior is applied as *linear*
weights, so the readout is the exact class probability mass ``sum(w)`` rather
than the amplitude form ``sum(sqrt(w))``; MAP against the literature is the
target here. Contraction is exact at large ``chi_max``; the cost of finite
``chi_max`` is governed by the spans of the detector constraints after the
mechanisms are laid out on the chain, which makes mechanism ordering the
central performance question (see arXiv:2310.10722 for the 3D-tensor-network
view of the same problem).
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import stim

from mdopt.mps.utils import create_custom_product_state
from mdopt.optimiser.utils import (
    COPY_LEFT,
    SWAP,
    XOR_BULK,
    XOR_LEFT,
    XOR_RIGHT,
)
from mdopt.optimiser.utils import apply_constraints


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
        detectors, observables = set(), set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.add(target.val)
            elif target.is_logical_observable_id():
                observables.add(target.val)
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
    for (detectors, observables), probability in merged.items():
        index = len(probs)
        probs.append(probability)
        for det in detectors:
            detector_rows[det].append(index)
        for obs in observables:
            observable_rows[obs].append(index)
    return DemProblem(
        probs=probs,
        detector_rows=[sorted(row) for row in detector_rows],
        observable_rows=[sorted(row) for row in observable_rows],
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
    )


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
    matrix[:, num_mech] = np.asarray(syndrome, dtype=int) % 2

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
    """[XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT] site lists for one detector."""
    sites = [m + offset for m in mechanisms]
    return [
        [sites[0]],
        sites[1:-1],
        [s for s in range(sites[0] + 1, sites[-1]) if s not in sites[1:-1]],
        [sites[-1]],
    ]


def _logical_sites(
    mechanisms: List[int], logical_site: int, offset: int
) -> List[List[int]]:
    """[COPY_LEFT, XOR_BULK, SWAP, XOR_RIGHT] for one observable readout."""
    sites = [m + offset for m in mechanisms]
    return [
        [logical_site],
        sites[:-1],
        [s for s in range(logical_site + 1, sites[-1]) if s not in sites[:-1]],
        [sites[-1]],
    ]


def decode_dem(
    problem: DemProblem,
    syndrome: np.ndarray,
    chi_max: int = int(1e4),
    cut: float = float(1e-17),
    renormalise: bool = True,
    silent: bool = True,
    tolerance: float = float(1e-12),
    representative: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood observable prediction for one syndrome.

    Returns
    -------
    (class_masses, predicted_flips)
        ``class_masses[c]`` is proportional to the total probability of the
        mechanism sets whose observable pattern is ``c`` (bit ``j`` of ``c``
        is observable ``j``), and ``predicted_flips`` is the argmax pattern as
        a bit array.
    """
    num_obs = problem.num_observables
    num_mech = problem.num_mechanisms
    offset = num_obs

    for det, row in enumerate(problem.detector_rows):
        if len(row) == 0 and int(syndrome[det]) % 2 == 1:
            raise ValueError(f"Detector {det} fired but no mechanism flips it.")
        if len(row) == 1:
            raise NotImplementedError(
                "A detector touched by a single mechanism pins that mechanism "
                "outright; eliminate it from the model before decoding."
            )

    base = (
        solve_representative(problem, syndrome)
        if representative is None
        else np.asarray(representative, dtype=int) % 2
    )
    if np.any((_detector_parities(problem, base) - np.asarray(syndrome) % 2) % 2):
        raise ValueError("The supplied representative does not match the syndrome.")

    # Change of variables: the XOR constraints project onto EVEN parity, i.e.
    # the zero-syndrome sector. Writing every consistent mechanism set as
    # m = base XOR f puts the free variable f exactly in that sector, so the
    # chain carries f (initialised all-zero) and each site is weighted by the
    # prior of m_i = base_i XOR f_i:
    #     base_i = 0 ->  (1-p_i, p_i)  for f_i = (0, 1)
    #     base_i = 1 ->  (p_i, 1-p_i)
    # The observable labels read off f; the representative's own observable
    # pattern is XORed back in at the end.
    state_string = "+" * num_obs + "0" * num_mech
    mps = create_custom_product_state(
        string=state_string, tolerance=tolerance, form="Right-canonical"
    )
    for i, probability in enumerate(problem.probs):
        q = probability if base[i] == 0 else 1.0 - probability
        weight_matrix = np.array([[1.0 - q, q], [q, 1.0 - q]])
        mps.tensors[offset + i] = np.einsum(
            "ab, ibj -> iaj", weight_matrix, mps.tensors[offset + i]
        )

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

    if observable_strings:
        mps = apply_constraints(
            mps,
            observable_strings,
            logicals_tensors,
            chi_max=chi_max,
            cut=cut,
            renormalise=renormalise,
            silent=silent,
            strategy="Optimised",
        )
    mps = apply_constraints(
        mps,
        detector_strings,
        constraints_tensors,
        chi_max=chi_max,
        cut=cut,
        renormalise=renormalise,
        silent=silent,
        strategy="Optimised",
    )

    logical_mps = mps.marginal(
        sites_to_marginalise=list(range(num_obs, num_obs + num_mech)),
        renormalise=renormalise,
    ).reverse()
    dense = np.real(np.asarray(logical_mps.dense(flatten=True, renormalise=False)))
    dense = np.abs(dense)

    if not np.isfinite(dense).all() or float(np.max(dense)) == 0.0:
        raise ArithmeticError(f"The class-mass vector collapsed at chi_max={chi_max}.")

    # Relabel from f-classes to m-classes: XOR in the representative's own
    # observable pattern.
    obs_offset = 0
    for j, row in enumerate(problem.observable_rows):
        if row and int(np.sum(base[row]) % 2):
            obs_offset |= 1 << j
    relabelled = np.empty_like(dense)
    for index in range(dense.size):
        relabelled[index ^ obs_offset] = dense[index]

    best = int(np.argmax(relabelled))
    # After reverse(), the first MPS logical site (observable 0) is the LAST
    # dense index bit, so bit j of the flat index is observable j.
    flips = np.array([(best >> j) & 1 for j in range(num_obs)], dtype=int)
    return relabelled, flips


def _detector_parities(problem: DemProblem, mechanisms: np.ndarray) -> np.ndarray:
    return np.array(
        [
            int(np.sum(mechanisms[row]) % 2) if row else 0
            for row in problem.detector_rows
        ],
        dtype=int,
    )
