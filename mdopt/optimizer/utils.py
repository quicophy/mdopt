"""
This module contains different combinatorial optimization utilities.

First, we define the tensors which represent logical operation.
We use the following tensors: IDENTITY, COPY, XOR, SWAP.

According to our convention, each tensor has legs (vL, vR, pU, pD),
where v stands for "virtual", p -- for "physical",
and L, R, U, D stand for "left", "right", "up", "down".
"""

from typing import List, Optional, cast
import numpy as np
from tqdm import tqdm
from matrex import msro
from numpy.random import Generator

from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.utils import find_orth_center
from mdopt.utils.utils import mpo_to_matrix
from mdopt.contractor.contractor import mps_mpo_contract

IDENTITY = np.eye(2, dtype=float).reshape((1, 1, 2, 2))

COPY_LEFT = np.fromfunction(
    lambda i, j, k, l: np.eye(2)[j, l], (1, 2, 2, 2), dtype=int
).astype(float, copy=False)

XOR_BULK = np.fromfunction(
    lambda i, j, k, l: (i ^ j ^ k ^ 1) * np.eye(2)[k, l], (2, 2, 2, 2), dtype=int
).astype(float, copy=False)

XOR_LEFT = np.fromfunction(
    lambda i, j, k, l: np.eye(2)[j, k] * np.eye(2)[k, l], (1, 2, 2, 2), dtype=int
).astype(float, copy=False)

XOR_RIGHT = np.fromfunction(
    lambda i, j, k, l: np.eye(2)[i, k] * np.eye(2)[k, l], (2, 1, 2, 2), dtype=int
).astype(float, copy=False)

SWAP = np.fromfunction(
    lambda i, j, k, l: np.eye(2)[i, j] * np.eye(2)[k, l], (2, 2, 2, 2), dtype=int
).astype(float, copy=False)


def optimize_qubit_order(H: np.ndarray) -> np.ndarray:
    """
    Given a binary parity check matrix ``H`` of shape
    ``(num_checks, num_qubits)``, returns a permutation ``perm`` that
    minimizes the bandwidth of ``H`` by reordering its columns (qubits).

    ``perm[i]`` is the original qubit index placed at MPS position ``i``
    after reordering.  The inverse ``inv_perm = np.argsort(perm)`` gives
    the new MPS position of original qubit ``j``.

    The algorithm builds the qubit interaction graph (nodes = qubits,
    undirected edge ``(i, j)`` whenever qubits ``i`` and ``j`` appear
    together in at least one check) and applies the Reverse
    Cuthill-McKee (RCM) algorithm to find a low-bandwidth ordering.

    Parameters
    ----------
    H : np.ndarray
        Binary parity check matrix of shape ``(num_checks, num_qubits)``.

    Returns
    -------
    perm : np.ndarray
        Permutation array of length ``num_qubits``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    A = ((H.T @ H) > 0).astype(np.uint8)
    np.fill_diagonal(A, 0)
    perm = reverse_cuthill_mckee(csr_matrix(A), symmetric_mode=True)
    return perm


def parity(bitstring: str, indices: list[int]) -> int:
    """Returns the parity of the bits at the given indices in the bitstring."""
    return sum(int(bitstring[i]) for i in indices) % 2


def random_constraints(num_bits: int, constraint_size: int, rng: Generator) -> dict:
    """
    Generate random XOR and SWAP site constraints for a bitstring.

    Parameters
    ----------
    num_bits : int
        The total number of bits in the bitstring. Must be at least 3.
    constraint_size : int
        The maximum possible length of the constraint, i.e., the number of XOR sites.
    rng : numpy.random.Generator
        A NumPy random number generator instance for reproducibility.

    Returns
    -------
    dict
        A dictionary containing the following keys:

        - 'xor_left_sites': list of int
            List with one index selected as the left XOR site.
        - 'xor_bulk_sites': list of int
            List of indices selected as bulk XOR sites, between left and right.
        - 'xor_right_sites': list of int
            List with one index selected as the right XOR site.
        - 'swap_sites': list of int
            List of indices between left and right XOR sites, excluding the bulk XOR sites.
        - 'all_constrained_bits': list of int
            Sorted list of all selected XOR site indices (left, bulk, right).

    Raises
    ------
    ValueError
        If `num_bits` is less than 3.
    """
    if num_bits < 3:
        raise ValueError(
            "Bitstring must have at least 3 bits for meaningful XOR and swap sites."
        )

    indices = sorted(rng.choice(num_bits, size=constraint_size, replace=False))  # type: ignore
    xor_left_sites = [indices[0]]
    xor_right_sites = [indices[-1]]
    xor_bulk_sites = indices[1:-1]
    swap_sites = [
        i for i in range(indices[0] + 1, indices[-1]) if i not in xor_bulk_sites
    ]

    return {
        "xor_left_sites": xor_left_sites,
        "xor_bulk_sites": xor_bulk_sites,
        "xor_right_sites": xor_right_sites,
        "swap_sites": swap_sites,
        "all_constrained_bits": indices,
    }


class ConstraintString:
    """
    Class for storing a string of logical constraints in the
    Matrix Product Operator format.
    Logical constraints are to be passed in the form of 4-dimensional MPO tensors.

    Attributes
    ----------
    constraints :  List[np.ndarray]
        A list of logical constraints of which the string consists.
    sites : List[List[int]]
        Each list inside corresponds to a constraint from the `constraints` list,
        and contains the sites to which each constraint is applied.
        For example, [[3, 5], [2, 4, 6], ...] means applying
        `constraints[0]` to sites 3 and 5, `constraints[1]` to sites 2, 4, 6 etc.

    Raises
    ------
    ValueError
        - Empty list of constraints.
        - Empty list of sites.
        - The ``sites`` list is longer than the ``constraints`` list.
        - Non-unique sites in the `sites` list.
        - The list of sites has gaps, indicating breaks in the constraints string.
    """

    def __init__(self, constraints: List[np.ndarray], sites: List[List[int]]) -> None:
        self.constraints = [np.asarray(constraint) for constraint in constraints]
        self.sites = sites

        if not self.constraints:
            raise ValueError("Empty list of constraints passed.")
        if not self.sites:
            raise ValueError("Empty list of sites passed.")
        if len(self.sites) != len(self.constraints):
            raise ValueError(
                f"We have {len(self.constraints)} constraint(s) in the constraints list, "
                f"but {len(self.sites)} constraint(s) assumed by the sites list."
            )

        seen = set()
        flat_sites = self.flat()
        uniq = [site for site in flat_sites if site not in seen and not seen.add(site)]  # type: ignore
        if uniq != flat_sites:
            raise ValueError("Non-unique sites encountered in the list.")

        fs = self.flat(sort=True)
        if fs != list(range(min(fs), max(fs) + 1)):
            raise ValueError("The string should not have breaks in it.")

    def __getitem__(self, index: int) -> tuple:
        """Returns (sites, mpo_tensor) pair at `index`."""
        return self.sites[index], self.constraints[index]

    def flat(self, sort: bool = False) -> List[int]:
        """Returns a flattened list of sites."""
        out = [item for sites in self.sites for item in sites]
        return sorted(out) if sort else out

    def span(self) -> int:
        """Returns the span (length) of the constraint string."""
        fs = self.flat()
        return max(fs) - min(fs) + 1

    def mpo(self) -> List[np.ndarray]:
        """Returns an MPO list aligned to the span of this constraint string."""
        fs = self.flat()
        base = min(fs)
        mpo = [np.array(None, dtype=float) for _ in range(self.span())]
        for idx, sites_sites in enumerate(self.sites):
            for site in sites_sites:
                mpo[site - base] = self.constraints[idx]
        return mpo


class TruncationCertificate:
    """A rigorous bound on the relative truncation error of a constraint sweep.

    Every check-constraint MPO is a partial permutation in the computational
    basis (operator norm one), and so is every partial application of one
    inside the zip-up contraction, so the part an SVD discards propagates to
    the end of the sweep with at most its own norm. A logical constraint has
    operator norm ``sqrt(2)`` (its copy tensor sends ``|+>`` to ``sqrt(2)``
    times a basis state), and every constraint's gain ``g_u`` is recorded
    with it. Writing ``d_t`` for the sum of discarded norms inside constraint
    ``t`` (the move of the orthogonality center to its start included, in the
    scale of that constraint's input) and ``r_u`` for the factor the state is
    renormalized by after constraint ``u``, the final state satisfies

        ``||psi - psi_exact|| / ||psi|| <= sum_t d_t * G_t * S_{t-1} / (S_T * ||psi||)``

    with ``S_t = r_1 * ... * r_t`` and ``G_t = g_t * ... * g_T``, since a
    discarded component passes through at most the rest of its own
    constraint and all later ones. The bound is loose, by about two orders
    of magnitude on the [[18,2]] bivariate bicycle code, but it ranks shots:
    there no verdict with a bound below 100 was wrong, the wrong rate climbs
    monotonically above that, and every globally sign-flipped posterior sat
    above 1e8, because its survival factor ``S_T`` had underflowed. The
    survival factor is the mechanism behind the sign flips: truncation is
    accurate relative to the mass present at the time, and the checks keep
    only ``S_T`` of it.

    The operator norm is the worst case the bound needs; ``survival`` instead
    divides out the factor by which each step would have changed the norm of
    the exact state, its mass gain. For a check and a logical constraint the
    two coincide (1 and ``sqrt(2)``); a bias operator has an operator norm
    above one but maps a unit-norm basis pair to a unit-norm prior, so its
    mass gain is one and it leaves ``survival`` alone.
    """

    def __init__(self) -> None:
        self.log_scale = 0.0
        self.log_gain = 0.0
        self.log_mass_gain = 0.0
        self.final_norm = 1.0
        self.terms: List[tuple[float, float, float]] = []
        self.bound = 0.0
        # The most negative class mass as a fraction of the peak, set by a
        # readout that clamps negatives (decode_dem); 0 for an exact
        # contraction, since every exact class mass is non-negative.
        self.negativity = 0.0

    def record(
        self,
        discarded: float,
        renormalization: float,
        gain: float = 1.0,
        mass_gain: Optional[float] = None,
    ) -> None:
        """One constraint: what its truncations discarded, the factor the
        state was renormalized by afterwards (1 when it was not), the
        operator norm of its MPO and, when it differs from that norm, the
        factor by which the exact operator changes the norm of the state
        (``survival`` divides this one out; it defaults to ``gain``)."""
        self.terms.append((float(discarded), self.log_scale, self.log_gain))
        self.log_scale += (
            float(np.log(renormalization)) if renormalization > 0 else -np.inf
        )
        self.log_gain += float(np.log(gain))
        self.log_mass_gain += float(np.log(gain if mass_gain is None else mass_gain))

    @property
    def survival(self) -> float:
        """The mass fraction the constraints kept, in amplitude units: the
        product of the renormalization factors and the final norm, with the
        mass gains of the constraints divided out. It is the same number
        whether or not the sweep renormalized."""
        return float(np.exp(self.log_scale - self.log_mass_gain) * self.final_norm)

    @property
    def discarded(self) -> float:
        """The sum of all discarded norms, in the original scale."""
        return float(sum(d * np.exp(log_s) for d, log_s, _ in self.terms))

    def finalize(self, final_norm: float = 1.0) -> float:
        """Evaluate and store the bound for a final state of norm ``final_norm``."""
        self.final_norm = float(final_norm)
        log_final = self.log_scale + (np.log(final_norm) if final_norm > 0 else -np.inf)
        with np.errstate(over="ignore", invalid="ignore"):
            self.bound = float(
                sum(
                    d * np.exp(self.log_gain - log_g + log_s - log_final)
                    for d, log_s, log_g in self.terms
                    if d > 0
                )
            )
        if np.isnan(self.bound):
            self.bound = float("inf")
        return self.bound


def _product_state_orth_center(mps: CanonicalMPS, tolerance: float = 1e-12) -> int:
    """The orthogonality center :func:`find_orth_center` assigns a product state.

    For a chain of bond dimension 1 every tensor is (1, d, 1); it is a left
    and a right isometry exactly when its vector has unit norm. The scan then
    reports the non-isometric sites as centers and the convention keeps the
    first of them, or site 0 when all sites are isometric; a biased chain
    whose first sites are still normalized (the decoders' logical prefix)
    therefore gets its first biased site, not site 0.
    """
    for site, tensor in enumerate(mps.tensors):
        gram = float(np.vdot(tensor, tensor).real)
        if not np.isclose(gram, 1.0, atol=tolerance, rtol=0.0):
            return site
    return 0


def apply_constraints(
    mps: CanonicalMPS,
    strings: List[List[List[int]]],
    logical_tensors: List[np.ndarray],
    chi_max: int = int(1e4),
    cut: float = float(1e-17),
    renormalize: bool = True,
    strategy: str = "Naive",
    silent: bool = False,
    dense: bool = False,
    return_entropies_and_bond_dims: bool = False,
    certificate: Optional[TruncationCertificate] = None,
    gain: float = 1.0,
) -> CanonicalMPS | np.ndarray:
    """
    This function applies logical constraints to an MPS.

    Parameters
    ----------
    mps : CanonicalMPS
        The MPS to which the logical constraints are being applied.
    strings : List[List[List[int]]]
        The list of arguments for :class:`ConstraintString`, one
        ``[xor_left, xor_bulk, swap, xor_right]`` site list per constraint.
    logical_tensors : List[np.ndarray]
        List of logical tensors for :class:`ConstraintString`.
    chi_max : int
        Maximum bond dimension to keep in the contractor.
    cut : float
        The lower boundary of the spectrum in the contractor.
        All singular values below this value are truncated.
    renormalize : bool
        Whether to renormalize the orthogonality center after each constraint application.
    result_to_explicit : bool
        Whether to transform the resulting MPS into the Explicit form.
    strategy : str
        The contractor strategy. Available options are "Optimized" and "Naive".
    silent : bool
        Whether to show the progress bar or not.
    return_entropies_and_bond_dims : bool
        Whether to return the entanglement entropies and bond dimensions at each bond.
    dense : bool
        Whether to perform the calculations in the dense form.
        To be used only for small systems (<= 20 sites).
    certificate : Optional[TruncationCertificate]
        When given, every constraint's discarded norms and renormalization
        factor are recorded in it (see :class:`TruncationCertificate`).
    gain : float
        The operator norm of every constraint MPO in ``strings``, recorded
        with the certificate: one for the check constraints (partial
        permutations) and ``sqrt(2)`` for the logical constraints, which the
        decoders pass.

    Returns
    -------
    mps : CanonicalMPS
        The resulting MPS.
    entropies, bond_dims : List[float], List[int]
        The list of entanglement entropies at each bond.
        Returned only if ``return_entropies_and_bond_dims`` is set to ``True``.
    """

    if strategy not in ("Naive", "Optimized"):
        raise ValueError(
            f"Unknown strategy {strategy!r}; expected 'Naive' or 'Optimized'."
        )

    entropies = []
    bond_dims = []

    if strategy == "Optimized":
        # Using matrix front minimization technique to optimize the order
        # in which to apply the checks.
        mpo_location_matrix = np.zeros((len(strings), mps.num_sites), dtype=int)
        for row_idx, sublist in enumerate(strings):
            for subsublist in sublist:
                for index in subsublist:  # type: ignore
                    mpo_location_matrix[row_idx, index] = 1
        optimized_order = msro(mpo_location_matrix)
        strings = [strings[index] for index in optimized_order]

    if dense:
        mps_dense = mps.dense(flatten=True)

    # One private copy up front, then every zip-up and move works in place:
    # the contractor used to deep-copy the whole chain once per constraint,
    # which on a 1700-site DEM chain was a measurable share of the decode.
    if strings and not dense:
        mps = mps.copy()

    for string in tqdm(strings, disable=silent):
        string = ConstraintString(logical_tensors, string)
        mpo = string.mpo()

        start_site = min(string.flat())
        discarded: List[float] = []
        if dense:
            # Pad MPO with identities to full length and apply as a matrix
            identities_l = [IDENTITY for _ in range(start_site)]
            identities_r = [IDENTITY for _ in range(len(mps) - len(mpo) - start_site)]
            full_mpo = identities_l + mpo + identities_r
            mpo_dense = mpo_to_matrix(full_mpo, interlace=False, group=True)
            mps_dense = mpo_dense @ mps_dense  # type: ignore[has-type]
            if certificate is not None:
                # Exact and unnormalized, but the gain still scales what
                # earlier sweeps discarded and what ``survival`` divides out.
                certificate.record(0.0, 1.0, gain)
            continue

        # Ensure orthogonality center is set and moved once per string
        if mps.orth_center is None and all(d == 1 for d in mps.bond_dimensions):
            # A product state (every bond of dimension 1) needs no isometry
            # scan: a (1, d, 1) tensor is an isometry exactly when its
            # vector has unit norm, so the scan's answer -- the first
            # non-isometric site, or site 0 when every site is isometric --
            # comes from the site norms alone. Same tolerance as the scan.
            mps.orth_center = _product_state_orth_center(mps)
        if mps.orth_center is None:
            orth_centers, flags_left, flags_right = find_orth_center(
                mps, return_orth_flags=True
            )

            # Managing possible issues with multiple orthogonality centers
            # arising if we do not renormalize while contracting.
            if orth_centers and len(orth_centers) == 1:
                mps.orth_center = orth_centers[0]
            # Convention.
            if all(flags_left) and all(flags_right):
                mps.orth_center = 0
            elif flags_left in ([True] + [False] * (mps.num_sites - 1)):
                if flags_right == [not flag for flag in flags_left]:
                    mps.orth_center = mps.num_sites - 1
            elif flags_left in ([True] * (mps.num_sites - 1) + [False]):
                if flags_right == [not flag for flag in flags_left]:
                    mps.orth_center = 0
            elif all(flags_right):
                mps.orth_center = 0
            elif all(flags_left):
                mps.orth_center = mps.num_sites - 1

            if isinstance(orth_centers, list) and len(orth_centers) > 1:
                mps.orth_center = orth_centers[0]

            mps = mps.move_orth_center(
                final_pos=start_site,
                renormalize=False,
                return_singular_values=False,
                inplace=True,
                discarded_norms=discarded if certificate is not None else None,
            )  # type: ignore

        # Contract MPO string into the MPS (uses contractor that preserves dtype & avoids diag())
        mps = mps_mpo_contract(  # type: ignore
            mps=mps,
            mpo=mpo,
            start_site=start_site,
            chi_max=chi_max,
            cut=cut,
            renormalize=False,
            inplace=True,
            discarded_norms=discarded if certificate is not None else None,
        )
        renormalization = 1.0
        if renormalize:
            orth_center_index = int(mps.orth_center)  # type: ignore
            center = mps.tensors[orth_center_index]
            norm = np.linalg.norm(center)
            if norm > 0:
                mps.tensors[orth_center_index] = center / norm
            renormalization = float(norm)
        if certificate is not None:
            certificate.record(sum(discarded), renormalization, gain)

        if return_entropies_and_bond_dims:
            # Use a copy to avoid mutating `mps` during entropy computation
            mps_copy = mps.copy()
            entropies.append(mps_copy.entanglement_entropy())
            bond_dims.append(mps_copy.bond_dimensions)

    if dense:
        return mps_dense

    if return_entropies_and_bond_dims:
        return cast(CanonicalMPS, mps), entropies, bond_dims  # type: ignore

    return cast(CanonicalMPS, mps)
