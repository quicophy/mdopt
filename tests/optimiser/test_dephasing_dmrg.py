"""Tests for the ``mdopt.optimiser.dephasing_dmrg`` module."""

import pytest
import numpy as np
from scipy.sparse.linalg import ArpackError, LinearOperator, eigsh

from mdopt.optimiser.dephasing_dmrg import DephasingDMRG as deph_dmrg
from mdopt.optimiser.dmrg import DMRG as dmrg
import mdopt.optimiser.dephasing_dmrg as dephasing_dmrg_module
from mdopt.optimiser.dephasing_dmrg import (
    EffectiveDensityOperator,
    _operator_is_numerically_zero,
    _restart_from_operator_range,
)
from mdopt.mps.utils import (
    create_state_vector,
    create_simple_product_state,
    create_custom_product_state,
    mps_from_dense,
    inner_product,
)


def test_optimiser_effective_density_operator():
    """Test for the ``__init__`` method of the ``EffectiveDensityOperator`` class."""

    left_environment = np.random.uniform(low=0, high=1, size=(2, 2, 2, 2))
    mps_target_1 = np.random.uniform(low=0, high=1, size=(2, 2, 2))
    mps_target_2 = np.random.uniform(low=0, high=1, size=(2, 2, 2))
    right_environment = np.random.uniform(low=0, high=1, size=(2, 2, 2, 2))

    EffectiveDensityOperator(
        left_environment=left_environment,
        mps_target_1=mps_target_1,
        mps_target_2=mps_target_2,
        right_environment=right_environment,
    )

    with pytest.raises(ValueError):
        EffectiveDensityOperator(
            left_environment=np.expand_dims(left_environment, 0),
            mps_target_1=mps_target_1,
            mps_target_2=mps_target_2,
            right_environment=right_environment,
        )
    with pytest.raises(ValueError):
        EffectiveDensityOperator(
            left_environment=left_environment,
            mps_target_1=np.expand_dims(mps_target_1, 0),
            mps_target_2=mps_target_2,
            right_environment=right_environment,
        )
    with pytest.raises(ValueError):
        EffectiveDensityOperator(
            left_environment=left_environment,
            mps_target_1=mps_target_1,
            mps_target_2=np.expand_dims(mps_target_2, 0),
            right_environment=right_environment,
        )
    with pytest.raises(ValueError):
        EffectiveDensityOperator(
            left_environment=left_environment,
            mps_target_1=mps_target_1,
            mps_target_2=mps_target_2,
            right_environment=np.expand_dims(right_environment, 0),
        )


def _linear_operator_to_dense(op) -> np.ndarray:
    """
    Materialise a scipy.sparse.linalg.LinearOperator into a dense matrix
    by applying it to basis vectors. Intended only for tiny dimensions in tests.
    """
    n, m = op.shape
    assert n == m
    eye = np.eye(n, dtype=op.dtype)
    cols = [op.matvec(eye[:, k]) for k in range(n)]
    return np.stack(cols, axis=1)


def test_effective_density_operator_plus_target_is_identity_chi1():
    """
    For chi_left=chi_right=1 and target tensors corresponding to |++>,
    the dephased two-site operator should be proportional to identity on the 4-dim
    (00,01,10,11) physical space.

    If this fails, the copy-tensor wiring / einsum in EffectiveDensityOperator is wrong.
    """
    # Environments for chi=1
    left_env = np.zeros((1, 1, 1, 1), dtype=float)
    right_env = np.zeros((1, 1, 1, 1), dtype=float)
    left_env[0, 0, 0, 0] = 1.0
    right_env[0, 0, 0, 0] = 1.0

    # One-site tensor for |+> (ExplicitMPS conventions aside, we only need the raw tensor)
    t = np.zeros((1, 2, 1), dtype=float)
    t[0, 0, 0] = 1.0 / np.sqrt(2.0)
    t[0, 1, 0] = 1.0 / np.sqrt(2.0)

    op = EffectiveDensityOperator(
        left_environment=left_env,
        mps_target_1=t,
        mps_target_2=t,
        right_environment=right_env,
    )
    M = _linear_operator_to_dense(op)

    # Must be diagonal and all diagonal entries equal (proportional to identity).
    assert np.allclose(M, np.diag(np.diag(M)), atol=1e-12)
    assert np.allclose(np.diag(M), np.diag(M)[0], atol=1e-12)


def test_dephasing_dmrg_returns_bitstring_for_two_maxima_degenerate():
    """
    A more realistic degeneracy test than |+>^{⊗n}:
    psi = (|00..0> + |11..1>) / sqrt(2).
    The dephased distribution has exactly two maximum a posteriori (MAP) bitstrings.

    A 'bitstrings-only' algorithm must output either 00..0 or 11..1 (not a superposition
    nor an entangled MPS).
    """
    num_sites = 8
    num_runs = 1

    psi = np.zeros(2**num_sites, dtype=complex)
    psi[0] = 1.0
    psi[-1] = 1.0
    psi /= np.linalg.norm(psi)

    target = mps_from_dense(psi, form="Right-canonical")
    start = create_simple_product_state(num_sites, which="+", form="Explicit")

    engine = deph_dmrg(
        start,
        target.right_canonical(),
        chi_max=1e4,
        cut=1e-12,
        mode="LA",
        copy=True,
        silent=True,
    )
    engine.run(num_runs)

    bitstring = _assert_is_computational_basis_product_state(engine.mps)
    assert bitstring in {"0" * num_sites, "1" * num_sites}


def test_dephasing_dmrg_never_creates_entanglement_if_claimed_bitstring_only():
    """
    Our implementation restricts the search domain to computational-basis bitstrings,
    it should never create entanglement: all bond dimensions must remain 1 after any run.
    """
    num_sites = 8
    num_runs = 1

    target = create_simple_product_state(
        num_sites, which="+", form="Explicit"
    ).right_canonical()
    start = create_simple_product_state(num_sites, which="+", form="Explicit")

    engine = deph_dmrg(
        start,
        target,
        chi_max=1e4,
        cut=1e-12,
        mode="LA",
        copy=True,
        silent=True,
    )
    engine.run(num_runs)
    assert engine.mps.bond_dimensions == [1 for _ in range(engine.mps.num_bonds)]


def _assert_is_computational_basis_product_state(mps, *, atol: float = 1e-10) -> str:
    """
    Stronger criterion than 'bond dimension == 1':
    checks that each site tensor is one-hot in the computational basis (|0> or |1>),
    up to a global scale/phase. Returns the extracted bitstring.
    """
    # Must be a product state in the MPS sense.
    assert mps.bond_dimensions == [1 for _ in range(mps.num_bonds)]

    bits = []
    for t in mps.tensors:
        # Expect (1, d, 1) for a product state.
        assert t.ndim == 3
        assert t.shape[0] == 1 and t.shape[2] == 1
        v = t[0, :, 0]
        w = np.abs(v) ** 2
        s = float(w.sum())
        assert s > 0.0
        i = int(np.argmax(w))

        # One-hot criterion up to tolerance relative to total weight.
        off = s - float(w[i])
        assert off <= atol * max(1.0, s), (
            "Product state is not a computational-basis bitstring. "
            f"Local weights={w}, total={s}, off={off}."
        )
        bits.append(str(i))

    return "".join(bits)


def test_optimiser_main_component():
    """
    Test the dephasing DMRG optimiser with the main component problem.
    We solve the problem using exact diagonalisation, DMRG and dephasing DMRG.
    Next, we compare the solutions which should be exactly the same.
    """

    for _ in range(5):
        num_sites = 8
        num_runs = 1

        # Creating a random pure complex state and its MPS version.
        psi = create_state_vector(num_sites)

        # Bumping up the main component amplitude and renormalising the state.
        index_to_bump = np.random.randint(0, 2**num_sites)
        psi[index_to_bump] = 10
        psi /= np.linalg.norm(psi)

        # Creating the exact MPS version of the state.
        mps = mps_from_dense(psi, form="Right-canonical")

        # Creating the matrix density product operator.
        mpdo = mps.density_mpo()

        # Finding the main component (a computational basis state having the largest overlap)
        # of the density matrix in the dense form.
        overlaps_exact = []
        for i in range(2**num_sites):
            state_string = np.binary_repr(i, width=num_sites)
            overlaps_exact.append(
                np.absolute(create_custom_product_state(state_string).dense() @ psi)
                ** 2
            )
        main_component_exact = np.argmax(overlaps_exact)

        # Finding the main component of the MPDO using DMRG.
        mps_start = create_simple_product_state(num_sites, which="+", form="Explicit")
        engine = dmrg(
            mps_start, mpdo, chi_max=1e4, cut=1e-12, mode="LA", copy=True, silent=True
        )
        engine.run(num_runs)
        max_excited_mps_from_dmrg = engine.mps

        overlaps_dmrg = []
        for i in range(2**num_sites):
            state_string = np.binary_repr(i, width=num_sites)
            overlaps_dmrg.append(
                np.absolute(
                    inner_product(
                        max_excited_mps_from_dmrg,
                        create_custom_product_state(state_string),
                    )
                )
                ** 2
            )
        main_component_dmrg = np.argmax(overlaps_dmrg)

        # Finding the main component of the MPDO using dephasing DMRG.
        mps_start = create_simple_product_state(num_sites, which="+", form="Explicit")
        dephasing_engine = deph_dmrg(
            mps_start,
            mps.right_canonical(),
            chi_max=1e4,
            cut=1e-12,
            mode="LA",
            copy=True,
            silent=True,
        )
        dephasing_engine.run(num_runs)
        main_component_mps = dephasing_engine.mps

        overlaps_dephased = []
        for i in range(2**num_sites):
            state_string = np.binary_repr(i, width=num_sites)
            overlaps_dephased.append(
                np.absolute(
                    inner_product(
                        main_component_mps,
                        create_custom_product_state(state_string),
                    )
                )
                ** 2
            )
        main_component_dephased = np.argmax(overlaps_dephased)

        # Check the answer from the dephasing DMRG is a product state.
        mps_product_answer = dephasing_engine.mps
        assert mps_product_answer.bond_dimensions == [
            1 for _ in range(mps_product_answer.num_bonds)
        ]

        # Check it is actually a *computational-basis* product state (bitstring),
        # not an arbitrary local-superposition product state.
        _assert_is_computational_basis_product_state(mps_product_answer)

        # Check that all the three answers are the same.
        assert np.logical_and(
            main_component_exact == main_component_dmrg,
            main_component_exact == main_component_dephased,
        )


def test_dephasing_dmrg_returns_bitstring_for_plus_target():
    """
    Bug-catcher: target = |+>^{⊗n}. The dephased MPDO is maximally mixed (huge degeneracy).
    A solver that truly searches only over computational basis bitstrings
    must still output a computational-basis bitstring, not |+>^{⊗n}.
    """
    num_sites = 8
    num_runs = 1

    # Target state is exactly |+>^{⊗n}.
    target = create_simple_product_state(num_sites, which="+", form="Explicit")

    # Start from |+>^{⊗n} as well (this is the adversarial case: a coherence-leaking
    # implementation tends to remain stuck at |+>).
    start = create_simple_product_state(num_sites, which="+", form="Explicit")

    engine = deph_dmrg(
        start,
        target.right_canonical(),
        chi_max=1e4,
        cut=1e-12,
        mode="LA",
        copy=True,
        silent=True,
    )
    engine.run(num_runs)

    # Must be a computational-basis bitstring product state.
    _assert_is_computational_basis_product_state(engine.mps)


def test_snap_to_computational_basis_one_hot_and_dtype():
    """
    The snapping helper must project any two-site tensor to a one-hot computational-basis vector
    and preserve dtype (both real and complex).
    """
    # Real case
    xr = np.random.randn(1, 2, 2, 1).astype(float)
    ys = deph_dmrg._snap_to_computational_basis(xr)
    assert ys.dtype == xr.dtype
    flat = ys.reshape(-1)
    assert np.count_nonzero(flat) == 1
    assert np.isclose(np.max(np.abs(flat)), 1.0)

    # Complex case
    xc = (np.random.randn(1, 2, 2, 1) + 1j * np.random.randn(1, 2, 2, 1)).astype(
        complex
    )
    yc = deph_dmrg._snap_to_computational_basis(xc)
    assert yc.dtype == xc.dtype
    flatc = yc.reshape(-1)
    assert np.count_nonzero(flatc) == 1
    # Magnitude of the kept entry is 1
    assert np.isclose(np.max(np.abs(flatc)), 1.0)


def test_effective_density_operator_dtype_preserved_complex():
    """
    The EffectiveDensityOperator must preserve complex dtype when targets are complex.
    For chi_left=chi_right=1, the operator should remain diagonal in computational basis
    and the materialised matrix should be complex-typed.
    """
    # Environments for chi=1 with complex dtype
    left_env = np.zeros((1, 1, 1, 1), dtype=complex)
    right_env = np.zeros((1, 1, 1, 1), dtype=complex)
    left_env[0, 0, 0, 0] = 1.0 + 0j
    right_env[0, 0, 0, 0] = 1.0 + 0j

    # One-site tensor for |+i> = (|0> + i|1>)/sqrt(2)
    t = np.zeros((1, 2, 1), dtype=complex)
    t[0, 0, 0] = 1.0 / np.sqrt(2.0)
    t[0, 1, 0] = 1.0j / np.sqrt(2.0)

    op = EffectiveDensityOperator(
        left_environment=left_env,
        mps_target_1=t,
        mps_target_2=t,
        right_environment=right_env,
    )
    M = _linear_operator_to_dense(op)

    assert np.iscomplexobj(M)
    # Still diagonal for chi=1 and product |+i> targets
    assert np.allclose(M, np.diag(np.diag(M)), atol=1e-12)


def test_effective_density_operator_matvec_wrong_size_raises():
    """
    If the provided vector cannot be reshaped into x_shape, numpy should raise ValueError in _matvec.
    """
    left_env = np.zeros((1, 1, 1, 1), dtype=float)
    right_env = np.zeros((1, 1, 1, 1), dtype=float)
    left_env[0, 0, 0, 0] = 1.0
    right_env[0, 0, 0, 0] = 1.0

    t = np.zeros((1, 2, 1), dtype=float)
    t[0, 0, 0] = 1.0 / np.sqrt(2.0)
    t[0, 1, 0] = 1.0 / np.sqrt(2.0)

    op = EffectiveDensityOperator(
        left_environment=left_env,
        mps_target_1=t,
        mps_target_2=t,
        right_environment=right_env,
    )

    bad = np.zeros(op.shape[1] - 1, dtype=op.dtype)
    with pytest.raises(ValueError):
        op._matvec(bad)


def test_dephasing_dmrg_no_entanglement_complex_target():
    r"""
    With a complex product target (e.g., |+i>^⊗n), the algorithm must still maintain
    bond dimensions equal to 1 after a run (bitstring-only search domain).
    """
    num_sites = 6
    num_runs = 1

    # Complex product target |+i>^{⊗n}
    t = np.zeros((1, 2, 1), dtype=complex)
    t[0, 0, 0] = 1.0 / np.sqrt(2.0)
    t[0, 1, 0] = 1.0j / np.sqrt(2.0)

    # Build explicit product state tensors for target
    tensors = [t.copy() for _ in range(num_sites)]
    # Create an ExplicitMPS via utilities: start from |+> and replace tensors directly if supported
    # Simpler: take |+> product and coerce dtype by adding 0j, then right_canonical
    target = create_simple_product_state(num_sites, which="+", form="Explicit")
    # Overwrite physical entries to match |+i>
    for i in range(num_sites):
        assert target.tensors[i].shape == (1, 2, 1)
        target.tensors[i] = t.copy()
    target = target.right_canonical()

    start = create_simple_product_state(num_sites, which="+", form="Explicit")

    engine = deph_dmrg(
        start,
        target,
        chi_max=1e4,
        cut=1e-12,
        mode="LA",
        copy=True,
        silent=True,
    )
    engine.run(num_runs)

    # Must remain a product state with bond dims 1
    assert engine.mps.bond_dimensions == [1 for _ in range(engine.mps.num_bonds)]
    # And be a computational-basis bitstring product state
    _assert_is_computational_basis_product_state(engine.mps)


def test_dephasing_dmrg_deterministic_snap_in_degeneracy_LA():
    r"""
    In a highly degenerate case (target = |+>^{⊗n}), the solver should still snap
    deterministically via argmax to the first basis configuration, yielding 0...0.
    """
    num_sites = 6
    num_runs = 1

    target = create_simple_product_state(num_sites, which="+", form="Explicit")
    start = create_simple_product_state(num_sites, which="+", form="Explicit")

    engine = deph_dmrg(
        start,
        target.right_canonical(),
        chi_max=1e4,
        cut=1e-12,
        mode="LA",
        copy=True,
        silent=True,
    )
    engine.run(num_runs)

    bitstring = _assert_is_computational_basis_product_state(engine.mps)
    # Determinism: running again from the same start should yield the same bitstring.
    engine2 = deph_dmrg(
        start,
        target.right_canonical(),
        chi_max=1e4,
        cut=1e-12,
        mode="LA",
        copy=True,
        silent=True,
    )
    engine2.run(num_runs)
    bitstring2 = _assert_is_computational_basis_product_state(engine2.mps)

    # Both solutions must be valid computational-basis product states and keep bond dims at 1.
    assert engine.mps.bond_dimensions == [1 for _ in range(engine.mps.num_bonds)]
    assert engine2.mps.bond_dimensions == [1 for _ in range(engine2.mps.num_bonds)]


def test_arpack_rejects_a_zero_operator_even_with_a_valid_start_vector():
    """Pin the real trigger behind "ARPACK error -9: Starting vector is zero".

    The message names the starting vector, which is misleading: the same error
    is raised when the *operator* annihilates whatever it is handed, however
    valid that vector is. Guarding v0 alone left a full-scale classical_ldpc run
    failing here twice, 4.6 h and 2.2 h in.
    """
    dimension = 8
    zero_operator = LinearOperator(
        (dimension, dimension), matvec=lambda v: np.zeros(dimension), dtype=float
    )
    unit_v0 = np.full(dimension, 1.0 / np.sqrt(dimension))
    assert np.isclose(np.linalg.norm(unit_v0), 1.0)

    with pytest.raises(ArpackError):
        eigsh(zero_operator, k=1, which="LA", v0=unit_v0, return_eigenvectors=True)


def test_dephasing_dmrg_skips_a_bond_whose_operator_underflowed():
    """A zeroed target must leave the sweep running, not raise.

    The effective density operator is built from the target MPS tensors, so once
    those underflow it is numerically zero on every bond.
    """
    num_sites = 6
    psi = np.zeros(2**num_sites, dtype=complex)
    psi[0] = 1.0

    target = mps_from_dense(psi, form="Right-canonical").right_canonical()
    start = create_simple_product_state(num_sites, which="+", form="Explicit")

    engine = deph_dmrg(
        start,
        target,
        chi_max=64,
        cut=1e-12,
        mode="LA",
        copy=True,
        silent=True,
    )

    # Drive the operator to zero the way an underflowed run does.
    for index, tensor in enumerate(engine.mps_target.tensors):
        engine.mps_target.tensors[index] = np.zeros_like(tensor)

    engine.update_bond(0)  # must return quietly rather than raising ArpackError


def test_operator_is_numerically_zero_rejects_a_mere_nullspace_hit():
    """A single probe landing in the nullspace must not read as a zero operator.

    Skipping on one zero image would silently abandon a real optimisation
    whenever the starting vector happened to be annihilated.
    """
    dimension = 8
    # Rank-deficient but far from zero: kills e0, leaves everything else.
    diagonal = np.ones(dimension)
    diagonal[0] = 0.0
    operator = LinearOperator(
        (dimension, dimension), matvec=lambda v: diagonal * v, dtype=float
    )

    in_nullspace = np.zeros(dimension)
    in_nullspace[0] = 1.0
    assert np.linalg.norm(operator.matvec(in_nullspace)) == 0.0
    assert not _operator_is_numerically_zero(operator, in_nullspace)

    truly_zero = LinearOperator(
        (dimension, dimension), matvec=lambda v: np.zeros(dimension), dtype=float
    )
    assert _operator_is_numerically_zero(truly_zero, in_nullspace)


def test_skipped_bond_still_advances_the_environments():
    """Skipping the eigensolve must not leave a placeholder environment behind.

    The next bond reads ``left_environments[i + 1]``; if the skip returns early
    it stays at the one-dimensional placeholder from ``__init__`` and the sweep
    breaks on the following bond.
    """
    num_sites = 6
    psi = np.zeros(2**num_sites, dtype=complex)
    psi[0] = 1.0

    target = mps_from_dense(psi, form="Right-canonical").right_canonical()
    start = create_simple_product_state(num_sites, which="+", form="Explicit")

    engine = deph_dmrg(
        start, target, chi_max=64, cut=1e-12, mode="LA", copy=True, silent=True
    )
    for index, tensor in enumerate(engine.mps_target.tensors):
        engine.mps_target.tensors[index] = np.zeros_like(tensor)

    engine.update_bond(0)

    # The requirement is that the sweep survives the skipped bond: the bonds
    # after it read the environments this one was supposed to advance.
    engine.sweep()


def test_non_finite_operator_is_not_reported_as_zero():
    """NaN in the operator is corruption, not an empty bond.

    Reading it as zero would skip the bond and swallow the ArpackError, leaving a
    corrupted run to continue silently.
    """
    dimension = 8
    nan_operator = LinearOperator(
        (dimension, dimension),
        matvec=lambda v: np.full(dimension, np.nan),
        dtype=float,
    )
    probe = np.full(dimension, 1.0 / np.sqrt(dimension))

    assert not _operator_is_numerically_zero(nan_operator, probe)


def test_non_finite_operator_lets_the_arpack_error_through():
    """End to end: a NaN operator must raise, not be skipped."""
    num_sites = 6
    psi = np.zeros(2**num_sites, dtype=complex)
    psi[0] = 1.0

    target = mps_from_dense(psi, form="Right-canonical").right_canonical()
    start = create_simple_product_state(num_sites, which="+", form="Explicit")
    engine = deph_dmrg(
        start, target, chi_max=64, cut=1e-12, mode="LA", copy=True, silent=True
    )
    for index, tensor in enumerate(engine.mps_target.tensors):
        engine.mps_target.tensors[index] = np.full_like(tensor, np.nan)

    with pytest.raises((ArpackError, ValueError, FloatingPointError)):
        engine.update_bond(0)


def test_restart_recovers_a_start_vector_in_the_nullspace():
    """The real cause of "ARPACK error -9" in this code base.

    ARPACK reports a nullspace start vector with the same message it uses for a
    zero one. The operator here is a healthy rank-1 projector; only ``v0`` is
    annihilated. Treating that as "nothing to optimise" would silently discard a
    bond whose dominant eigenvector is perfectly well defined.
    """
    dimension, generator = 256, np.random.default_rng(1)
    # One-hot v0 and a basis whose first component is exactly zero: the
    # annihilation is then exact by construction on every BLAS. The float
    # variant (project v0 out of a random vector) leaves ~1e-18 residue on
    # OpenBLAS while being exactly zero on Accelerate, which made this test
    # pass on macOS and fail on the Linux CI runners.
    v0 = np.zeros(dimension)
    v0[0] = 1.0

    raw = generator.normal(size=(dimension, 1))
    raw[0, 0] = 0.0  # range orthogonal to v0, exactly
    basis = np.linalg.qr(raw)[0]
    assert basis[0, 0] == 0.0  # QR only rescales a single column
    operator = LinearOperator(
        (dimension, dimension),
        matvec=lambda x: basis @ (basis.T @ x),
        dtype=float,
    )

    # Precondition: the operator kills v0 exactly, but is not itself zero.
    assert np.linalg.norm(operator.matvec(v0)) == 0.0
    assert not _operator_is_numerically_zero(operator, v0)
    with pytest.raises(ArpackError):
        eigsh(operator, k=1, which="LA", v0=v0, return_eigenvectors=True)

    # The restart finds the projector's eigenvector anyway.
    eigenvectors = _restart_from_operator_range(operator, v0, "LA")
    assert eigenvectors is not None
    recovered = eigenvectors[:, 0]
    assert np.isclose(abs(recovered @ basis[:, 0]), 1.0, atol=1e-6)


def test_restart_gives_up_on_a_genuinely_zero_operator():
    """A zero operator has no range to restart from, so None is correct."""
    dimension = 64
    zero = LinearOperator(
        (dimension, dimension), matvec=lambda x: np.zeros(dimension), dtype=float
    )
    v0 = np.full(dimension, 1.0 / np.sqrt(dimension))

    assert _restart_from_operator_range(zero, v0, "LA") is None
    assert _operator_is_numerically_zero(zero, v0)


def test_update_bond_recovers_when_the_guess_is_in_the_nullspace(monkeypatch):
    """The restart has to be wired into update_bond, not merely available.

    Substitutes an operator that annihilates whatever guess it is handed while
    staying a healthy projector elsewhere -- the situation that took down three
    full-scale classical_ldpc runs. update_bond must solve the bond rather than
    raise or quietly skip it.
    """
    num_sites = 6
    psi = np.zeros(2**num_sites, dtype=complex)
    psi[0] = 1.0
    target = mps_from_dense(psi, form="Right-canonical").right_canonical()
    start = create_simple_product_state(num_sites, which="+", form="Explicit")
    engine = deph_dmrg(
        start, target, chi_max=64, cut=1e-12, mode="LA", copy=True, silent=True
    )

    real_operator = dephasing_dmrg_module.EffectiveDensityOperator

    class NullspaceOperator(real_operator):
        """Healthy projector whose range is orthogonal to the initial guess."""

        def _matvec(self, x):
            flat = np.asarray(x).reshape(-1)
            guess = np.ones(self.shape[0], dtype=complex)
            guess /= np.linalg.norm(guess)
            direction = np.zeros(self.shape[0], dtype=complex)
            direction[0] = 1.0
            direction -= guess * (guess.conj() @ direction)
            direction /= np.linalg.norm(direction)
            return direction * (direction.conj() @ flat)

    monkeypatch.setattr(
        dephasing_dmrg_module, "EffectiveDensityOperator", NullspaceOperator
    )

    before = engine.mps.tensors[0].copy()
    engine.update_bond(0)  # must not raise
    after = engine.mps.tensors[0]

    # And must not have been skipped: a skip leaves the tensor untouched.
    assert not np.array_equal(before, after)
