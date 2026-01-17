"""Tests for the ``mdopt.optimiser.dephasing_dmrg`` module."""

import pytest
import numpy as np

from mdopt.optimiser.dephasing_dmrg import DephasingDMRG as deph_dmrg
from mdopt.optimiser.dmrg import DMRG as dmrg
from mdopt.optimiser.dephasing_dmrg import EffectiveDensityOperator
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
    If the implementation truly restricts the search domain to computational-basis bitstrings,
    it should never create entanglement: all bond dimensions must remain 1 after any run.

    This catches the failure mode you just saw (bond dims jumping to 2).
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
        mdpo = mps.density_mpo()

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

        # Finding the main component of the MDPO using DMRG.
        mps_start = create_simple_product_state(num_sites, which="+", form="Explicit")
        engine = dmrg(
            mps_start, mdpo, chi_max=1e4, cut=1e-12, mode="LA", copy=True, silent=True
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

        # Finding the main component of the MDPO using dephasing DMRG.
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
    Regression/bug-catcher:
    target = |+>^{⊗n}. The dephased MDPO is maximally mixed (huge degeneracy).
    A solver that *truly searches only over computational basis bitstrings*
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
