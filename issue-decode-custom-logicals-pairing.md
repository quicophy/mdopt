# `decode_custom` logical readout uses a non-gauge-invariant pairing (missing X↔Z swap in `custom_code_logicals`)

## Description

`custom_code_checks` swaps the two component bits of every qubit when placing
**stabilizer** constraint strings, with a comment explaining why:

> A stabiliser's syndrome is a symplectic product: its Z part detects X errors
> and its X part detects Z errors. The parity check therefore acts on the
> *opposite* component to the one `pauli_to_mps` would assign […] for a
> self-dual code such as Steane the swap is invisible, which is why it went
> unnoticed.

`custom_code_logicals` applies **no such swap**: it places each logical
constraint on the *same* components as the logical operator's own Pauli string
(`bitstring = pauli_to_mps(logical)`, then `np.nonzero` directly).

That pairing is not gauge-invariant. The logical site is supposed to record
the residual's logical class, which requires a pairing that annihilates every
stabilizer (so that all stabilizer-equivalent residuals in one coset map to
the same logical label). The symplectic (swapped) pairing has this property;
the same-component pairing does not. With the unswapped pairing, gauge copies
of the same residual class are scattered across different logical labels, the
class amplitudes are diluted/scrambled, and the MAP readout can flip — even at
fully converged bond dimension.

As with the stabilizer half of this bug, self-dual codes mask it, and
`decode_css` is unaffected (its logical constraints already pair X-class
readout with the Z̄-support components — cf. the Shor-code walkthrough in the
thesis, where the logical-X constraint acts on the X-components along the
Z̄-type block support).

## Observed effect

Cross-checking `decode_custom` against an independent exact-MAP tensor-network
decoder on the qldpc **rotated surface-5 code** (n = 25, depolarizing
p = 0.01, 1000 sampled errors, `chi_max` ∈ {32, 128}, `renormalise=True`):

- `decode_custom` failed **7/1000 shots — including 6 weight-1 errors**
  (X on qubit 20 ×3, Z on qubit 23 ×2, Z on qubit 1), deterministically:
  the same shots fail at χ=32 and χ=128, so this is converged behavior, not
  truncation. A distance-5 code must correct all weight-1 errors under MAP.
- Exact contraction (probability semantics *and* mdopt's amplitude semantics —
  both checked) decodes all 1000 shots correctly, ruling out the
  amplitude-vs-probability marginalization as the cause.
- Feeding `decode_custom` **component-swapped logical strings** (X↔Z per
  character, supports unchanged) fixes all 7 failures while leaving 40
  non-trivial control shots correct.

## Suggested fix

In `custom_code_logicals` (mdopt/examples/decoding/decoding.py), apply the
same two-bit swap to the logical bitstrings that `custom_code_checks` applies
to the stabilizer bitstrings:

```python
bitstring = pauli_to_mps(logical)
bitstring = "".join(
    bitstring[i + 1] + bitstring[i] for i in range(0, len(bitstring), 2)
)
```

(in both the X-logical and Z-logical loops), or factor the swap into a shared
helper used by both functions.

## Regression test suggestion

Decode all weight-1 errors of a non-self-dual CSS code (e.g., the rotated
surface-5) through `decode_custom` at moderate `chi_max` and assert 100%
success — this fails before the fix and passes after.

*(Found 2026-08-29 while cross-validating mdopt against the TNDecoding
syndrome-frame decoder on identical sampled errors.)*
