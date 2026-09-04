"""Decoding of classical and quantum error-correcting codes with MPS.

This package provides the error-based decoders for CSS and custom stabiliser
codes (:mod:`mdopt.decoding.decoding`) and the detector-error-model decoder
for circuit-level noise (:mod:`mdopt.decoding.dem`).
"""

from mdopt.decoding.decoding import (
    apply_bitflip_bias,
    apply_depolarising_bias,
    css_code_logicals,
    css_code_stabilisers,
    decode_css,
    decode_custom,
    decode_message,
    generate_pauli_error_string,
    linear_code_constraint_sites,
    linear_code_prepare_message,
    pauli_to_mps,
)
from mdopt.decoding.dem import (
    DemProblem,
    decode_dem,
    dem_to_problem,
    solve_representative,
)
