"""Decoding of classical and quantum error-correcting codes with MPS.

This package provides the error-based decoders for CSS and custom stabilizer
codes (:mod:`mdopt.decoding.decoding`), the detector-error-model decoder for
circuit-level noise (:mod:`mdopt.decoding.dem`) and the class-sum decoder that
resums the posterior around BP-OSD candidates (:mod:`mdopt.decoding.class_sum`).
"""

from mdopt.decoding.class_sum import (
    ClassSumDecoder,
    ParityChain,
    active_set_class_masses,
    multiplet_safe_rank,
    problem_matrices,
)
from mdopt.decoding.decoding import (
    apply_bitflip_bias,
    apply_depolarizing_bias,
    css_code_logicals,
    css_code_stabilizers,
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
