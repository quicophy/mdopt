"""The deprecated mdopt.examples.decoding.{decoding,dem} paths must still work.

Each shim warns on import and aliases its sys.modules entry to the moved
module, so every name -- public or private -- resolves to the same object.
"""

import importlib
import sys
import warnings

import pytest

LEGACY_TO_NEW = {
    "mdopt.examples.decoding.decoding": "mdopt.decoding.decoding",
    "mdopt.examples.decoding.dem": "mdopt.decoding.dem",
}
PROBE_SYMBOLS = {
    "mdopt.examples.decoding.decoding": (
        "decode_css",
        "decode_custom",
        "generate_pauli_error_string",
        "_score_tie",
        "_swap_pauli_components",
    ),
    "mdopt.examples.decoding.dem": (
        "DemProblem",
        "decode_dem",
        "dem_to_problem",
        "solve_representative",
        "_validated_syndrome",
        "_detector_parities",
    ),
}


@pytest.mark.parametrize("legacy", sorted(LEGACY_TO_NEW))
def test_legacy_path_warns_and_aliases_the_moved_module(legacy):
    new_name = LEGACY_TO_NEW[legacy]
    # Force a fresh import so the module body (and its warning) executes.
    sys.modules.pop(legacy, None)
    with pytest.warns(DeprecationWarning, match="moved to"):
        legacy_module = importlib.import_module(legacy)
    new_module = importlib.import_module(new_name)

    assert legacy_module is new_module
    assert sys.modules[legacy] is new_module
    for symbol in PROBE_SYMBOLS[legacy]:
        assert getattr(legacy_module, symbol) is getattr(new_module, symbol), symbol

    # A second import is served from sys.modules and must not warn again.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert importlib.import_module(legacy) is new_module
