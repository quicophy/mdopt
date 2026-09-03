"""Deprecated location: this module moved to :mod:`mdopt.decoding.dem`.

Importing from here keeps working (this module aliases itself to the new one,
so every name -- public or private -- resolves identically), but new code
should import from ``mdopt.decoding.dem``.
"""

import sys
import warnings

import mdopt.decoding.dem as _moved

warnings.warn(
    "mdopt.examples.decoding.dem has moved to mdopt.decoding.dem; "
    "this import path is deprecated and will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
