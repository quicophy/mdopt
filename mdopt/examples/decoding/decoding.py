"""Deprecated location: this module moved to :mod:`mdopt.decoding.decoding`.

Importing from here keeps working (this module aliases itself to the new one,
so every name -- public or private -- resolves identically), but new code
should import from ``mdopt.decoding.decoding``.
"""

import sys
import warnings

import mdopt.decoding.decoding as _moved

warnings.warn(
    "mdopt.examples.decoding.decoding has moved to mdopt.decoding.decoding; "
    "this import path is deprecated and will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
