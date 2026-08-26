"""Shared pytest fixtures.

Several test helpers (:func:`mdopt.mps.utils.create_state_vector`,
:func:`mdopt.utils.utils.create_random_mpo`) draw from the legacy global
``np.random`` stream when no explicit generator is passed. Without a seed those
tests are unreproducible: a failure triggered by a rare draw cannot be replayed.
"""

import zlib

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed_global_rng(request):
    """Seed the global ``np.random`` stream deterministically for every test.

    The seed is derived from the test's node id rather than being a single
    constant, so tests still see different draws from one another, but each one
    is reproducible on its own and independent of execution order (``-k``,
    ``-x`` and ``pytest-xdist`` shuffling all stay reproducible).
    """
    np.random.seed(zlib.crc32(request.node.nodeid.encode()) & 0xFFFFFFFF)


@pytest.fixture
def rng(request):
    """A seeded :class:`numpy.random.Generator` for tests that need one.

    ``seed_global_rng`` above only seeds the legacy ``np.random`` singleton, so a
    test that reaches for ``np.random.default_rng()`` still draws from fresh OS
    entropy and stays unreproducible. Ask for this fixture instead.
    """
    return np.random.default_rng(zlib.crc32(request.node.nodeid.encode()) & 0xFFFFFFFF)
