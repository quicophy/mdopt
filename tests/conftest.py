"""Shared pytest fixtures.

Several test helpers (:func:`mdopt.mps.utils.create_state_vector`,
:func:`mdopt.utils.utils.create_random_mpo`) draw from the legacy global
``np.random`` stream when no explicit generator is passed. Without a seed those
tests are unreproducible: a failure triggered by a rare draw cannot be replayed.
"""

import zlib

import numpy as np
import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(autouse=True, scope="session")
def single_threaded_blas():
    """Run the suite on one BLAS thread.

    The wheels' multithreaded OpenBLAS makes the small QR and SVD
    factorizations the decoders live on up to a hundred times slower than
    single-threaded (an 800 x 400 QR takes seconds instead of tens of
    milliseconds), which the campaign scripts avoid by pinning the thread
    count and the test suite did not.
    """
    with threadpool_limits(limits=1, user_api="blas"):
        yield


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
