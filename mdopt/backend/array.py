"""
Array backend selection and helpers.
"""

import os
import importlib

# ----------------------------------------------------------------------
# Backend selection: default to NumPy; allow CUPY if available.
# ----------------------------------------------------------------------
_BACKEND_ENV = os.getenv("MDOPT_BACKEND", "numpy").lower()


def _load_backend():
    if _BACKEND_ENV == "cupy":
        try:
            return importlib.import_module("cupy")
        except (ImportError, ModuleNotFoundError):
            # Graceful fallback on machines without CuPy (e.g., macOS)
            pass
    return importlib.import_module("numpy")


_xp = _load_backend()

# Flag for quick checks elsewhere
GPU = _xp.__name__ == "cupy"


# ----------------------------------------------------------------------
# Introspection helpers
# ----------------------------------------------------------------------
def backend_name() -> str:
    """Return the active backend name: 'cupy' or 'numpy'."""
    return "cupy" if GPU else "numpy"


def is_cuda_backend() -> bool:
    """True iff the active backend is CuPy."""
    return GPU


# ----------------------------------------------------------------------
# Host/device transfer + streams
# ----------------------------------------------------------------------
if GPU:
    # CuPy-specific helpers
    from cupy import cuda as _cuda  # type: ignore

    def to_device(a):
        """Move/ensure array is on device."""
        return _xp.asarray(a)

    def to_host(a):
        """Move/ensure array is on host (NumPy)."""
        return _xp.asnumpy(a)

    def stream():
        """Return a non-blocking CUDA stream context manager."""
        return _cuda.Stream(non_blocking=True)

    def synchronize():
        """Synchronize the current CUDA device."""
        _cuda.Device().synchronize()

else:
    # NumPy "no-op" fallbacks
    def to_device(a):
        return _xp.asarray(a)

    def to_host(a):
        return a

    class _NullStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    def stream():
        return _NullStream()

    def synchronize():
        pass


# ----------------------------------------------------------------------
# Convenience wrappers (work for both backends)
# ----------------------------------------------------------------------
def einsum(expr, *args, **kw):
    return _xp.einsum(expr, *args, **kw)


def svd(x, full_matrices=False):
    # Expose a consistent SVD surface; for CuPy this is GPU-accelerated
    return _xp.linalg.svd(x, full_matrices=full_matrices)


def asfortran(a):
    return _xp.asfortranarray(a)


# ----------------------------------------------------------------------
# Module-level attribute forwarding
# This lets callers do: xp.asarray, xp.linalg.svd, xp.random, etc.
# ----------------------------------------------------------------------
def __getattr__(name):
    return getattr(_xp, name)
