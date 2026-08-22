"""
Array backend selection and helpers.
"""

import os
import importlib
import warnings

# ----------------------------------------------------------------------
# Backend selection: default to NumPy; allow CUPY if available.
# ----------------------------------------------------------------------
_BACKEND_ENV = os.getenv("MDOPT_BACKEND", "numpy").lower()


def _cuda_device_available(cupy) -> bool:
    """
    Whether CuPy can actually reach a CUDA device.

    Importing CuPy succeeds on machines with no GPU or an outdated driver --
    the failure only surfaces on the first device call. Probing here keeps
    :data:`GPU` meaning "GPU usable" rather than "CuPy importable", so we fall
    back to NumPy instead of raising from deep inside a contraction.
    """
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pylint: disable=broad-except
        # Any driver/runtime problem (e.g. cudaErrorInsufficientDriver) means
        # there is no device we can use.
        return False


def _load_backend():
    if _BACKEND_ENV == "cupy":
        try:
            cupy = importlib.import_module("cupy")
        except (ImportError, ModuleNotFoundError):
            # Graceful fallback on machines without CuPy (e.g., macOS)
            warnings.warn(
                "MDOPT_BACKEND=cupy was requested but CuPy is not installed; "
                "falling back to the NumPy backend.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if _cuda_device_available(cupy):
                return cupy
            warnings.warn(
                "MDOPT_BACKEND=cupy was requested and CuPy imported, but no "
                "usable CUDA device was found; falling back to the NumPy "
                "backend.",
                RuntimeWarning,
                stacklevel=2,
            )
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
    if hasattr(_xp, name):
        return getattr(_xp, name)
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"The current backend ('{_xp.__name__}') also does not have this attribute. "
        "This may be due to a typo or an API change."
    )
