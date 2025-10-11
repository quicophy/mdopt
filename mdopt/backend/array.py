import os

_backend = os.getenv("MDOPT_BACKEND", "numpy").lower()

if _backend == "cupy":
    import cupy as xp
    from cupy.linalg import svd as xp_svd
    from cupy import cuda

    def to_device(a):
        return xp.asarray(a)

    def to_host(a):
        return xp.asnumpy(a)

    def stream():
        return cuda.Stream(non_blocking=True)

    def synchronize():
        cuda.Device().synchronize()

    GPU = True
else:
    import numpy as xp
    from numpy.linalg import svd as xp_svd

    def to_device(a):
        return a

    def to_host(a):
        return a

    class _Null:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def stream():
        return _Null()

    def synchronize():
        pass

    GPU = False


def einsum(expr, *args, **kw):  # delegate to xp
    return xp.einsum(expr, *args, **kw)


def svd(x, full_matrices=False):
    return xp_svd(x, full_matrices=full_matrices)


def asfortran(a):
    # CuPy/NumPy both expose asfortranarray
    return xp.asfortranarray(a)
