"""Tests for backend selection in :mod:`mdopt.backend.array`."""

import importlib
import sys
import types
import warnings

import pytest


def _reload_backend(monkeypatch, backend_env, fake_cupy):
    """Import a fresh copy of the backend module under a controlled environment."""
    for name in [n for n in sys.modules if n.startswith("mdopt")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "cupy", raising=False)

    if fake_cupy is None:
        monkeypatch.setitem(sys.modules, "cupy", None)
    else:
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    if backend_env is None:
        monkeypatch.delenv("MDOPT_BACKEND", raising=False)
    else:
        monkeypatch.setenv("MDOPT_BACKEND", backend_env)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("mdopt.backend.array")
    return module, caught


def _fake_cupy(num_devices=0, raises=False):
    """A stand-in for CuPy that reports a given number of CUDA devices."""
    module = types.ModuleType("cupy")

    def get_device_count():
        if raises:
            raise RuntimeError("cudaErrorInsufficientDriver")
        return num_devices

    runtime = types.SimpleNamespace(getDeviceCount=get_device_count)
    module.cuda = types.SimpleNamespace(runtime=runtime, Stream=object, Device=object)
    module.asarray = lambda a: a
    return module


def test_defaults_to_numpy(monkeypatch):
    """With no MDOPT_BACKEND set, NumPy is used and nothing is warned about."""
    module, caught = _reload_backend(monkeypatch, None, None)
    assert module.GPU is False
    assert module.backend_name() == "numpy"
    assert not caught


def test_falls_back_when_cupy_missing(monkeypatch):
    """Requesting CuPy without it installed warns and falls back to NumPy."""
    module, caught = _reload_backend(monkeypatch, "cupy", None)
    assert module.GPU is False
    assert any("not installed" in str(w.message) for w in caught)


def test_falls_back_when_no_cuda_device(monkeypatch):
    """CuPy imports fine without a GPU, so the device count decides."""
    module, caught = _reload_backend(monkeypatch, "cupy", _fake_cupy(num_devices=0))
    assert module.GPU is False
    assert module.backend_name() == "numpy"
    assert any("no usable CUDA device" in str(w.message) for w in caught)


def test_falls_back_when_driver_raises(monkeypatch):
    """A driver too old to serve the runtime must not select the GPU backend."""
    module, caught = _reload_backend(monkeypatch, "cupy", _fake_cupy(raises=True))
    assert module.GPU is False
    assert any("no usable CUDA device" in str(w.message) for w in caught)


def test_uses_cupy_when_device_present(monkeypatch):
    """A usable device must still select the CuPy backend, without warning."""
    module, caught = _reload_backend(monkeypatch, "cupy", _fake_cupy(num_devices=1))
    assert module.GPU is True
    assert module.backend_name() == "cupy"
    assert not caught


def test_stream_is_a_context_manager_on_numpy(monkeypatch):
    """The NumPy fallback still provides the stream()/synchronize() surface."""
    module, _ = _reload_backend(monkeypatch, None, None)
    with module.stream():
        pass
    module.synchronize()
