"""The decoding drivers' prior-reweighting and certificate options.

The drivers (bivariate bicycle, CSP, surface) expose ``--prior_reweighting``
and ``--store_certificate``. The arguments sit at the end of every signature
because positional callers exist, the certificate lands in the pickle as
per-shot ``survival`` and ``bound`` arrays, and a reweighted run is tagged in
the file name so plotting globs never pool it with plain data.
"""

import importlib
import inspect
import pickle

import numpy as np
import pytest

DRIVERS = ["quantum_bivariate_bicycle", "quantum_csp", "quantum_surface"]


def _driver(name):
    return importlib.import_module(f"mdopt.examples.decoding.{name}")


@pytest.mark.parametrize("name", DRIVERS)
def test_new_arguments_are_last_in_every_driver_signature(name):
    """Inserting them earlier would shift the positional callers."""
    driver = _driver(name)
    for function, tail in [
        ("run_single_experiment", ["prior_reweighting", "store_certificate"]),
        ("run_experiment", ["prior_reweighting", "store_certificate"]),
        ("save_experiment_data", ["prior_reweighting"]),
    ]:
        names = list(inspect.signature(getattr(driver, function)).parameters)
        assert names[-len(tail) :] == tail, f"{name}.{function}: {names}"
        defaults = inspect.signature(getattr(driver, function)).parameters
        assert defaults["prior_reweighting"].default is None
        if "store_certificate" in tail:
            assert defaults["store_certificate"].default is False


@pytest.mark.parametrize("name", DRIVERS)
def test_certificate_values_accepts_both_decoder_shapes(name):
    driver = _driver(name)

    class Certificate:
        survival = 0.25
        bound = 3.0

    assert driver.certificate_values(([1.0, 0.0], 1)) == ([1.0, 0.0], 1, np.nan, np.nan) or (
        driver.certificate_values(([1.0, 0.0], 1))[:2] == ([1.0, 0.0], 1)
        and all(np.isnan(v) for v in driver.certificate_values(([1.0, 0.0], 1))[2:])
    )
    assert driver.certificate_values(([1.0, 0.0], 0, Certificate())) == (
        [1.0, 0.0],
        0,
        0.25,
        3.0,
    )
    assert driver.prior_reweighting_choice("None") is None
    assert driver.prior_reweighting_choice("BP") == "BP"


def _surface_run(tmp_path, monkeypatch, **options):
    """A tiny L=3 surface-code run in a scratch cwd; returns (data, file name)."""
    surface = _driver("quantum_surface")
    monkeypatch.chdir(tmp_path)
    lattice_size, chi, p, bias, shots, seed = 3, 16, 0.05, 1e-3, 4, 7
    errors = surface.generate_errors(lattice_size, p, shots, "Bitflip", seed)
    data = surface.run_experiment(
        lattice_size,
        chi,
        p,
        bias,
        shots,
        "Bitflip",
        seed,
        errors,
        True,
        1,
        0,
        0,
        **options,
    )
    surface.save_experiment_data(
        data,
        lattice_size,
        chi,
        p,
        "Bitflip",
        bias,
        shots,
        seed,
        0,
        0,
        **{k: v for k, v in options.items() if k == "prior_reweighting"},
    )
    files = sorted(f.name for f in tmp_path.glob("*.pkl"))
    assert len(files) == 1
    with open(tmp_path / files[0], "rb") as handle:
        stored = pickle.load(handle)
    return stored, files[0]


def test_reweighted_run_stores_certificate_arrays_and_tags_the_file(
    tmp_path, monkeypatch
):
    stored, file_name = _surface_run(
        tmp_path, monkeypatch, prior_reweighting="BP", store_certificate=True
    )
    assert file_name.endswith("_seed7_priorBP.pkl")
    assert stored["prior_reweighting"] == "BP"
    assert len(stored["failures"]) == 4
    for key in ("survival", "bound"):
        assert isinstance(stored[key], np.ndarray)
        assert stored[key].shape == (4,)
        assert np.all(np.isfinite(stored[key]))
    # Survival is a mass fraction the checks kept; a trivial shot keeps all of it.
    assert np.all(stored["survival"] > 0) and np.all(stored["survival"] <= 1 + 1e-9)
    assert np.all(stored["bound"] >= 0)


def test_plain_run_is_unchanged_by_the_new_options(tmp_path, monkeypatch):
    stored, file_name = _surface_run(tmp_path, monkeypatch)
    assert file_name.endswith("_seed7.pkl")
    assert stored["prior_reweighting"] is None
    assert "survival" not in stored and "bound" not in stored
    assert len(stored["failures"]) == 4
