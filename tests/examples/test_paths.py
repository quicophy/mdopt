"""Tests for locating the example assets that live outside the package."""

import pathlib

import pytest

from mdopt.examples import paths


def test_assets_root_resolves_inside_a_checkout():
    """In a working copy the assets are the repository's top-level examples/."""
    root = paths.assets_root()
    assert root.is_dir()
    assert root.name == "examples"
    assert (root.parent / "pyproject.toml").is_file()


def test_assets_root_is_not_the_package_directory():
    """The lookup anchors on pyproject.toml, not on the directory name.

    ``mdopt/examples`` is itself called "examples", so matching on the name
    alone would resolve to the package that contains this module and silently
    read assets from inside the wheel.
    """
    package_dir = pathlib.Path(paths.__file__).resolve().parent
    assert paths.assets_root() != package_dir
    assert not paths.assets_root().is_relative_to(package_dir)


def test_environment_override_is_honoured(tmp_path, monkeypatch):
    """A cluster can point the lookup at scratch storage."""
    (tmp_path / "decoding").mkdir()
    monkeypatch.setenv("MDOPT_EXAMPLES_ASSETS", str(tmp_path))
    assert paths.assets_root() == tmp_path.resolve()


def test_an_override_without_decoding_is_rejected(tmp_path, monkeypatch):
    """A typo that names a real directory must fail, not redirect the output.

    Without this check figures_dir() happily creates `decoding/figures` inside
    whatever the variable points at, so a mistyped path silently scatters output
    somewhere unrelated instead of failing.
    """
    monkeypatch.setenv("MDOPT_EXAMPLES_ASSETS", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no 'decoding' subdirectory"):
        paths.assets_root()
    assert not (tmp_path / "decoding").exists(), "must not have created anything"


def test_a_bad_override_fails_loudly(tmp_path, monkeypatch):
    """Pointing at a missing directory must raise rather than fall back.

    Falling back to the checkout would silently read the wrong data on a
    cluster, which is worse than not running at all.
    """
    monkeypatch.setenv("MDOPT_EXAMPLES_ASSETS", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="MDOPT_EXAMPLES_ASSETS"):
        paths.assets_root()


def test_missing_dataset_names_itself(tmp_path, monkeypatch):
    """The datasets are untracked, so the error has to say what is missing."""
    monkeypatch.setenv("MDOPT_EXAMPLES_ASSETS", str(tmp_path))
    (tmp_path / "decoding").mkdir()
    with pytest.raises(FileNotFoundError, match="data-quantum-surface"):
        paths.data_dir("data-quantum-surface")
    # ...unless the caller says it may be absent.
    assert paths.data_dir("data-quantum-surface", required=False).name == (
        "data-quantum-surface"
    )


def test_figure_paths_are_absolute_and_flat(tmp_path, monkeypatch):
    """figure() must ignore any directory part of the name it is handed.

    Scripts pass bare file names; letting a stray path component through would
    scatter output outside the figures directory.
    """
    monkeypatch.setenv("MDOPT_EXAMPLES_ASSETS", str(tmp_path))
    (tmp_path / "decoding").mkdir()
    target = paths.figure("../../escape.pdf")
    assert target.is_absolute()
    assert target.parent == paths.figures_dir()
    assert target.name == "escape.pdf"


def test_figures_dir_is_created_on_demand(tmp_path, monkeypatch):
    monkeypatch.setenv("MDOPT_EXAMPLES_ASSETS", str(tmp_path))
    (tmp_path / "decoding").mkdir()
    assert not (tmp_path / "decoding" / "figures").exists()
    assert paths.figures_dir().is_dir()
